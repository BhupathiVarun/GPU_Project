
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;
using namespace nvcuda;

#ifndef FINAL_BM
#define FINAL_BM 16
#endif
#ifndef FINAL_BN
#define FINAL_BN 16
#endif
#ifndef FINAL_BK
#define FINAL_BK 16
#endif
#ifndef FINAL_RM
#define FINAL_RM 2
#endif
#ifndef FINAL_RN
#define FINAL_RN 2
#endif

#define FINAL_VALID_T4_CONFIG ( \
    (FINAL_BM == 8  && FINAL_BN == 8  && FINAL_BK == 8)  || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 8)  || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 16) || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 32) || \
    (FINAL_BM == 8  && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 16 && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 32 && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 32 && FINAL_BN == 32 && FINAL_BK == 32) )

static_assert(FINAL_VALID_T4_CONFIG,
              "FINAL_BM/BN/BK must match one of the eight full T4 configurations");
static_assert(FINAL_BM * FINAL_BN <= 1024,
              "FINAL_BM * FINAL_BN exceeds CUDA's 1024-thread block limit");
static_assert((FINAL_RM == 1 || FINAL_RM == 2 || FINAL_RM == 4 || FINAL_RM == 8),
              "FINAL_RM must be one of C2's tested values: 1, 2, 4, 8");
static_assert((FINAL_RN == 1 || FINAL_RN == 2 || FINAL_RN == 4 || FINAL_RN == 8),
              "FINAL_RN must be one of C2's tested values: 1, 2, 4, 8");

#define TIMED_RUNS 20
#define WARMUP_RUNS 2
#define FP64_TOLERANCE 1.0e-9
#define TENSOR_TOLERANCE 1.0e-2

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            cerr << "CUDA error: " << cudaGetErrorString(err__)              \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status__ = (call);                                    \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                             \
            cerr << "cuBLAS error (status " << static_cast<int>(status__)    \
                 << ") at " << __FILE__ << ":" << __LINE__ << endl;         \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


template <int BM, int BN, int BK, int RM, int RN>
__global__ void __launch_bounds__(BM * BN)
matmulFinalFP64(const double *__restrict__ A,
                const double *__restrict__ B,
                double *__restrict__ C,
                int N)
{
    static_assert(BM * BN <= 1024, "too many threads in one block");

    constexpr int OUT_M = BM * RM;
    constexpr int OUT_N = BN * RN;
    constexpr int AS_STRIDE = BK + 1;       // padded shared-memory row
    constexpr int BS_STRIDE = OUT_N + 1;    // padded shared-memory row
    constexpr int THREADS = BM * BN;

    extern __shared__ double shared[];
    double *As = shared;
    double *Bs = As + static_cast<size_t>(OUT_M) * AS_STRIDE;

    const int tx = threadIdx.x;             // fastest-changing lane coordinate
    const int ty = threadIdx.y;
    const int tid = ty * BN + tx;

    const int rowBase = blockIdx.y * OUT_M;
    const int colBase = blockIdx.x * OUT_N;

    // RM*RN independent partial sums. These are intended to stay in registers.
    double accum[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            accum[i][j] = 0.0;
        }
    }

    const int kTiles = (N + BK - 1) / BK;

    for (int tile = 0; tile < kTiles; ++tile) {
        const int kBase = tile * BK;

        // Load A tile (OUT_M x BK). Consecutive flattened indices map to
        // consecutive k positions within each row.
        const int aElements = OUT_M * BK;
        for (int idx = tid; idx < aElements; idx += THREADS) {
            const int localRow = idx / BK;
            const int localK = idx - localRow * BK;
            const int globalRow = rowBase + localRow;
            const int globalK = kBase + localK;

            As[static_cast<size_t>(localRow) * AS_STRIDE + localK] =
                (globalRow < N && globalK < N)
                    ? A[static_cast<size_t>(globalRow) * N + globalK]
                    : 0.0;
        }

        // Load B tile (BK x OUT_N). Consecutive flattened indices map to
        // consecutive output columns, producing coalesced row-major accesses.
        const int bElements = BK * OUT_N;
        for (int idx = tid; idx < bElements; idx += THREADS) {
            const int localK = idx / OUT_N;
            const int localCol = idx - localK * OUT_N;
            const int globalK = kBase + localK;
            const int globalCol = colBase + localCol;

            Bs[static_cast<size_t>(localK) * BS_STRIDE + localCol] =
                (globalK < N && globalCol < N)
                    ? B[static_cast<size_t>(globalK) * N + globalCol]
                    : 0.0;
        }

        __syncthreads();

        // Register blocking: RM A values and RN B values feed RM*RN FMAs.
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double aValues[RM];
            double bValues[RN];

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                const int localRow = ty + i * BM;
                aValues[i] = As[static_cast<size_t>(localRow) * AS_STRIDE + kk];
            }

            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int localCol = tx + j * BN;
                bValues[j] = Bs[static_cast<size_t>(kk) * BS_STRIDE + localCol];
            }

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    accum[i][j] = fma(aValues[i], bValues[j], accum[i][j]);
                }
            }
        }

        // No thread may overwrite As/Bs until every thread has finished using
        // the previous K tile.
        __syncthreads();
    }

    // Unique ownership: each valid C element is written by exactly one thread.
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int globalRow = rowBase + ty + i * BM;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int globalCol = colBase + tx + j * BN;
            if (globalRow < N && globalCol < N) {
                C[static_cast<size_t>(globalRow) * N + globalCol] = accum[i][j];
            }
        }
    }
}
