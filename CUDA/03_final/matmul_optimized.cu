
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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int TC_TILE_M = 64;
constexpr int TC_TILE_N = 64;
constexpr int TC_TILE_K = 64;
constexpr int TC_PAD = 8;
constexpr int TC_THREADS = 128;

using WmmaFragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::row_major>;
using WmmaFragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::row_major>;
using WmmaFragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                                 float>;

__global__ void __launch_bounds__(TC_THREADS)
matmulFinalWMMA(const __half *__restrict__ A,
                const __half *__restrict__ B,
                float *__restrict__ C,
                int N)
{
    __shared__ __align__(128) __half As[TC_TILE_M][TC_TILE_K + TC_PAD];
    __shared__ __align__(128) __half Bs[TC_TILE_K][TC_TILE_N + TC_PAD];

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int blockRow = blockIdx.y * TC_TILE_M;
    const int blockCol = blockIdx.x * TC_TILE_N;

    WmmaFragC accum[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(accum[i][j], 0.0f);
        }
    }

    for (int kBase = 0; kBase < N; kBase += TC_TILE_K) {
        // Vectorized 16-byte global loads: eight consecutive FP16 values.
        for (int idx = tid; idx < TC_TILE_M * TC_TILE_K / 8;
             idx += TC_THREADS) {
            const int r = idx / (TC_TILE_K / 8);
            const int c = (idx % (TC_TILE_K / 8)) * 8;
            *reinterpret_cast<int4 *>(&As[r][c]) =
                *reinterpret_cast<const int4 *>(
                    &A[static_cast<size_t>(blockRow + r) * N + kBase + c]);
        }

        for (int idx = tid; idx < TC_TILE_K * TC_TILE_N / 8;
             idx += TC_THREADS) {
            const int r = idx / (TC_TILE_N / 8);
            const int c = (idx % (TC_TILE_N / 8)) * 8;
            *reinterpret_cast<int4 *>(&Bs[r][c]) =
                *reinterpret_cast<const int4 *>(
                    &B[static_cast<size_t>(kBase + r) * N + blockCol + c]);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TC_TILE_K; kk += WMMA_K) {
            WmmaFragA a[2];
            WmmaFragB b[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(
                    a[i], &As[warpRow * 32 + i * WMMA_M][kk],
                    TC_TILE_K + TC_PAD);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(
                    b[j], &Bs[kk][warpCol * 32 + j * WMMA_N],
                    TC_TILE_N + TC_PAD);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(accum[i][j], a[i], b[j], accum[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(
                C + static_cast<size_t>(blockRow + warpRow * 32 + i * WMMA_M) * N
                  + blockCol + warpCol * 32 + j * WMMA_N,
                accum[i][j], N, wmma::mem_row_major);
        }
    }
}


struct Timing {
    double medianMs = 0.0;
    double averageMs = 0.0;
    double gflops = 0.0;
};

struct ErrorStats {
    double maxAbs = 0.0;
    double rmse = 0.0;
};

template <typename Launch>
static Timing benchmark(const string &name,
                        Launch launch,
                        double operations,
                        cudaEvent_t start,
                        cudaEvent_t stop)
{
    // Warm-up runs are not included in timing samples.
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> samples;
    samples.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    sort(samples.begin(), samples.end());
    if (samples.size() % 2 == 0) {
        const size_t hi = samples.size() / 2;
        const size_t lo = hi - 1;
        Timing t;
        t.medianMs = 0.5 * (samples[lo] + samples[hi]);
        for (float x : samples) t.averageMs += x;
        t.averageMs /= samples.size();
        t.gflops = (t.medianMs > 0.0)
                 ? operations / (t.medianMs * 1.0e6) : 0.0;
        return t;
    }

    Timing t;
    t.medianMs = samples[samples.size() / 2];
    for (float x : samples) t.averageMs += x;
    t.averageMs /= samples.size();
    t.gflops = (t.medianMs > 0.0)
             ? operations / (t.medianMs * 1.0e6) : 0.0;
    (void)name;
    return t;
}

static void cpuMatrixMultiply(const vector<double> &A,
                              const vector<double> &B,
                              vector<double> &C,
                              int N)
{
    fill(C.begin(), C.end(), 0.0);
    // i-k-j order improves CPU cache locality. For each output element, k is
    // still accumulated in ascending order, matching the CUDA reduction order.
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            const double a = A[static_cast<size_t>(i) * N + k];
            const size_t bRow = static_cast<size_t>(k) * N;
            const size_t cRow = static_cast<size_t>(i) * N;
            for (int j = 0; j < N; ++j) {
                C[cRow + j] += a * B[bRow + j];
            }
        }
    }
}

static ErrorStats compareDouble(const vector<double> &actual,
                                const vector<double> &reference)
{
    ErrorStats e;
    double sumSq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = fabs(actual[i] - reference[i]);
        e.maxAbs = max(e.maxAbs, d);
        sumSq += d * d;
    }
    e.rmse = sqrt(sumSq / max<size_t>(1, actual.size()));
    return e;
}

static ErrorStats compareFloatToDouble(const vector<float> &actual,
                                       const vector<double> &reference,
                                       int N,
                                       int paddedN)
{
    ErrorStats e;
    double sumSq = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double d = fabs(static_cast<double>(
                actual[static_cast<size_t>(i) * paddedN + j])
                - reference[static_cast<size_t>(i) * N + j]);
            e.maxAbs = max(e.maxAbs, d);
            sumSq += d * d;
        }
    }
    e.rmse = sqrt(sumSq / max<size_t>(1, static_cast<size_t>(N) * N));
    return e;
}

template <typename CopyInputs>
static double measureH2D(CopyInputs copyInputs,
                         cudaEvent_t start,
                         cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));
    copyInputs();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}