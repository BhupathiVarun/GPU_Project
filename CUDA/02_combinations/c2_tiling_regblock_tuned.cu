
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <cuda_runtime.h>

using namespace std;

#ifndef C2_TILE_SIZE
#define C2_TILE_SIZE 16
#endif

static_assert(C2_TILE_SIZE == 8 || C2_TILE_SIZE == 16 || C2_TILE_SIZE == 32,
              "C2_TILE_SIZE must match a T4-tested value: 8, 16, or 32");
static_assert(C2_TILE_SIZE * C2_TILE_SIZE <= 1024,
              "C2 block contains more than 1024 threads");

#define TIMED_RUNS 5
#define ERROR_TOLERANCE 1.0e-9

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            cerr << "CUDA Error: " << cudaGetErrorString(err__)            \
                 << " at line " << __LINE__ << endl;                       \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


template <int RM, int RN>
__global__ void __launch_bounds__(C2_TILE_SIZE * C2_TILE_SIZE)
matmulTiledRegBlocked(
    const double *__restrict__ A,
    const double *__restrict__ B,
    double *__restrict__ C,
    int N)
{
    constexpr int S = C2_TILE_SIZE;
    constexpr int BK = C2_TILE_SIZE;
    constexpr int OUT_M = S * RM;
    constexpr int OUT_N = S * RN;
    constexpr int AS_STRIDE = BK + 1;
    constexpr int BS_STRIDE = OUT_N + 1;
    constexpr int THREAD_COUNT = S * S;

    // Dynamic shared memory lets the host skip oversized variants at runtime
    // rather than causing compilation to fail for a large 8x8 specialization.
    extern __shared__ double shared[];
    double *As = shared;
    double *Bs = As + static_cast<size_t>(OUT_M) * AS_STRIDE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * S + tx;

    const int rowBlockBase = blockIdx.y * OUT_M;
    const int colBlockBase = blockIdx.x * OUT_N;

    // RM*RN accumulators are intended to stay in registers.
    double acc[RM][RN];

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    const int kTiles = (N + BK - 1) / BK;

    for (int tile = 0; tile < kTiles; ++tile) {
        const int kBase = tile * BK;

        // 1. Load A tile: OUT_M x BK. Threads cooperatively traverse the
        //    row-major tile; out-of-range matrix elements are zero-filled.
        const int aElements = OUT_M * BK;
        for (int idx = tid; idx < aElements; idx += THREAD_COUNT) {
            const int localRow = idx / BK;
            const int localK = idx % BK;
            const int globalRow = rowBlockBase + localRow;
            const int globalK = kBase + localK;

            As[static_cast<size_t>(localRow) * AS_STRIDE + localK] =
                (globalRow < N && globalK < N)
                    ? A[static_cast<size_t>(globalRow) * N + globalK]
                    : 0.0;
        }

        // 2. Load B tile: BK x OUT_N.
        const int bElements = BK * OUT_N;
        for (int idx = tid; idx < bElements; idx += THREAD_COUNT) {
            const int localK = idx / OUT_N;
            const int localCol = idx % OUT_N;
            const int globalK = kBase + localK;
            const int globalCol = colBlockBase + localCol;

            Bs[static_cast<size_t>(localK) * BS_STRIDE + localCol] =
                (globalK < N && globalCol < N)
                    ? B[static_cast<size_t>(globalK) * N + globalCol]
                    : 0.0;
        }

        // All threads must finish loading before any thread reads shared tiles.
        __syncthreads();

        // 3. Register-blocked dot products. For each k, load RM A values and
        //    RN B values from shared memory, then reuse them for RM*RN FMAs.
        #pragma unroll 1
        for (int kk = 0; kk < BK; ++kk) {
            double aValues[RM];
            double bValues[RN];

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                const int localRow = ty + i * S;
                aValues[i] = As[static_cast<size_t>(localRow) * AS_STRIDE + kk];
            }

            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int localCol = tx + j * S;
                bValues[j] = Bs[static_cast<size_t>(kk) * BS_STRIDE + localCol];
            }

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    acc[i][j] += aValues[i] * bValues[j];
                }
            }
        }

        // Do not overwrite shared tiles until every thread has consumed them.
        __syncthreads();
    }

    // 4. Store each thread's RM x RN outputs. Every valid C element is written
    //    exactly once; partial edge tiles are masked.
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int globalRow = rowBlockBase + ty + i * S;

        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int globalCol = colBlockBase + tx + j * S;

            if (globalRow < N && globalCol < N) {
                C[static_cast<size_t>(globalRow) * N + globalCol] = acc[i][j];
            }
        }
    }
}