#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;


#define BLOCK_X 32   // one warp wide: lanes map to consecutive columns
#define BLOCK_Y 8    // same 32 x 8 block as the best coalesced variant (t3)

static_assert(BLOCK_X == 32, "BLOCK_X must equal the warp size (32)");
static_assert(BLOCK_X * BLOCK_Y <= 1024, "block exceeds 1024 threads");

// Check CUDA errors
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA Error: " << cudaGetErrorString(err)     \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

template <int RM, int RN>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
matmulRegBlocked(
    const double *A,
    const double *B,
    double *C,
    int N)
{
    const int rowBase = blockIdx.y * (BLOCK_Y * RM) + threadIdx.y;
    const int colBase = blockIdx.x * (BLOCK_X * RN) + threadIdx.x;

    // Row offsets of A and column indices of B, clamped so loads stay in range.
    int rowOff[RM];
    int colIdx[RN];

    #pragma unroll
    for (int i = 0; i < RM; i++)
    {
        int r = rowBase + i * BLOCK_Y;
        rowOff[i] = (r < N ? r : N - 1) * N;
    }

    #pragma unroll
    for (int j = 0; j < RN; j++)
    {
        int c = colBase + j * BLOCK_X;
        colIdx[j] = (c < N ? c : N - 1);
    }

    // RM * RN accumulators live in registers for the whole k loop.
    double acc[RM][RN];

    #pragma unroll
    for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++)
            acc[i][j] = 0.0;

    for (int k = 0; k < N; k++)
    {
        double a[RM];
        double b[RN];

        // RM + RN global loads ...
        #pragma unroll
        for (int i = 0; i < RM; i++)
            a[i] = A[rowOff[i] + k];

        const int kN = k * N;

        #pragma unroll
        for (int j = 0; j < RN; j++)
            b[j] = B[kN + colIdx[j]];

        // ... feed RM * RN multiply-adds.
        #pragma unroll
        for (int i = 0; i < RM; i++)
            #pragma unroll
            for (int j = 0; j < RN; j++)
                acc[i][j] += a[i] * b[j];
    }

    // Write the RM x RN results. For each j the lanes write consecutive columns.
    #pragma unroll
    for (int i = 0; i < RM; i++)
    {
        int r = rowBase + i * BLOCK_Y;

        if (r < N)
        {
            #pragma unroll
            for (int j = 0; j < RN; j++)
            {
                int c = colBase + j * BLOCK_X;

                if (c < N)
                    C[r * N + c] = acc[i][j];
            }
        }
    }
}