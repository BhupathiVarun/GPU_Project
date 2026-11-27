#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA Error: " << cudaGetErrorString(err)     \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// Shared-memory tiled matrix multiplication
__global__ void matrixMultiplyTiled(
    const double *A,
    const double *B,
    double *C,
    int N)
{
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    int tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < tiles; t++)
    {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Cooperatively load A and B tiles
        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] =
                A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (bRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute using shared-memory values
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += As[threadIdx.y][k] *
                   Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}