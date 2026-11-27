
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;


#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

static_assert(TILE_SIZE > 0, "TILE_SIZE must be positive");
static_assert(TILE_SIZE * TILE_SIZE <= 1024,
              "TILE_SIZE creates more than 1024 threads per block");

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA Error: " << cudaGetErrorString(err)     \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

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

void cpuMatrixMultiply(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
    }
}

