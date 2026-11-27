
#include <iostream>
#include <cuda_runtime.h>

using namespace std;


// CUDA kernel: each thread calculates one element of C
__global__ void matrixMultiply(
    double *A,
    double *B,
    double *C,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        double sum = 0.0;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
