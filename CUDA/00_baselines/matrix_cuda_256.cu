
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

int main()
{
    int N = 256;

    size_t bytes = N * N * sizeof(double);

    // Allocate host memory
    double *h_A = new double[N * N];
    double *h_B = new double[N * N];
    double *h_C = new double[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    // Allocate GPU memory
    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy matrices from CPU to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define GPU thread configuration
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch CUDA kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    // Wait for GPU execution
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    bool correct = true;

    double expected = 2.0 * N;

    for (int i = 0; i < N * N; i++)
    {
        if (abs(h_C[i] - expected) > 1e-9)
        {
            correct = false;
            break;
        }
    }