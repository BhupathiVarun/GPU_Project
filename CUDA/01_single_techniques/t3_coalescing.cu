#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        cerr << "CUDA Error: " << cudaGetErrorString(err)          \
             << " at line " << __LINE__ << endl;                   \
        exit(EXIT_FAILURE);                                        \
    }                                                               \
} while (0)


__global__ void matmulCoalesced(
    const double *A, const double *B, double *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


__global__ void matmulStrided(
    const double *A, const double *B, double *C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void cpuMatrixMultiply(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

template <typename KernelLauncher>
double benchmarkKernel(
    KernelLauncher launch,
    cudaEvent_t start,
    cudaEvent_t stop,
    int runs = 5)
{
    vector<float> times;
    times.reserve(runs);

    
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    sort(times.begin(), times.end());
    if (times.size() % 2 == 0)
        return (times[times.size()/2 - 1] + times[times.size()/2]) / 2.0;
    return times[times.size()/2];
}

double maxAbsError(const vector<double>& actual,
                   const vector<double>& reference)
{
    double result = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
        result = max(result, fabs(actual[i] - reference[i]));
    return result;
}

