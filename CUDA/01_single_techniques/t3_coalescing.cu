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

int main(int argc, char *argv[])
{
    int N = 512;
    if (argc > 1)
        N = atoi(argv[1]);

    if (N <= 0) {
        cerr << "Matrix size must be positive." << endl;
        return EXIT_FAILURE;
    }

    const size_t elements = static_cast<size_t>(N) * N;
    const size_t bytes = elements * sizeof(double);

    vector<double> h_A(elements), h_B(elements);
    vector<double> h_C_coalesced(elements), h_C_strided(elements);
    vector<double> h_reference(elements);

    
    for (size_t i = 0; i < elements; ++i) {
        h_A[i] =
            (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;
        h_B[i] =
            (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    double *d_A = nullptr, *d_B = nullptr;
    double *d_C_coalesced = nullptr, *d_C_strided = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_coalesced, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_strided, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    
    const dim3 threadsPerBlock(32, 8);
    const dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const double coalesced_ms = benchmarkKernel(
        [&]() {
            matmulCoalesced<<<blocksPerGrid, threadsPerBlock>>>(
                d_A, d_B, d_C_coalesced, N);
        }, start, stop);

    const double strided_ms = benchmarkKernel(
        [&]() {
            matmulStrided<<<blocksPerGrid, threadsPerBlock>>>(
                d_A, d_B, d_C_strided, N);
        }, start, stop);

    CUDA_CHECK(cudaMemcpy(
        h_C_coalesced.data(), d_C_coalesced, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_C_strided.data(), d_C_strided, bytes, cudaMemcpyDeviceToHost));

    cpuMatrixMultiply(h_A, h_B, h_reference, N);

    const double coalesced_error = maxAbsError(h_C_coalesced, h_reference);
    const double strided_error = maxAbsError(h_C_strided, h_reference);
    const bool coalesced_ok = coalesced_error <= 1e-9;
    const bool strided_ok = strided_error <= 1e-9;

    const double operations = 2.0 * N * N * N;
    const double coalesced_gflops = operations / (coalesced_ms * 1.0e6);
    const double strided_gflops = operations / (strided_ms * 1.0e6);
    const double speedup = strided_ms / coalesced_ms;

    cout << fixed << setprecision(6);
    cout << "\n======= T3 MEMORY COALESCING EXPERIMENT =======\n";
    cout << "Matrix size                 : " << N << " x " << N << "\n";
    cout << "Threads per block           : 32 x 8 = 256\n";
    cout << "Measured runs per kernel    : 5 (median)\n\n";

    cout << "COALESCED MAPPING\n";
    cout << "Kernel median time          : " << coalesced_ms << " ms\n";
    cout << "Performance                 : " << coalesced_gflops << " GFLOP/s\n";
    cout << "Maximum absolute error      : " << coalesced_error << "\n";
    cout << "Correctness                 : "
         << (coalesced_ok ? "PASSED" : "FAILED") << "\n\n";

    cout << "DELIBERATELY STRIDED MAPPING\n";
    cout << "Kernel median time          : " << strided_ms << " ms\n";
    cout << "Performance                 : " << strided_gflops << " GFLOP/s\n";
    cout << "Maximum absolute error      : " << strided_error << "\n";
    cout << "Correctness                 : "
         << (strided_ok ? "PASSED" : "FAILED") << "\n\n";

    cout << "Strided time / coalesced time: " << speedup << "x\n";
    cout << "(A value above 1 means the coalesced mapping was faster.)\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_coalesced));
    CUDA_CHECK(cudaFree(d_C_strided));

    return (coalesced_ok && strided_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
