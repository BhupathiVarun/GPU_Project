
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

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

// Basic CUDA matrix multiplication kernel
__global__ void matrixMultiply(
    const double *A,
    const double *B,
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

// CPU reference implementation
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
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int N = 256;

    if (argc > 1)
        N = atoi(argv[1]);

    if (N <= 0)
    {
        cerr << "Matrix size must be positive." << endl;
        return 1;
    }

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(double);

    vector<double> h_A(elements);
    vector<double> h_B(elements);
    vector<double> h_C(elements);
    vector<double> h_reference(elements);

    // Initialize deterministic input matrices.
    // These formulas match the CPU project programs.
    for (size_t i = 0; i < elements; i++)
    {
        h_A[i] =
            (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;

        h_B[i] =
            (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure host-to-device transfer time
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // Configure GPU threads and blocks
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Warm-up execution
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel execution time
    vector<float> kernelTimes;

    for (int run = 0; run < 5; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C, N
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds, start, stop
        ));

        kernelTimes.push_back(milliseconds);
    }

    // Median and average kernel time
    sort(kernelTimes.begin(), kernelTimes.end());

    double median_ms = kernelTimes[kernelTimes.size() / 2];

    double average_ms = 0.0;

    for (float t : kernelTimes)
        average_ms += t;

    average_ms /= kernelTimes.size();

    // Measure device-to-host transfer time
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // CPU reference for correctness checking
    cpuMatrixMultiply(h_A, h_B, h_reference, N);

    bool correct = true;
    double maxError = 0.0;

    for (size_t i = 0; i < elements; i++)
    {
        double error = fabs(h_C[i] - h_reference[i]);

        maxError = max(maxError, error);

        if (error > 1e-9)
            correct = false;
    }

    // Performance calculation
    double operations = 2.0 * N * N * N;

    double gflops = operations / (median_ms * 1e6);

    cout << "\n========== BASIC CUDA MATRIX MULTIPLICATION ==========\n";

    cout << "GPU matrix size          : " << N << " x " << N << endl;
    cout << "Threads per block        : "
         << threadsPerBlock.x * threadsPerBlock.y << endl;

    cout << "Blocks in grid           : "
         << blocksPerGrid.x << " x "
         << blocksPerGrid.y << endl;

    cout << "\n--- Timing Results ---\n";

    cout << "H2D transfer time        : " << h2d_ms << " ms" << endl;
    cout << "D2H transfer time        : " << d2h_ms << " ms" << endl;
    cout << "Kernel median time       : " << median_ms << " ms" << endl;
    cout << "Kernel average time      : " << average_ms << " ms" << endl;

    cout << "\n--- Performance ---\n";

    cout << "GPU performance          : " << gflops << " GFLOP/s" << endl;

    cout << "\n--- Correctness ---\n";

    cout << "C[0]                     : " << h_C[0] << endl;
    cout << "Maximum absolute error   : " << maxError << endl;

    cout << "Correctness              : "
         << (correct ? "PASSED" : "FAILED") << endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return correct ? 0 : 1;
}
