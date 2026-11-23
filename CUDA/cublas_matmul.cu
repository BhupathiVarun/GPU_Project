#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                            \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

#define CUBLAS_CHECK(call)                                    \
do {                                                          \
    cublasStatus_t status = call;                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
        std::cerr << "cuBLAS error, status = "                \
                  << status                                   \
                  << " at line " << __LINE__ << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    size_t n = values.size();

    if (n % 2 == 0)
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;

    return values[n / 2];
}

int main(int argc, char* argv[])
{
    int N = 512;

    if (argc > 1)
        N = std::atoi(argv[1]);

    if (N <= 0) {
        std::cerr << "Matrix size must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(double);

    // Host matrices
    std::vector<double> A(elements);
    std::vector<double> B(elements);
    std::vector<double> C(elements, 0.0);
    std::vector<double> C_ref(elements, 0.0);

    // Same deterministic input data as the custom CUDA program
    for (size_t i = 0; i < elements; i++) {
        A[i] = (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;
        B[i] = (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    // CPU reference calculation
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C_ref[i * N + j] = sum;
        }
    }

    // Allocate GPU memory
    double *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Use double-precision matrix multiplication
    const double alpha = 1.0;
    const double beta = 0.0;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure Host-to-Device transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // Warm-up cuBLAS
    // Row-major C = A * B:
    // cuBLAS sees the row-major arrays as column-major transposes.
    // Therefore calculate C^T = B^T * A^T by passing B before A.
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure cuBLAS execution time
    std::vector<double> timings;
    const int runs = 5;

    for (int r = 0; r < runs; r++) {
        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        timings.push_back(elapsed_ms);
    }

    double median_ms = median(timings);

    double average_ms = 0.0;
    for (double t : timings)
        average_ms += t;

    average_ms /= timings.size();

    // Measure Device-to-Host transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // Verify numerical correctness
    double max_error = 0.0;

    for (size_t i = 0; i < elements; i++) {
        double error = std::abs(C[i] - C_ref[i]);

        if (error > max_error)
            max_error = error;
    }

    bool passed = max_error < 1e-9;

    // Performance calculation
    double operations = 2.0 * N * N * N;
    double gflops = operations / (median_ms * 1.0e6);

    // Display results
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n======= cuBLAS MATRIX MULTIPLICATION =======\n";
    std::cout << "Matrix size             : "
              << N << " x " << N << "\n";

    std::cout << "\n--- Timing Results ---\n";
    std::cout << "H2D transfer time       : "
              << h2d_ms << " ms\n";

    std::cout << "D2H transfer time       : "
              << d2h_ms << " ms\n";

    std::cout << "Kernel median time      : "
              << median_ms << " ms\n";

    std::cout << "Kernel average time     : "
              << average_ms << " ms\n";

    std::cout << "\n--- Performance ---\n";
    std::cout << "GPU performance         : "
              << gflops << " GFLOP/s\n";

    std::cout << "\n--- Correctness ---\n";
    std::cout << "C[0]                    : "
              << C[0] << "\n";

    std::cout << "Maximum absolute error  : "
              << max_error << "\n";

    std::cout << "Correctness             : "
              << (passed ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
