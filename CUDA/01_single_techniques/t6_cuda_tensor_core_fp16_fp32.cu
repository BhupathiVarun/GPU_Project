#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

    if (N <= 0 || N % 8 != 0) {
        std::cerr << "Matrix size must be positive and divisible by 8."
                  << std::endl;
        return EXIT_FAILURE;
    }

    size_t elements = static_cast<size_t>(N) * N;

    size_t half_bytes = elements * sizeof(__half);
    size_t float_bytes = elements * sizeof(float);

    // Host matrices
    std::vector<double> A_double(elements);
    std::vector<double> B_double(elements);
    std::vector<double> C_reference(elements, 0.0);

    std::vector<__half> A_half(elements);
    std::vector<__half> B_half(elements);

    std::vector<float> C_float(elements, 0.0f);

    // Generate the same deterministic input data
    // and convert inputs to FP16.
    for (size_t i = 0; i < elements; i++) {

        A_double[i] =
            (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;

        B_double[i] =
            (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;

        A_half[i] = __float2half(static_cast<float>(A_double[i]));
        B_half[i] = __float2half(static_cast<float>(B_double[i]));
    }

    // FP64 CPU reference
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            double sum = 0.0;

            for (int k = 0; k < N; k++) {
                sum += A_double[i * N + k] *
                       B_double[k * N + j];
            }

            C_reference[i * N + j] = sum;
        }
    }

    // Allocate GPU memory
    __half *d_A, *d_B;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, float_bytes));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Request Tensor Core-capable algorithm
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Host-to-Device transfer timing
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        d_A, A_half.data(), half_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, B_half.data(), half_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // Warm-up
    // Row-major C = A * B is calculated as
    // column-major C^T = B^T * A^T.
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, N,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int runs = 5;

    std::vector<double> timings;

    for (int r = 0; r < runs; r++) {

        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, N,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, start, stop));

        timings.push_back(elapsed_ms);
    }

    double median_ms = median(timings);

    double average_ms = 0.0;

    for (double t : timings)
        average_ms += t;

    average_ms /= timings.size();

    // Device-to-Host transfer timing
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        C_float.data(), d_C, float_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // Accuracy analysis against FP64 CPU reference
    double max_abs_error = 0.0;
    double squared_error_sum = 0.0;

    for (size_t i = 0; i < elements; i++) {

        double error =
            std::abs(static_cast<double>(C_float[i]) -
                     C_reference[i]);

        if (error > max_abs_error)
            max_abs_error = error;

        squared_error_sum += error * error;
    }

    double rmse =
        std::sqrt(squared_error_sum / elements);

    // FP16 input with FP32 accumulation has lower
    // input precision than the FP64 reference.
    const double tolerance = 1e-2;

    bool passed = max_abs_error < tolerance;

    // Performance calculation
    double operations = 2.0 * N * N * N;

    double gflops =
        operations / (median_ms * 1.0e6);

}
