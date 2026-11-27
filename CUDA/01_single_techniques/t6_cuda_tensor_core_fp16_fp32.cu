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
