
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;
using namespace nvcuda;

#ifndef FINAL_BM
#define FINAL_BM 16
#endif
#ifndef FINAL_BN
#define FINAL_BN 16
#endif
#ifndef FINAL_BK
#define FINAL_BK 16
#endif
#ifndef FINAL_RM
#define FINAL_RM 2
#endif
#ifndef FINAL_RN
#define FINAL_RN 2
#endif

#define FINAL_VALID_T4_CONFIG ( \
    (FINAL_BM == 8  && FINAL_BN == 8  && FINAL_BK == 8)  || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 8)  || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 16) || \
    (FINAL_BM == 16 && FINAL_BN == 16 && FINAL_BK == 32) || \
    (FINAL_BM == 8  && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 16 && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 32 && FINAL_BN == 32 && FINAL_BK == 16) || \
    (FINAL_BM == 32 && FINAL_BN == 32 && FINAL_BK == 32) )

static_assert(FINAL_VALID_T4_CONFIG,
              "FINAL_BM/BN/BK must match one of the eight full T4 configurations");
static_assert(FINAL_BM * FINAL_BN <= 1024,
              "FINAL_BM * FINAL_BN exceeds CUDA's 1024-thread block limit");
static_assert((FINAL_RM == 1 || FINAL_RM == 2 || FINAL_RM == 4 || FINAL_RM == 8),
              "FINAL_RM must be one of C2's tested values: 1, 2, 4, 8");
static_assert((FINAL_RN == 1 || FINAL_RN == 2 || FINAL_RN == 4 || FINAL_RN == 8),
              "FINAL_RN must be one of C2's tested values: 1, 2, 4, 8");

#define TIMED_RUNS 20
#define WARMUP_RUNS 2
#define FP64_TOLERANCE 1.0e-9
#define TENSOR_TOLERANCE 1.0e-2

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            cerr << "CUDA error: " << cudaGetErrorString(err__)              \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status__ = (call);                                    \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                             \
            cerr << "cuBLAS error (status " << static_cast<int>(status__)    \
                 << ") at " << __FILE__ << ":" << __LINE__ << endl;         \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
