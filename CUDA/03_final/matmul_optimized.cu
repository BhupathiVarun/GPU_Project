
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

    template <int BM, int BN, int BK, int RM, int RN>
__global__ void __launch_bounds__(BM * BN)
matmulFinalFP64(const double *__restrict__ A,
                const double *__restrict__ B,
                double *__restrict__ C,
                int N)
{
    static_assert(BM * BN <= 1024, "too many threads in one block");

    constexpr int OUT_M = BM * RM;
    constexpr int OUT_N = BN * RN;
    constexpr int AS_STRIDE = BK + 1;       // padded shared-memory row
    constexpr int BS_STRIDE = OUT_N + 1;    // padded shared-memory row
    constexpr int THREADS = BM * BN;

    extern __shared__ double shared[];
    double *As = shared;
    double *Bs = As + static_cast<size_t>(OUT_M) * AS_STRIDE;

    const int tx = threadIdx.x;             // fastest-changing lane coordinate
    const int ty = threadIdx.y;
    const int tid = ty * BN + tx;

    const int rowBase = blockIdx.y * OUT_M;
    const int colBase = blockIdx.x * OUT_N;

    // RM*RN independent partial sums. These are intended to stay in registers.
    double accum[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            accum[i][j] = 0.0;
        }
    }

    const int kTiles = (N + BK - 1) / BK;

    for (int tile = 0; tile < kTiles; ++tile) {
        const int kBase = tile * BK;

        // Load A tile (OUT_M x BK). Consecutive flattened indices map to
        // consecutive k positions within each row.
        const int aElements = OUT_M * BK;
        for (int idx = tid; idx < aElements; idx += THREADS) {
            const int localRow = idx / BK;
            const int localK = idx - localRow * BK;
            const int globalRow = rowBase + localRow;
            const int globalK = kBase + localK;

            As[static_cast<size_t>(localRow) * AS_STRIDE + localK] =
                (globalRow < N && globalK < N)
                    ? A[static_cast<size_t>(globalRow) * N + globalK]
                    : 0.0;
        }

        // Load B tile (BK x OUT_N). Consecutive flattened indices map to
        // consecutive output columns, producing coalesced row-major accesses.
        const int bElements = BK * OUT_N;
        for (int idx = tid; idx < bElements; idx += THREADS) {
            const int localK = idx / OUT_N;
            const int localCol = idx - localK * OUT_N;
            const int globalK = kBase + localK;
            const int globalCol = colBase + localCol;

            Bs[static_cast<size_t>(localK) * BS_STRIDE + localCol] =
                (globalK < N && globalCol < N)
                    ? B[static_cast<size_t>(globalK) * N + globalCol]
                    : 0.0;
        }

        __syncthreads();

        // Register blocking: RM A values and RN B values feed RM*RN FMAs.
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double aValues[RM];
            double bValues[RN];

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                const int localRow = ty + i * BM;
                aValues[i] = As[static_cast<size_t>(localRow) * AS_STRIDE + kk];
            }

            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int localCol = tx + j * BN;
                bValues[j] = Bs[static_cast<size_t>(kk) * BS_STRIDE + localCol];
            }

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    accum[i][j] = fma(aValues[i], bValues[j], accum[i][j]);
                }
            }
        }

        // No thread may overwrite As/Bs until every thread has finished using
        // the previous K tile.
        __syncthreads();
    }

    // Unique ownership: each valid C element is written by exactly one thread.
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int globalRow = rowBase + ty + i * BM;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int globalCol = colBase + tx + j * BN;
            if (globalRow < N && globalCol < N) {
                C[static_cast<size_t>(globalRow) * N + globalCol] = accum[i][j];
            }
        }
    }
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int TC_TILE_M = 64;
constexpr int TC_TILE_N = 64;
constexpr int TC_TILE_K = 64;
constexpr int TC_PAD = 8;
constexpr int TC_THREADS = 128;

using WmmaFragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::row_major>;
using WmmaFragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::row_major>;
using WmmaFragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                                 float>;

__global__ void __launch_bounds__(TC_THREADS)
matmulFinalWMMA(const __half *__restrict__ A,
                const __half *__restrict__ B,
                float *__restrict__ C,
                int N)
{
    __shared__ __align__(128) __half As[TC_TILE_M][TC_TILE_K + TC_PAD];
    __shared__ __align__(128) __half Bs[TC_TILE_K][TC_TILE_N + TC_PAD];

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int blockRow = blockIdx.y * TC_TILE_M;
    const int blockCol = blockIdx.x * TC_TILE_N;

    WmmaFragC accum[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(accum[i][j], 0.0f);
        }
    }

    for (int kBase = 0; kBase < N; kBase += TC_TILE_K) {
        // Vectorized 16-byte global loads: eight consecutive FP16 values.
        for (int idx = tid; idx < TC_TILE_M * TC_TILE_K / 8;
             idx += TC_THREADS) {
            const int r = idx / (TC_TILE_K / 8);
            const int c = (idx % (TC_TILE_K / 8)) * 8;
            *reinterpret_cast<int4 *>(&As[r][c]) =
                *reinterpret_cast<const int4 *>(
                    &A[static_cast<size_t>(blockRow + r) * N + kBase + c]);
        }

        for (int idx = tid; idx < TC_TILE_K * TC_TILE_N / 8;
             idx += TC_THREADS) {
            const int r = idx / (TC_TILE_N / 8);
            const int c = (idx % (TC_TILE_N / 8)) * 8;
            *reinterpret_cast<int4 *>(&Bs[r][c]) =
                *reinterpret_cast<const int4 *>(
                    &B[static_cast<size_t>(kBase + r) * N + blockCol + c]);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TC_TILE_K; kk += WMMA_K) {
            WmmaFragA a[2];
            WmmaFragB b[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(
                    a[i], &As[warpRow * 32 + i * WMMA_M][kk],
                    TC_TILE_K + TC_PAD);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(
                    b[j], &Bs[kk][warpCol * 32 + j * WMMA_N],
                    TC_TILE_N + TC_PAD);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(accum[i][j], a[i], b[j], accum[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(
                C + static_cast<size_t>(blockRow + warpRow * 32 + i * WMMA_M) * N
                  + blockCol + warpCol * 32 + j * WMMA_N,
                accum[i][j], N, wmma::mem_row_major);
        }
    }
}

struct Timing {
    double medianMs = 0.0;
    double averageMs = 0.0;
    double gflops = 0.0;
};

struct ErrorStats {
    double maxAbs = 0.0;
    double rmse = 0.0;
};

template <typename Launch>
static Timing benchmark(const string &name,
                        Launch launch,
                        double operations,
                        cudaEvent_t start,
                        cudaEvent_t stop)
{
    // Warm-up runs are not included in timing samples.
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> samples;
    samples.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    sort(samples.begin(), samples.end());
    if (samples.size() % 2 == 0) {
        const size_t hi = samples.size() / 2;
        const size_t lo = hi - 1;
        Timing t;
        t.medianMs = 0.5 * (samples[lo] + samples[hi]);
        for (float x : samples) t.averageMs += x;
        t.averageMs /= samples.size();
        t.gflops = (t.medianMs > 0.0)
                 ? operations / (t.medianMs * 1.0e6) : 0.0;
        return t;
    }

    Timing t;
    t.medianMs = samples[samples.size() / 2];
    for (float x : samples) t.averageMs += x;
    t.averageMs /= samples.size();
    t.gflops = (t.medianMs > 0.0)
             ? operations / (t.medianMs * 1.0e6) : 0.0;
    (void)name;
    return t;
}

static void cpuMatrixMultiply(const vector<double> &A,
                              const vector<double> &B,
                              vector<double> &C,
                              int N)
{
    fill(C.begin(), C.end(), 0.0);
    // i-k-j order improves CPU cache locality. For each output element, k is
    // still accumulated in ascending order, matching the CUDA reduction order.
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            const double a = A[static_cast<size_t>(i) * N + k];
            const size_t bRow = static_cast<size_t>(k) * N;
            const size_t cRow = static_cast<size_t>(i) * N;
            for (int j = 0; j < N; ++j) {
                C[cRow + j] += a * B[bRow + j];
            }
        }
    }
}

static ErrorStats compareDouble(const vector<double> &actual,
                                const vector<double> &reference)
{
    ErrorStats e;
    double sumSq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = fabs(actual[i] - reference[i]);
        e.maxAbs = max(e.maxAbs, d);
        sumSq += d * d;
    }
    e.rmse = sqrt(sumSq / max<size_t>(1, actual.size()));
    return e;
}

static ErrorStats compareFloatToDouble(const vector<float> &actual,
                                       const vector<double> &reference,
                                       int N,
                                       int paddedN)
{
    ErrorStats e;
    double sumSq = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double d = fabs(static_cast<double>(
                actual[static_cast<size_t>(i) * paddedN + j])
                - reference[static_cast<size_t>(i) * N + j]);
            e.maxAbs = max(e.maxAbs, d);
            sumSq += d * d;
        }
    }
    e.rmse = sqrt(sumSq / max<size_t>(1, static_cast<size_t>(N) * N));
    return e;
}

template <typename CopyInputs>
static double measureH2D(CopyInputs copyInputs,
                         cudaEvent_t start,
                         cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));
    copyInputs();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main(int argc, char **argv)
{
    int N = 256;
    if (argc >= 2) N = atoi(argv[1]);
    if (N <= 0) {
        cerr << "Matrix size must be positive.\n";
        return EXIT_FAILURE;
    }

    const size_t maxSize = numeric_limits<size_t>::max();
    if (static_cast<size_t>(N) > maxSize / static_cast<size_t>(N)) {
        cerr << "Matrix element count overflows size_t.\n";
        return EXIT_FAILURE;
    }
    const size_t elements = static_cast<size_t>(N) * N;
    if (elements > maxSize / sizeof(double)) {
        cerr << "Matrix allocation size overflows size_t.\n";
        return EXIT_FAILURE;
    }
    const size_t doubleBytes = elements * sizeof(double);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << fixed << setprecision(6);
    cout << "============================================================\n";
    cout << "FINAL GPU MATRIX MULTIPLICATION\n";
    cout << "GPU                         : " << prop.name << " ("
         << prop.multiProcessorCount << " SMs, CC "
         << prop.major << "." << prop.minor << ")\n";
    cout << "Matrix size                 : " << N << " x " << N << "\n";
    cout << "FP64 T4 tile BM, BN, BK     : "
         << FINAL_BM << ", " << FINAL_BN << ", " << FINAL_BK << "\n";
    cout << "FP64 C2 register tile RM,RN : "
         << FINAL_RM << ", " << FINAL_RN << "\n";
    cout << "FP64 output tile per block  : "
         << FINAL_BM * FINAL_RM << " x " << FINAL_BN * FINAL_RN << "\n";
    cout << "FP64 threads per block      : " << FINAL_BM * FINAL_BN << "\n";
    cout << "Warm-ups / timed runs       : " << WARMUP_RUNS
         << " / " << TIMED_RUNS << "\n";

    vector<double> h_A(elements), h_B(elements), h_C(elements), h_reference(elements);
    for (size_t i = 0; i < elements; ++i) {
        h_A[i] = (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;
        h_B[i] = (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    cout << "\nComputing FP64 CPU reference...\n";
    const auto cpuStart = chrono::high_resolution_clock::now();
    cpuMatrixMultiply(h_A, h_B, h_reference, N);
    const auto cpuStop = chrono::high_resolution_clock::now();
    const double cpuMs = chrono::duration<double, milli>(cpuStop - cpuStart).count();
    cout << "CPU reference time          : " << cpuMs << " ms\n";

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, doubleBytes));
    CUDA_CHECK(cudaMalloc(&d_B, doubleBytes));
    CUDA_CHECK(cudaMalloc(&d_C, doubleBytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const double h2dMs = measureH2D([&]() {
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), doubleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), doubleBytes, cudaMemcpyHostToDevice));
    }, start, stop);
    cout << "FP64 H2D transfer time      : " << h2dMs << " ms\n";
    // final
    const size_t outM = static_cast<size_t>(FINAL_BM) * FINAL_RM;
    const size_t outN = static_cast<size_t>(FINAL_BN) * FINAL_RN;
    const size_t asElements = outM * (static_cast<size_t>(FINAL_BK) + 1);
    const size_t bsElements = static_cast<size_t>(FINAL_BK) * (outN + 1);
    const size_t finalSharedBytes = (asElements + bsElements) * sizeof(double);

    const size_t defaultShared = static_cast<size_t>(prop.sharedMemPerBlock);
    const size_t optinShared = (prop.sharedMemPerBlockOptin > 0)
        ? static_cast<size_t>(prop.sharedMemPerBlockOptin) : defaultShared;

    bool fp64FinalRan = false;
    Timing finalFp64Timing;
    ErrorStats finalFp64Error;

    cout << "\n======= FINAL CUSTOM FP64 KERNEL =======\n";
    cout << "Shared memory required      : " << finalSharedBytes << " bytes\n";
    cout << "Default shared memory limit : " << defaultShared << " bytes\n";
    cout << "Opt-in shared memory limit  : " << optinShared << " bytes\n";

    if (finalSharedBytes > optinShared) {
        cout << "FINAL FP64 kernel skipped: selected T4/C2 configuration exceeds this GPU's shared-memory limit.\n";
    } else {
        if (finalSharedBytes > defaultShared) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void *)matmulFinalFP64<FINAL_BM, FINAL_BN, FINAL_BK,
                                               FINAL_RM, FINAL_RN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(finalSharedBytes)));
        }

        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(
            &attr, (const void *)matmulFinalFP64<FINAL_BM, FINAL_BN, FINAL_BK,
                                                  FINAL_RM, FINAL_RN>));
        int activeBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks,
            (const void *)matmulFinalFP64<FINAL_BM, FINAL_BN, FINAL_BK,
                                          FINAL_RM, FINAL_RN>,
            FINAL_BM * FINAL_BN, finalSharedBytes));

        cout << "Registers per thread        : " << attr.numRegs << "\n";
        cout << "Local memory per thread     : " << attr.localSizeBytes
             << " bytes (nonzero can indicate spills)\n";
        cout << "Active blocks per SM        : " << activeBlocks << "\n";

        const dim3 block(FINAL_BN, FINAL_BM);
        const dim3 grid(static_cast<unsigned>((N + outN - 1) / outN),
                        static_cast<unsigned>((N + outM - 1) / outM));
        auto launchFinal = [&]() {
            matmulFinalFP64<FINAL_BM, FINAL_BN, FINAL_BK, FINAL_RM, FINAL_RN>
                <<<grid, block, finalSharedBytes>>>(d_A, d_B, d_C, N);
        };

        const double fp64Ops = 2.0 * static_cast<double>(N) * N * N;
        finalFp64Timing = benchmark("final_fp64_custom", launchFinal,
                                    fp64Ops, start, stop);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, doubleBytes, cudaMemcpyDeviceToHost));
        finalFp64Error = compareDouble(h_C, h_reference);
        fp64FinalRan = true;

        cout << "Block shape                 : " << FINAL_BN << " x " << FINAL_BM
             << " threads\n";
        cout << "Grid                        : " << grid.x << " x " << grid.y << " blocks\n";
        cout << "Kernel median time          : " << finalFp64Timing.medianMs << " ms\n";
        cout << "Kernel average time         : " << finalFp64Timing.averageMs << " ms\n";
        cout << "Performance                 : " << finalFp64Timing.gflops << " GFLOP/s\n";
        cout << "Maximum absolute error      : " << finalFp64Error.maxAbs << "\n";
        cout << "RMSE                        : " << finalFp64Error.rmse << "\n";
        cout << "Correctness                 : "
             << (finalFp64Error.maxAbs <= FP64_TOLERANCE ? "PASSED" : "FAILED") << "\n";
        cout << "RESULT,final_fp64_custom," << N << ","
             << finalFp64Timing.medianMs << "," << finalFp64Timing.averageMs << ","
             << finalFp64Timing.gflops << "," << finalFp64Error.maxAbs << "\n";
    }
    // cublas 
    cout << "\n======= cuBLAS DGEMM (FP64 REFERENCE) =======\n";
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    const double alphaD = 1.0;
    const double betaD = 0.0;
    auto launchDgemm = [&]() {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alphaD,
                                 d_B, N, d_A, N, &betaD, d_C, N));
    };
    const double fp64Ops = 2.0 * static_cast<double>(N) * N * N;
    Timing dgemmTiming = benchmark("cublas_dgemm_fp64", launchDgemm,
                                  fp64Ops, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, doubleBytes, cudaMemcpyDeviceToHost));
    ErrorStats dgemmError = compareDouble(h_C, h_reference);

    cout << "Kernel median time          : " << dgemmTiming.medianMs << " ms\n";
    cout << "Kernel average time         : " << dgemmTiming.averageMs << " ms\n";
    cout << "Performance                 : " << dgemmTiming.gflops << " GFLOP/s\n";
    cout << "Maximum absolute error      : " << dgemmError.maxAbs << "\n";
    cout << "Correctness                 : "
         << (dgemmError.maxAbs <= FP64_TOLERANCE ? "PASSED" : "FAILED") << "\n";
    cout << "RESULT,cublas_dgemm_fp64," << N << ","
         << dgemmTiming.medianMs << "," << dgemmTiming.averageMs << ","
         << dgemmTiming.gflops << "," << dgemmError.maxAbs << "\n";

    // tensor core
    cout << "\n======= TENSOR CORE PATH (FP16 INPUT / FP32 ACCUMULATION) =======\n";
    bool tensorRan = false;
    if (prop.major < 7) {
        cout << "Tensor Core WMMA skipped: this GPU has compute capability below 7.0.\n";
    } else {
        const int P = ((N + TC_TILE_M - 1) / TC_TILE_M) * TC_TILE_M;
        const size_t paddedElements = static_cast<size_t>(P) * P;
        const size_t halfBytes = paddedElements * sizeof(__half);
        const size_t floatBytes = paddedElements * sizeof(float);

        vector<__half> h_Ah(paddedElements, __float2half(0.0f));
        vector<__half> h_Bh(paddedElements, __float2half(0.0f));
        vector<float> h_Cf(paddedElements, 0.0f);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const size_t src = static_cast<size_t>(i) * N + j;
                const size_t dst = static_cast<size_t>(i) * P + j;
                h_Ah[dst] = __float2half(static_cast<float>(h_A[src]));
                h_Bh[dst] = __float2half(static_cast<float>(h_B[src]));
            }
        }

        __half *d_Ah = nullptr;
        __half *d_Bh = nullptr;
        float *d_Cf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Ah, halfBytes));
        CUDA_CHECK(cudaMalloc(&d_Bh, halfBytes));
        CUDA_CHECK(cudaMalloc(&d_Cf, floatBytes));

        const double tensorH2D = measureH2D([&]() {
            CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah.data(), halfBytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bh, h_Bh.data(), halfBytes, cudaMemcpyHostToDevice));
        }, start, stop);

        const dim3 wmmaBlock(TC_THREADS);
        const dim3 wmmaGrid(P / TC_TILE_N, P / TC_TILE_M);
        auto launchWmma = [&]() {
            matmulFinalWMMA<<<wmmaGrid, wmmaBlock>>>(d_Ah, d_Bh, d_Cf, P);
        };

        const double tensorOps = 2.0 * static_cast<double>(P) * P * P;
        Timing wmmaTiming = benchmark("final_wmma_tensor_core", launchWmma,
                                     tensorOps, start, stop);
        CUDA_CHECK(cudaMemcpy(h_Cf.data(), d_Cf, floatBytes, cudaMemcpyDeviceToHost));
        ErrorStats wmmaError = compareFloatToDouble(h_Cf, h_reference, N, P);

        cout << "Original matrix size        : " << N << " x " << N << "\n";
        cout << "Padded WMMA matrix size     : " << P << " x " << P << "\n";
        cout << "Block tile                  : 64 x 64\n";
        cout << "Warps per block             : 4\n";
        cout << "H2D transfer time           : " << tensorH2D << " ms\n";
        cout << "Kernel median time          : " << wmmaTiming.medianMs << " ms\n";
        cout << "Kernel average time         : " << wmmaTiming.averageMs << " ms\n";
        cout << "Effective throughput        : " << wmmaTiming.gflops << " GFLOP/s (padded work)\n";
        cout << "Maximum absolute error      : " << wmmaError.maxAbs << "\n";
        cout << "RMSE                        : " << wmmaError.rmse << "\n";
        cout << "FP16-input tolerance        : " << TENSOR_TOLERANCE << "\n";
        cout << "Accuracy check              : "
             << (wmmaError.maxAbs <= TENSOR_TOLERANCE ? "PASSED" : "REVIEW ERROR") << "\n";
        cout << "RESULT,final_wmma_fp16_fp32," << N << ","
             << wmmaTiming.medianMs << "," << wmmaTiming.averageMs << ","
             << wmmaTiming.gflops << "," << wmmaError.maxAbs << "\n";

        // Vendor Tensor Core reference with the same FP16 inputs/FP32 output.
        const float alphaF = 1.0f;
        const float betaF = 0.0f;
        auto launchGemmEx = [&]() {
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                P, P, P,
                &alphaF,
                d_Bh, CUDA_R_16F, P,
                d_Ah, CUDA_R_16F, P,
                &betaF,
                d_Cf, CUDA_R_32F, P,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
        };

        Timing gemmExTiming = benchmark("cublas_gemmex_fp16_fp32", launchGemmEx,
                                       tensorOps, start, stop);
        CUDA_CHECK(cudaMemcpy(h_Cf.data(), d_Cf, floatBytes, cudaMemcpyDeviceToHost));
        ErrorStats gemmExError = compareFloatToDouble(h_Cf, h_reference, N, P);

        cout << "\n--- cuBLAS Tensor Core Reference ---\n";
        cout << "Kernel median time          : " << gemmExTiming.medianMs << " ms\n";
        cout << "Kernel average time         : " << gemmExTiming.averageMs << " ms\n";
        cout << "Effective throughput        : " << gemmExTiming.gflops << " GFLOP/s (padded work)\n";
        cout << "Maximum absolute error      : " << gemmExError.maxAbs << "\n";
        cout << "RMSE                        : " << gemmExError.rmse << "\n";
        cout << "RESULT,cublas_gemmex_fp16_fp32," << N << ","
             << gemmExTiming.medianMs << "," << gemmExTiming.averageMs << ","
             << gemmExTiming.gflops << "," << gemmExError.maxAbs << "\n";

        tensorRan = true;
        CUDA_CHECK(cudaFree(d_Ah));
        CUDA_CHECK(cudaFree(d_Bh));
        CUDA_CHECK(cudaFree(d_Cf));
    }

    cout << "\n======= FINAL SUMMARY =======\n";
    cout << "N                           : " << N << "\n";
    if (fp64FinalRan) {
        cout << "Custom FP64                 : " << finalFp64Timing.medianMs
             << " ms, " << finalFp64Timing.gflops << " GFLOP/s, max error "
             << finalFp64Error.maxAbs << "\n";
    } else {
        cout << "Custom FP64                 : SKIPPED (shared-memory limit)\n";
    }
    cout << "cuBLAS DGEMM FP64           : " << dgemmTiming.medianMs
         << " ms, " << dgemmTiming.gflops << " GFLOP/s, max error "
         << dgemmError.maxAbs << "\n";
    cout << "Tensor Core path            : "
         << (tensorRan ? "executed (FP16 input / FP32 output)" : "skipped") << "\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    const bool fp64Correct = dgemmError.maxAbs <= FP64_TOLERANCE &&
        (!fp64FinalRan || finalFp64Error.maxAbs <= FP64_TOLERANCE);
    return fp64Correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
