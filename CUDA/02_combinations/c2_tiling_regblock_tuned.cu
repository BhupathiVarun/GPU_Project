
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <cuda_runtime.h>

using namespace std;

#ifndef C2_TILE_SIZE
#define C2_TILE_SIZE 16
#endif

static_assert(C2_TILE_SIZE == 8 || C2_TILE_SIZE == 16 || C2_TILE_SIZE == 32,
              "C2_TILE_SIZE must match a T4-tested value: 8, 16, or 32");
static_assert(C2_TILE_SIZE * C2_TILE_SIZE <= 1024,
              "C2 block contains more than 1024 threads");

#define TIMED_RUNS 5
#define ERROR_TOLERANCE 1.0e-9

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            cerr << "CUDA Error: " << cudaGetErrorString(err__)            \
                 << " at line " << __LINE__ << endl;                       \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

template <int RM, int RN>
__global__ void __launch_bounds__(C2_TILE_SIZE * C2_TILE_SIZE)
matmulTiledRegBlocked(
    const double *__restrict__ A,
    const double *__restrict__ B,
    double *__restrict__ C,
    int N)
{
    constexpr int S = C2_TILE_SIZE;
    constexpr int BK = C2_TILE_SIZE;
    constexpr int OUT_M = S * RM;
    constexpr int OUT_N = S * RN;
    constexpr int AS_STRIDE = BK + 1;
    constexpr int BS_STRIDE = OUT_N + 1;
    constexpr int THREAD_COUNT = S * S;

    // Dynamic shared memory lets the host skip oversized variants at runtime
    // rather than causing compilation to fail for a large 8x8 specialization.
    extern __shared__ double shared[];
    double *As = shared;
    double *Bs = As + static_cast<size_t>(OUT_M) * AS_STRIDE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * S + tx;

    const int rowBlockBase = blockIdx.y * OUT_M;
    const int colBlockBase = blockIdx.x * OUT_N;

    // RM*RN accumulators are intended to stay in registers.
    double acc[RM][RN];

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    const int kTiles = (N + BK - 1) / BK;

    for (int tile = 0; tile < kTiles; ++tile) {
        const int kBase = tile * BK;

        // 1. Load A tile: OUT_M x BK. Threads cooperatively traverse the
        //    row-major tile; out-of-range matrix elements are zero-filled.
        const int aElements = OUT_M * BK;
        for (int idx = tid; idx < aElements; idx += THREAD_COUNT) {
            const int localRow = idx / BK;
            const int localK = idx % BK;
            const int globalRow = rowBlockBase + localRow;
            const int globalK = kBase + localK;

            As[static_cast<size_t>(localRow) * AS_STRIDE + localK] =
                (globalRow < N && globalK < N)
                    ? A[static_cast<size_t>(globalRow) * N + globalK]
                    : 0.0;
        }

        // 2. Load B tile: BK x OUT_N.
        const int bElements = BK * OUT_N;
        for (int idx = tid; idx < bElements; idx += THREAD_COUNT) {
            const int localK = idx / OUT_N;
            const int localCol = idx % OUT_N;
            const int globalK = kBase + localK;
            const int globalCol = colBlockBase + localCol;

            Bs[static_cast<size_t>(localK) * BS_STRIDE + localCol] =
                (globalK < N && globalCol < N)
                    ? B[static_cast<size_t>(globalK) * N + globalCol]
                    : 0.0;
        }

        // All threads must finish loading before any thread reads shared tiles.
        __syncthreads();

        // 3. Register-blocked dot products. For each k, load RM A values and
        //    RN B values from shared memory, then reuse them for RM*RN FMAs.
        #pragma unroll 1
        for (int kk = 0; kk < BK; ++kk) {
            double aValues[RM];
            double bValues[RN];

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                const int localRow = ty + i * S;
                aValues[i] = As[static_cast<size_t>(localRow) * AS_STRIDE + kk];
            }

            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int localCol = tx + j * S;
                bValues[j] = Bs[static_cast<size_t>(kk) * BS_STRIDE + localCol];
            }

            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    acc[i][j] += aValues[i] * bValues[j];
                }
            }
        }

        // Do not overwrite shared tiles until every thread has consumed them.
        __syncthreads();
    }

    // 4. Store each thread's RM x RN outputs. Every valid C element is written
    //    exactly once; partial edge tiles are masked.
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int globalRow = rowBlockBase + ty + i * S;

        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int globalCol = colBlockBase + tx + j * S;

            if (globalRow < N && globalCol < N) {
                C[static_cast<size_t>(globalRow) * N + globalCol] = acc[i][j];
            }
        }
    }
}

typedef void (*LaunchFn)(const double *, const double *, double *, int,
                         dim3, dim3, size_t);
typedef cudaError_t (*AttributesFn)(cudaFuncAttributes *);
typedef cudaError_t (*OccupancyFn)(int *, int, size_t);
typedef cudaError_t (*SetDynamicSmemFn)(int);

template <int RM, int RN>
static void launchVariant(
    const double *A, const double *B, double *C, int N,
    dim3 grid, dim3 block, size_t sharedBytes)
{
    matmulTiledRegBlocked<RM, RN><<<grid, block, sharedBytes>>>(A, B, C, N);
}

template <int RM, int RN>
static cudaError_t getVariantAttributes(cudaFuncAttributes *attr)
{
    return cudaFuncGetAttributes(attr, matmulTiledRegBlocked<RM, RN>);
}

template <int RM, int RN>
static cudaError_t getVariantOccupancy(int *blocksPerSM, int threads,
                                       size_t sharedBytes)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocksPerSM, matmulTiledRegBlocked<RM, RN>, threads, sharedBytes);
}

template <int RM, int RN>
static cudaError_t setVariantDynamicSmem(int bytes)
{
    return cudaFuncSetAttribute(
        matmulTiledRegBlocked<RM, RN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
}

static void cpuMatrixMultiply(
    const vector<double> &A,
    const vector<double> &B,
    vector<double> &C,
    int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[static_cast<size_t>(i) * N + k]
                     * B[static_cast<size_t>(k) * N + j];
            }
            C[static_cast<size_t>(i) * N + j] = sum;
        }
    }
}

struct Variant {
    const char *tag;
    const char *name;
    int rm;
    int rn;
    LaunchFn launch;
    AttributesFn getAttributes;
    OccupancyFn getOccupancy;
    SetDynamicSmemFn setDynamicSmem;
    dim3 block;
    dim3 grid;
    size_t sharedBytes;
};

struct Result {
    bool supported;
    bool correct;
    double medianMs;
    double averageMs;
    double d2hMs;
    double gflops;
    double maxError;
    double c0;
    int registersPerThread;
    size_t localBytesPerThread;
    size_t sharedBytes;
    int activeBlocksPerSM;
};

static size_t sharedBytesFor(int rm, int rn)
{
    const size_t S = C2_TILE_SIZE;
    const size_t bk = C2_TILE_SIZE;
    const size_t outM = S * static_cast<size_t>(rm);
    const size_t outN = S * static_cast<size_t>(rn);
    const size_t aBytes = outM * (bk + 1) * sizeof(double);
    const size_t bBytes = bk * (outN + 1) * sizeof(double);
    return aBytes + bBytes;
}

static dim3 gridFor(int N, int rm, int rn)
{
    const int outM = C2_TILE_SIZE * rm;
    const int outN = C2_TILE_SIZE * rn;
    return dim3((N + outN - 1) / outN,
                (N + outM - 1) / outM);
}

static Result runVariant(
    const Variant &v,
    const double *d_A,
    const double *d_B,
    double *d_C,
    vector<double> &h_C,
    const vector<double> &h_reference,
    int N,
    size_t bytes,
    size_t maxSharedPerBlock,
    size_t defaultSharedPerBlock,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    Result r = {};
    r.supported = false;
    r.correct = false;
    r.sharedBytes = v.sharedBytes;

    if (v.sharedBytes > maxSharedPerBlock) {
        cout << "\nSKIPPED " << v.name << ": requires " << v.sharedBytes
             << " bytes shared memory; this GPU supports at most "
             << maxSharedPerBlock << " bytes per block.\n";
        return r;
    }

    // Ask CUDA to permit an opt-in dynamic shared-memory size when required.
    if (v.sharedBytes > defaultSharedPerBlock) {
        if (v.sharedBytes > static_cast<size_t>(numeric_limits<int>::max())) {
            cout << "\nSKIPPED " << v.name << ": shared-memory size exceeds int range.\n";
            return r;
        }
        CUDA_CHECK(v.setDynamicSmem(static_cast<int>(v.sharedBytes)));
    }

    cudaFuncAttributes attr;
    CUDA_CHECK(v.getAttributes(&attr));
    r.registersPerThread = attr.numRegs;
    r.localBytesPerThread = static_cast<size_t>(attr.localSizeBytes);

    CUDA_CHECK(v.getOccupancy(&r.activeBlocksPerSM,
                              static_cast<int>(v.block.x * v.block.y),
                              v.sharedBytes));

    if (r.activeBlocksPerSM <= 0) {
        cout << "\nSKIPPED " << v.name
             << ": CUDA reports zero active blocks per SM for this configuration.\n";
        return r;
    }

    // Clear C so a missing output store cannot pass by reusing an old result.
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    // Warm-up (excluded from timing).
    v.launch(d_A, d_B, d_C, N, v.grid, v.block, v.sharedBytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> times;
    times.reserve(TIMED_RUNS);

    for (int run = 0; run < TIMED_RUNS; ++run) {
        CUDA_CHECK(cudaEventRecord(start));
        v.launch(d_A, d_B, d_C, N, v.grid, v.block, v.sharedBytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    sort(times.begin(), times.end());
    r.medianMs = times[times.size() / 2];
    r.averageMs = 0.0;
    for (float ms : times) {
        r.averageMs += ms;
    }
    r.averageMs /= static_cast<double>(times.size());

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));
    r.d2hMs = d2h;

    r.maxError = 0.0;
    r.correct = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        const double error = fabs(h_C[i] - h_reference[i]);
        r.maxError = max(r.maxError, error);
        if (error > ERROR_TOLERANCE) {
            r.correct = false;
        }
    }

    r.c0 = h_C[0];
    const double operations = 2.0 * static_cast<double>(N)
                            * static_cast<double>(N)
                            * static_cast<double>(N);
    r.gflops = (r.medianMs > 0.0)
             ? operations / (r.medianMs * 1.0e6)
             : 0.0;
    r.supported = true;
    return r;
}

static void printVariant(const Variant &v, int N, double h2dMs,
                         const Result &r)
{
    const double outM = static_cast<double>(C2_TILE_SIZE * v.rm);
    const double outN = static_cast<double>(C2_TILE_SIZE * v.rn);
    const double globalLoadsPerFma = 1.0 / outM + 1.0 / outN;
    const double sharedLoadsPerFma =
        static_cast<double>(v.rm + v.rn) /
        static_cast<double>(v.rm * v.rn);

    cout << "\n======= C2 TILING + REGISTER BLOCKING: "
         << v.name << " =======\n";
    cout << "Matrix size                 : " << N << " x " << N << '\n';
    cout << "T4-selected base tile       : " << C2_TILE_SIZE
         << " x " << C2_TILE_SIZE << '\n';
    cout << "K tile depth (BK)           : " << C2_TILE_SIZE << '\n';
    cout << "Register tile/thread       : " << v.rm << " x " << v.rn
         << " = " << v.rm * v.rn << " outputs\n";
    cout << "Output tile/block           : " << C2_TILE_SIZE * v.rm
         << " x " << C2_TILE_SIZE * v.rn << '\n';
    cout << "Threads per block           : " << v.block.x << " x "
         << v.block.y << " = " << v.block.x * v.block.y << '\n';
    cout << "Grid                        : " << v.grid.x << " x "
         << v.grid.y << " = " << v.grid.x * v.grid.y << " blocks\n";
    cout << "Shared-memory tile loads/FMA: " << globalLoadsPerFma
         << " (theoretical global loads)\n";
    cout << "Shared loads/FMA            : " << sharedLoadsPerFma
         << " (theoretical shared loads)\n";

    cout << "\n--- GPU Resource Usage ---\n";
    cout << "Registers per thread        : " << r.registersPerThread << '\n';
    cout << "Local memory per thread     : " << r.localBytesPerThread
         << " bytes";
    if (r.localBytesPerThread > 0) {
        cout << " (inspect ptxas output for spills)";
    }
    cout << '\n';
    cout << "Shared memory per block     : " << r.sharedBytes << " bytes\n";
    cout << "Active blocks per SM        : " << r.activeBlocksPerSM << '\n';

    cout << "\n--- Timing Results ---\n";
    cout << "H2D transfer time           : " << h2dMs << " ms\n";
    cout << "D2H transfer time           : " << r.d2hMs << " ms\n";
    cout << "Kernel median time          : " << r.medianMs << " ms\n";
    cout << "Kernel average time         : " << r.averageMs << " ms\n";

    cout << "\n--- Performance ---\n";
    cout << "GPU performance             : " << r.gflops << " GFLOP/s\n";

    cout << "\n--- Correctness ---\n";
    cout << "C[0]                        : " << r.c0 << '\n';
    cout << "Maximum absolute error      : " << r.maxError << '\n';
    cout << "Correctness                 : "
         << (r.correct ? "PASSED" : "FAILED") << '\n';
}

int main(int argc, char *argv[])
{
    int N = 256;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (N <= 0) {
        cerr << "Matrix size must be positive.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t maxSharedPerBlock =
        (prop.sharedMemPerBlockOptin > 0)
            ? static_cast<size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<size_t>(prop.sharedMemPerBlock);
    const size_t defaultSharedPerBlock =
        static_cast<size_t>(prop.sharedMemPerBlock);

    cout << fixed << setprecision(6);
    cout << "GPU                         : " << prop.name << " ("
         << prop.multiProcessorCount << " SMs, CC "
         << prop.major << "." << prop.minor << ")\n";
    cout << "Default shared memory/block: " << defaultSharedPerBlock
         << " bytes\n";
    cout << "Maximum shared memory/block: " << maxSharedPerBlock
         << " bytes\n";
    cout << "C2 selected T4 tile         : " << C2_TILE_SIZE
         << " x " << C2_TILE_SIZE << "; BK=" << C2_TILE_SIZE << "\n";
    cout << "Timed runs per variant      : " << TIMED_RUNS
         << " (after one warm-up)\n";

    const size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (elements > numeric_limits<size_t>::max() / sizeof(double)) {
        cerr << "Matrix allocation size overflows size_t.\n";
        return EXIT_FAILURE;
    }
    const size_t bytes = elements * sizeof(double);

    vector<double> h_A(elements);
    vector<double> h_B(elements);
    vector<double> h_C(elements);
    vector<double> h_reference(elements);

    // Deterministic input generation matching T3, T4, T5 and C1.
    for (size_t i = 0; i < elements; ++i) {
        h_A[i] =
            (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;
        h_B[i] =
            (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    cout << "\nComputing FP64 CPU reference...\n";
    cpuMatrixMultiply(h_A, h_B, h_reference, N);

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure input transfer once; all variants use the same device matrices.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));
    const double h2dMs = h2d;

    const dim3 block(C2_TILE_SIZE, C2_TILE_SIZE);

    char tags[4][64];
    snprintf(tags[0], sizeof(tags[0]), "c2_s%d_rm1x1", C2_TILE_SIZE);
    snprintf(tags[1], sizeof(tags[1]), "c2_s%d_rm2x2", C2_TILE_SIZE);
    snprintf(tags[2], sizeof(tags[2]), "c2_s%d_rm4x4", C2_TILE_SIZE);
    snprintf(tags[3], sizeof(tags[3]), "c2_s%d_rm8x8", C2_TILE_SIZE);

    Variant variants[4] = {
        { tags[0], "1x1 (tiled reference; no register blocking)",
          1, 1, launchVariant<1,1>, getVariantAttributes<1,1>,
          getVariantOccupancy<1,1>, setVariantDynamicSmem<1,1>,
          block, gridFor(N,1,1), sharedBytesFor(1,1) },
        { tags[1], "2x2 (4 outputs per thread)",
          2, 2, launchVariant<2,2>, getVariantAttributes<2,2>,
          getVariantOccupancy<2,2>, setVariantDynamicSmem<2,2>,
          block, gridFor(N,2,2), sharedBytesFor(2,2) },
        { tags[2], "4x4 (16 outputs per thread)",
          4, 4, launchVariant<4,4>, getVariantAttributes<4,4>,
          getVariantOccupancy<4,4>, setVariantDynamicSmem<4,4>,
          block, gridFor(N,4,4), sharedBytesFor(4,4) },
        { tags[3], "8x8 (64 outputs per thread; register-heavy)",
          8, 8, launchVariant<8,8>, getVariantAttributes<8,8>,
          getVariantOccupancy<8,8>, setVariantDynamicSmem<8,8>,
          block, gridFor(N,8,8), sharedBytesFor(8,8) }
    };

    Result results[4] = {};
    bool allCorrect = true;

    for (int i = 0; i < 4; ++i) {
        results[i] = runVariant(
            variants[i], d_A, d_B, d_C, h_C, h_reference, N, bytes,
            maxSharedPerBlock, defaultSharedPerBlock, start, stop);

        if (!results[i].supported) {
            continue;
        }

        printVariant(variants[i], N, h2dMs, results[i]);
        if (!results[i].correct) {
            allCorrect = false;
        }
    }

    // Pick the fastest supported configuration that passed correctness.
    int best = -1;
    for (int i = 0; i < 4; ++i) {
        if (!results[i].supported || !results[i].correct) {
            continue;
        }
        if (best < 0 || results[i].medianMs < results[best].medianMs) {
            best = i;
        }
    }

    cout << "\n======= C2 TILING + REGISTER BLOCKING SUMMARY (N = "
         << N << ", T4 tile = " << C2_TILE_SIZE << "x" << C2_TILE_SIZE
         << ") =======\n";
    printf("%-42s %12s %10s %9s %6s %8s %7s\n",
           "Configuration", "Median (ms)", "GFLOP/s", "vs 1x1",
           "Regs", "Smem B", "Blk/SM");

    for (int i = 0; i < 4; ++i) {
        if (!results[i].supported) {
            printf("%-42s %12s\n", variants[i].name, "SKIPPED");
            continue;
        }

        double speedup = 0.0;
        if (results[0].supported && results[0].correct &&
            results[i].medianMs > 0.0) {
            speedup = results[0].medianMs / results[i].medianMs;
        }

        printf("%-42s %12.6f %10.3f %8.2fx %6d %8zu %7d%s\n",
               variants[i].name,
               results[i].medianMs,
               results[i].gflops,
               speedup,
               results[i].registersPerThread,
               results[i].sharedBytes,
               results[i].activeBlocksPerSM,
               i == best ? "  <== best" : "");
    }

    if (best >= 0) {
        cout << "\nBest C2 configuration for N = " << N << ": "
             << variants[best].name << '\n';
        cout << "Median time                 : " << results[best].medianMs
             << " ms\n";
        cout << "Performance                 : " << results[best].gflops
             << " GFLOP/s\n";
        if (results[0].supported && results[0].correct) {
            cout << "Speedup vs C2 tiled 1x1     : "
                 << results[0].medianMs / results[best].medianMs << "x\n";
        }
    } else {
        cerr << "No supported C2 configuration passed correctness.\n";
        allCorrect = false;
    }

    // Same seven-column CSV format as T5:
    // RESULT,variant,N,median_ms,average_ms,gflops,max_abs_error
    cout << "\n";
    for (int i = 0; i < 4; ++i) {
        if (!results[i].supported) {
            printf("SKIP,%s,%d,unsupported_configuration\n", tags[i], N);
        } else {
            printf("RESULT,%s,%d,%.6f,%.6f,%.3f,%.3e\n",
                   tags[i], N,
                   results[i].medianMs,
                   results[i].averageMs,
                   results[i].gflops,
                   results[i].maxError);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return (allCorrect && best >= 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
