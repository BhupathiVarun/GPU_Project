#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;


#define BLOCK_X 32   // one warp wide: lanes map to consecutive columns
#define BLOCK_Y 8    // same 32 x 8 block as the best coalesced variant (t3)

static_assert(BLOCK_X == 32, "BLOCK_X must equal the warp size (32)");
static_assert(BLOCK_X * BLOCK_Y <= 1024, "block exceeds 1024 threads");

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

template <int RM, int RN>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
matmulRegBlocked(
    const double *A,
    const double *B,
    double *C,
    int N)
{
    const int rowBase = blockIdx.y * (BLOCK_Y * RM) + threadIdx.y;
    const int colBase = blockIdx.x * (BLOCK_X * RN) + threadIdx.x;

    // Row offsets of A and column indices of B, clamped so loads stay in range.
    int rowOff[RM];
    int colIdx[RN];

    #pragma unroll
    for (int i = 0; i < RM; i++)
    {
        int r = rowBase + i * BLOCK_Y;
        rowOff[i] = (r < N ? r : N - 1) * N;
    }

    #pragma unroll
    for (int j = 0; j < RN; j++)
    {
        int c = colBase + j * BLOCK_X;
        colIdx[j] = (c < N ? c : N - 1);
    }

    // RM * RN accumulators live in registers for the whole k loop.
    double acc[RM][RN];

    #pragma unroll
    for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++)
            acc[i][j] = 0.0;

    for (int k = 0; k < N; k++)
    {
        double a[RM];
        double b[RN];

        // RM + RN global loads ...
        #pragma unroll
        for (int i = 0; i < RM; i++)
            a[i] = A[rowOff[i] + k];

        const int kN = k * N;

        #pragma unroll
        for (int j = 0; j < RN; j++)
            b[j] = B[kN + colIdx[j]];

        // ... feed RM * RN multiply-adds.
        #pragma unroll
        for (int i = 0; i < RM; i++)
            #pragma unroll
            for (int j = 0; j < RN; j++)
                acc[i][j] += a[i] * b[j];
    }

    // Write the RM x RN results. For each j the lanes write consecutive columns.
    #pragma unroll
    for (int i = 0; i < RM; i++)
    {
        int r = rowBase + i * BLOCK_Y;

        if (r < N)
        {
            #pragma unroll
            for (int j = 0; j < RN; j++)
            {
                int c = colBase + j * BLOCK_X;

                if (c < N)
                    C[r * N + c] = acc[i][j];
            }
        }
    }
}

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
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
    }
}


struct Variant
{
    const char *tag;       // short id used in the RESULT line
    const char *name;      // heading
    int rm, rn;            // outputs per thread (rows x columns)
    KernelFn kernel;
    dim3 block;
    dim3 grid;
};

struct Result
{
    double d2h_ms;
    double median_ms;
    double average_ms;
    double gflops;
    double maxError;
    double c0;
    bool correct;
    int regsPerThread;
    size_t localBytes;     // > 0 means registers spilled to local memory
    int blocksPerSM;
};


static dim3 gridFor(int N, int rm, int rn)
{
    return dim3((N + BLOCK_X * rn - 1) / (BLOCK_X * rn),
                (N + BLOCK_Y * rm - 1) / (BLOCK_Y * rm));
}

// Resource usage, warm-up, five timed runs, copy back and verify one variant.
static Result runVariant(
    const Variant &v,
    const double *d_A, const double *d_B, double *d_C,
    vector<double> &h_C, const vector<double> &h_reference,
    int N, size_t bytes,
    cudaEvent_t start, cudaEvent_t stop)
{
    Result r;
    KernelFn kernel = v.kernel;

    // Registers per thread, spills and occupancy
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void *)kernel));
    r.regsPerThread = attr.numRegs;
    r.localBytes = attr.localSizeBytes;

    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, (const void *)kernel,
        v.block.x * v.block.y, 0));
    r.blocksPerSM = blocksPerSM;

    // Clear the output so a variant that fails to write cannot pass by
    // reusing the previous variant's result.
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    // Warm-up execution
    kernel<<<v.grid, v.block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel execution time
    vector<float> kernelTimes;

    for (int run = 0; run < 5; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        kernel<<<v.grid, v.block>>>(d_A, d_B, d_C, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        kernelTimes.push_back(milliseconds);
    }

    sort(kernelTimes.begin(), kernelTimes.end());

    r.median_ms = kernelTimes[kernelTimes.size() / 2];

    r.average_ms = 0.0;
    for (float t : kernelTimes)
        r.average_ms += t;
    r.average_ms /= kernelTimes.size();

    // Device-to-host transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));
    r.d2h_ms = d2h;

    // Correctness against the CPU reference
    r.correct = true;
    r.maxError = 0.0;

    for (size_t i = 0; i < h_C.size(); i++)
    {
        double error = fabs(h_C[i] - h_reference[i]);

        r.maxError = max(r.maxError, error);

        if (error > 1e-9)
            r.correct = false;
    }

    r.c0 = h_C[0];

    double operations = 2.0 * N * N * N;
    r.gflops = operations / (r.median_ms * 1e6);

    return r;
}

static void printVariant(const Variant &v, int N, double h2d_ms, const Result &r)
{
    double loadsPerFma = (double)(v.rm + v.rn) / (double)(v.rm * v.rn);

    cout << "\n======= REGISTER BLOCKING: " << v.name << " =======\n";

    cout << "Matrix size             : " << N << " x " << N << endl;
    cout << "Outputs per thread      : " << v.rm << " x " << v.rn
         << " = " << v.rm * v.rn << endl;
    cout << "Output tile per block   : " << BLOCK_Y * v.rm << " x "
         << BLOCK_X * v.rn << endl;
    cout << "Block shape             : " << v.block.x << " x " << v.block.y
         << " = " << v.block.x * v.block.y << " threads" << endl;
    cout << "Blocks in grid          : " << v.grid.x << " x " << v.grid.y
         << " = " << v.grid.x * v.grid.y << endl;
    cout << "Global loads per FMA    : " << loadsPerFma << " (theory)" << endl;

    cout << "\n--- Register Usage ---\n";

    cout << "Registers per thread    : " << r.regsPerThread << endl;
    cout << "Local memory (spills)   : " << r.localBytes << " bytes"
         << (r.localBytes > 0 ? "  <-- registers spilled" : "") << endl;
    cout << "Active blocks per SM    : " << r.blocksPerSM << endl;

    cout << "\n--- Timing Results ---\n";

    cout << "H2D transfer time       : " << h2d_ms << " ms\n";
    cout << "D2H transfer time       : " << r.d2h_ms << " ms\n";
    cout << "Kernel median time      : " << r.median_ms << " ms\n";
    cout << "Kernel average time     : " << r.average_ms << " ms\n";

    cout << "\n--- Performance ---\n";

    cout << "GPU performance         : " << r.gflops << " GFLOP/s\n";

    cout << "\n--- Correctness ---\n";

    cout << "C[0]                    : " << r.c0 << endl;
    cout << "Maximum absolute error  : " << r.maxError << endl;

    cout << "Correctness             : "
         << (r.correct ? "PASSED" : "FAILED") << endl;
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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU : " << prop.name << " (" << prop.multiProcessorCount
         << " SMs, CC " << prop.major << "." << prop.minor << ")" << endl;

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(double);

    vector<double> h_A(elements);
    vector<double> h_B(elements);
    vector<double> h_C(elements);
    vector<double> h_reference(elements);

    // Same deterministic inputs as the CPU and other CUDA programs
    for (size_t i = 0; i < elements; i++)
    {
        h_A[i] =
            (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;

        h_B[i] =
            (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
    }

    // CPU reference (computed once, shared by all variants)
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

    // Measure host-to-device transfer time (shared by all variants)
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));
    double h2d_ms = h2d;

    dim3 block(BLOCK_X, BLOCK_Y);

    const int numVariants = 4;

    Variant variants[numVariants] = {
        { "t5_regblock_1x1", "1 x 1 (reference, no register blocking)",
          1, 1, matmulRegBlocked<1, 1>, block, gridFor(N, 1, 1) },

        { "t5_regblock_2x2", "2 x 2 (4 outputs per thread)",
          2, 2, matmulRegBlocked<2, 2>, block, gridFor(N, 2, 2) },

        { "t5_regblock_4x4", "4 x 4 (16 outputs per thread)",
          4, 4, matmulRegBlocked<4, 4>, block, gridFor(N, 4, 4) },

        { "t5_regblock_8x8", "8 x 8 (64 outputs per thread, register heavy)",
          8, 8, matmulRegBlocked<8, 8>, block, gridFor(N, 8, 8) }
    };

    Result results[numVariants];
    bool allCorrect = true;

    for (int i = 0; i < numVariants; i++)
    {
        results[i] = runVariant(variants[i], d_A, d_B, d_C, h_C,
                                h_reference, N, bytes, start, stop);

        printVariant(variants[i], N, h2d_ms, results[i]);

        if (!results[i].correct)
            allCorrect = false;
    }

    // Summary: the effect of register blocking is the ratio to the 1 x 1 run
    printf("\n======= REGISTER BLOCKING SUMMARY (N = %d) =======\n", N);
    printf("%-46s %12s %10s %9s %6s %7s\n",
           "Variant", "Median (ms)", "GFLOP/s", "vs 1x1", "Regs", "Spill");

    for (int i = 0; i < numVariants; i++)
    {
        printf("%-46s %12.6f %10.3f %8.2fx %6d %6zuB\n",
               variants[i].name,
               results[i].median_ms,
               results[i].gflops,
               results[0].median_ms / results[i].median_ms,
               results[i].regsPerThread,
               results[i].localBytes);
    }

    // One machine-readable line per variant
    // RESULT,variant,N,median_ms,average_ms,gflops,max_abs_error
    printf("\n");
    for (int i = 0; i < numVariants; i++)
    {
        printf("RESULT,%s,%d,%.6f,%.6f,%.3f,%.3e\n",
               variants[i].tag, N,
               results[i].median_ms, results[i].average_ms,
               results[i].gflops, results[i].maxError);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return allCorrect ? 0 : 1;
}