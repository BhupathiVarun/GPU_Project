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
