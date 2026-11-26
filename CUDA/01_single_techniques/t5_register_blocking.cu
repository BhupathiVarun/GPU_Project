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
