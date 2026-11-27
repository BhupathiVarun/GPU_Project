
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
