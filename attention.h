// attention.h - Common definitions for HW5: Attention
// CPSC 5600, Seattle University

#ifndef ATTENTION_H
#define ATTENTION_H

#include <cuda_runtime.h>

// Fixed dimensions for this assignment
#define SEQ_LEN 1024   // N: sequence length (number of tokens)
#define HEAD_DIM 64    // d: head dimension
#define TILE_SIZE 32   // tile width for tiled kernels

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK_LAUNCH() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- Interface ----
// Each implementation provides this function.
// Q, K, V are device pointers to N×d matrices (row-major).
// O is a device pointer to the N×d output matrix (row-major).
void run_attention(const float *d_Q, const float *d_K, const float *d_V,
                   float *d_O, int N, int d);

#endif // ATTENTION_H
