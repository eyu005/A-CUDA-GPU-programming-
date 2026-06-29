// naive_attention.cu - Reference implementation (slow but correct)
// CPSC 5600, Seattle University
//
// This implementation computes attention in three separate steps,
// each reading/writing global memory:
//   1. S = Q · Kᵀ           (N×d) × (d×N) → (N×N)
//   2. P = softmax(S)        row-wise softmax of S
//   3. O = P · V             (N×N) × (N×d) → (N×d)
//
// This is memory-inefficient because the full N×N score matrix S
// must be materialized in global memory.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "attention.h"

// ---- Kernel 1: S = Q · Kᵀ ----
// Each thread computes one element of S.
// S[row][col] = dot(Q[row], K[col])  (K is transposed)
__global__ void naive_matmul_QKt(const float *Q, const float *K, float *S,
                                  int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];  // K transposed: K[col][k]
        }
        S[row * N + col] = sum;
    }
}

// Causal mask: sets S[i][j] = -INFINITY for j > i
__global__ void naive_causal_mask(float *S, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row < N && col < N && col > row)
        S[row * N + col] = -INFINITY;
}

// ---- Kernel 2: Row-wise softmax of S ----
// Each block handles one row. Uses a single block of 1024 threads
// (which equals N for our dimensions).
// Step 1: find max (reduction)
// Step 2: compute exp(S[i] - max), find sum (reduction)
// Step 3: divide by sum
__global__ void naive_softmax(float *S, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Load this thread's element
    float val = (tid < N) ? S[row * N + tid] : -INFINITY;

    // ---- Find row max ----
    sdata[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // ---- Compute exp(val - max) and find sum ----
    float exp_val = (tid < N) ? expf(val - row_max) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // ---- Normalize ----
    if (tid < N)
        S[row * N + tid] = exp_val / row_sum;
}

// ---- Kernel 3: O = P · V ----
// Each thread computes one element of O.
// O[row][col] = dot(P[row], V[:][col])
__global__ void naive_matmul_PV(const float *P, const float *V, float *O,
                                 int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += P[row * N + k] * V[k * d + col];
        }
        O[row * d + col] = sum;
    }
}

// ---- Entry point ----
void run_attention(const float *d_Q, const float *d_K, const float *d_V,
                   float *d_O, int N, int d) {
    // Allocate intermediate matrices
    float *d_S;  // N×N attention scores
    CUDA_CHECK(cudaMalloc(&d_S, N * N * sizeof(float)));

    // Step 1: S = Q · Kᵀ
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((N + TILE_SIZE - 1) / TILE_SIZE,
               (N + TILE_SIZE - 1) / TILE_SIZE);
    naive_matmul_QKt<<<grid1, block1>>>(d_Q, d_K, d_S, N, d);
    CUDA_CHECK_LAUNCH();

    // Causal mask
    naive_causal_mask<<<grid1, block1>>>(d_S, N);
    CUDA_CHECK_LAUNCH();

    // Step 2: softmax(S) -- one block per row, N threads per block
    naive_softmax<<<N, N, N * sizeof(float)>>>(d_S, N);
    CUDA_CHECK_LAUNCH();

    // Step 3: O = softmax(S) · V
    dim3 block3(TILE_SIZE, TILE_SIZE);
    dim3 grid3((d + TILE_SIZE - 1) / TILE_SIZE,
               (N + TILE_SIZE - 1) / TILE_SIZE);
    naive_matmul_PV<<<grid3, block3>>>(d_S, d_V, d_O, N, d);
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_S));
}
