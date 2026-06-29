// Eunsang Yu 
// CPSC 5600, Seattle University
// tiled_attention.cu - Tiled implementation (YOUR CODE HERE)
// Implement attention using tiled matrix multiplication with shared memory.
// This should produce the same results as naive_attention.cu but faster.
//
// The key optimization: instead of each thread reading from global memory
// for every multiply-add, we collaboratively load tiles into shared memory
// and compute from there.
//
// Overview of what you need to implement:
//   1. Tiled S = Q · Kᵀ       -- Part 1 (60 pts)
//   2. Row-wise softmax of S   -- Part 2 (30 pts)
//   3. Tiled O = softmax(S) · V -- Part 2 (included in 30 pts)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "attention.h"

// ============================================================
// Part 1: Tiled matrix multiply for S = Q · Kᵀ
// ============================================================
//
// Recall the tiled matmul pattern:
//   - Each block computes a TILE_SIZE × TILE_SIZE tile of the output
//   - For each "step" along the shared dimension (d=64):
//       1. Load a tile of Q into shared memory
//       2. Load a tile of Kᵀ into shared memory
//       3. __syncthreads()
//       4. Each thread accumulates partial dot products
//       5. __syncthreads()
//   - Write final results to global memory
//
// Key detail: since we're computing Q · Kᵀ, the K matrix is accessed
// in transposed order. If S[row][col] = dot(Q[row], K[col]), then
// we need K[col][k] which is K stored in row-major at K[col * d + k].
//
// Grid: (N/TILE_SIZE) × (N/TILE_SIZE) blocks
// Block: TILE_SIZE × TILE_SIZE threads (32×32 = 1024)
// Shared memory: two tiles of TILE_SIZE × TILE_SIZE floats

__global__ void tiled_matmul_QKt(const float *Q, const float *K, float *S,
                                  int N, int d) {
    // Shared-memory tiles for a block of Q and a transposed block of K.
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Kts[TILE_SIZE][TILE_SIZE];

    // Global coordinates of the S element computed by this thread.
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // TODO: Accumulate the dot product across tiles
    float sum = 0.0f;
    
    int numTiles = (d + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
    
        // TODO: Collaboratively load tile of Q into Qs
        // What global memory index maps to Qs[threadIdx.y][threadIdx.x]?
        // Be careful with bounds checking (the tile may extend past d).

        // Load one element of the current Q tile into shared memory.
        int sCol = t * TILE_SIZE + threadIdx.x;
        // Checking bounds to see if tile extend past d
        if (row < N && sCol < d) {
            Qs[threadIdx.y][threadIdx.x] = Q[row * d + sCol];
        } else {
            Qs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load one element of the current K tile into shared memory.
        // We read K in row-major order but arrange it in shared memory so it is
        // used like a tile of Kᵀ during the dot-product computation.

        int kRow = blockIdx.x * TILE_SIZE + threadIdx.x;
        int kCol = t * TILE_SIZE + threadIdx.y;
        if (kRow < N && kCol < d) {
            Kts[threadIdx.y][threadIdx.x] = K[kRow * d + kCol];
        } else {
            Kts[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
    
        // Accumulate this tile's contribution to the dot product.
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Qs[threadIdx.y][k] * Kts[k][threadIdx.x];
        }
    
        __syncthreads();
    }

    // Write result to S (with bounds check)
    if (row < N && col < N)
        S[row * N + col] = sum;
}


// ============================================================
// Part 2: Softmax (row-wise)
// ============================================================
//
// You may reuse the naive softmax kernel or write your own.
// The naive approach (reading/writing S from global memory) is fine
// for Part 2. Fusing softmax into the tiled approach is extra credit.
//
// Softmax for row i:
//   1. m = max(S[i][0..N-1])              -- for numerical stability
//   2. exp_vals[j] = exp(S[i][j] - m)     -- shifted exponentials
//   3. sum = Σ exp_vals[j]                 -- normalization factor
//   4. S[i][j] = exp_vals[j] / sum        -- normalize in-place

__global__ void softmax_kernel(float *S, int N) {
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


// ============================================================
// Part 2 (continued): Tiled O = softmax(S) · V
// ============================================================
//
// This is another tiled matrix multiply, but now:
//   - A = softmax(S), which is N×N
//   - B = V, which is N×d
//   - O = A · B, which is N×d
//
// This is a standard tiled matmul (no transpose needed).
// The shared dimension is N (not d), so you'll have more tiles to
// iterate over: numTiles = N / TILE_SIZE = 1024 / 32 = 32.

__global__ void tiled_matmul_SV(const float *S, const float *V, float *O,
                                 int N, int d) {
    // Implement tiled matrix multiply for O = S · V
    // This is very similar to tiled_matmul_QKt but without the transpose.
    // Grid: ((d + TILE_SIZE-1)/TILE_SIZE) × ((N + TILE_SIZE-1)/TILE_SIZE)
    // The shared dimension to tile over is N (the columns of S / rows of V).
    __shared__ float Ss[TILE_SIZE][TILE_SIZE];
    __shared__ float Vs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of S
        int sCol = t * TILE_SIZE + threadIdx.x;

        if (row < N && sCol < N) {
            Ss[threadIdx.y][threadIdx.x] = S[row * N + sCol];
        } else {
            Ss[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of V
        int vRow = t * TILE_SIZE + threadIdx.y;
        if (vRow < N && col < d) {
            Vs[threadIdx.y][threadIdx.x] = V[vRow * d + col];
        } else {
            Vs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Ss[threadIdx.y][k] * Vs[k][threadIdx.x];
        }

        __syncthreads();
    }
    // Write result
    if (row < N && col < d) {
        O[row * d + col] = sum;
    }
}

// EXTRA CREDIT: Causal masking (5 pts)
// Implements lower-triangular attention
// Position i can only attend to positions j <= i.
// Sets S[i][j] = -INFINITY for j > i so softmax maps those to 0.
__global__ void causal_mask_kernel(float *S, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row < N && col < N && col > row)
        S[row * N + col] = -INFINITY;
}

// ---- Entry point (modify launch configs as needed) ----
void run_attention(const float *d_Q, const float *d_K, const float *d_V,
                   float *d_O, int N, int d) {
    // Allocate intermediate matrix S (N×N)
    float *d_S;
    CUDA_CHECK(cudaMalloc(&d_S, N * N * sizeof(float)));

    // Step 1: S = Q · Kᵀ (tiled)
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((N + TILE_SIZE - 1) / TILE_SIZE,
               (N + TILE_SIZE - 1) / TILE_SIZE);
    tiled_matmul_QKt<<<grid1, block1>>>(d_Q, d_K, d_S, N, d);
    CUDA_CHECK_LAUNCH();

    // Extra Credit: Causal mask -- future positions are masked before softmax.
    causal_mask_kernel<<<grid1, block1>>>(d_S, N);
    CUDA_CHECK_LAUNCH();

    // Step 2: softmax(S)
    // TODO: Choose appropriate launch configuration
    softmax_kernel<<<N, N, N * sizeof(float)>>>(d_S, N);
    CUDA_CHECK_LAUNCH();

    // Step 3: O = softmax(S) · V (tiled)
    dim3 block3(TILE_SIZE, TILE_SIZE);
    dim3 grid3((d + TILE_SIZE - 1) / TILE_SIZE,
               (N + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_SV<<<grid3, block3>>>(d_S, d_V, d_O, N, d);
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_S));
}
