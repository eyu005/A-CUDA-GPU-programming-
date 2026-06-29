# HW5: Attention — CPSC 5600, Seattle University

CUDA implementation of scaled dot-product attention, the core operation powering Transformer models. The project compares a naive global-memory baseline against an optimized tiled shared-memory version.

## Background

Attention computes:

```
O = softmax(Q · Kᵀ) · V
```

where Q, K, V are matrices of shape N×d (sequence length × head dimension). This assignment implements that formula on the GPU in two ways and measures the speedup from shared memory tiling.

**Fixed dimensions:**
- Sequence length N = 1024
- Head dimension d = 64
- Tile size = 32

## Files

| File | Description |
|---|---|
| `attention.h` | Shared constants, macros, and the `run_attention` interface |
| `naive_attention.cu` | Reference implementation — three separate kernel passes over global memory |
| `tiled_attention.cu` | Optimized implementation — tiled matrix multiply using shared memory |
| `attention_test.cu` | Test harness: timing, correctness checking, NaN/Inf detection |
| `Makefile` | Build targets for naive, tiled, testing, and profiling |

## How It Works

**Naive (`naive_attention.cu`)** computes attention in three steps, each reading and writing global GPU memory:
1. `S = Q · Kᵀ` — materializes the full N×N score matrix
2. `softmax(S)` — row-wise softmax with a parallel reduction
3. `O = softmax(S) · V` — final weighted sum

**Tiled (`tiled_attention.cu`)** replaces the global-memory matmuls with tiled matrix multiplication: threads collaboratively load 32×32 tiles of Q, K, and V into on-chip shared memory, then compute from there. This dramatically reduces expensive global memory traffic.

Also includes **causal masking** (extra credit): sets `S[i][j] = -∞` for `j > i` so each token can only attend to past positions.

## Building

Requires CUDA and `nvcc`. The Makefile targets `sm_86` (RTX 3080); adjust `-arch` for your GPU.

```bash
# Build naive reference
make naive

# Build tiled implementation
make tiled

# Build both, generate reference output, and verify correctness
make test

# Profile tiled implementation with nvprof
make profile

# Clean build artifacts
make clean
```

## Usage

```bash
# Run and print average kernel time
./tiled_attention

# Save naive output as reference
./naive_attention --save-ref

# Check tiled output against reference (tolerance 1e-3)
./tiled_attention --check

# Print sample input/output values and per-run timing
./tiled_attention --verbose
```

## Results

Running `make test` generates a reference output from the naive kernel, then verifies the tiled kernel matches it within a floating-point tolerance of `1e-3`.
