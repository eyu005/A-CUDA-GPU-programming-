// attention_test.cu - Test harness for HW5: Attention
// CPSC 5600, Seattle University
//
// Generates random Q, K, V matrices, runs the attention implementation,
// and optionally compares against a reference output for correctness.
//
// Usage:
//   ./attention_test              Run and print timing
//   ./attention_test --check      Run and compare against reference output
//   ./attention_test --save-ref   Run naive and save reference output to file
//   ./attention_test --verbose    Print sample output values

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "attention.h"

// Reference output file for correctness checking
#define REF_FILE "reference_output.bin"

// Tolerance for floating point comparison
#define TOLERANCE 1e-3f

// ---- Utility functions ----

void fill_random(float *arr, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
}

float max_abs_diff(const float *a, const float *b, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

int count_mismatches(const float *a, const float *b, int size, float tol) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > tol)
            count++;
    }
    return count;
}

void print_sample(const char *name, const float *arr, int rows, int cols,
                   int num_rows, int num_cols) {
    printf("%s (first %d×%d):\n", name, num_rows, num_cols);
    for (int i = 0; i < num_rows && i < rows; i++) {
        printf("  [%3d] ", i);
        for (int j = 0; j < num_cols && j < cols; j++)
            printf("%8.4f ", arr[i * cols + j]);
        printf("...\n");
    }
    printf("\n");
}

// ---- GPU timing helper ----

typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

void timer_create(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}

void timer_start(GpuTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

float timer_stop(GpuTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}

void timer_destroy(GpuTimer *t) {
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
}

// ---- Main ----

int main(int argc, char **argv) {
    int check_ref = 0;
    int save_ref = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--check") == 0) check_ref = 1;
        else if (strcmp(argv[i], "--save-ref") == 0) save_ref = 1;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else {
            fprintf(stderr, "Usage: %s [--check] [--save-ref] [--verbose]\n", argv[0]);
            return 1;
        }
    }

    int N = SEQ_LEN;
    int d = HEAD_DIM;

    printf("Attention Test Harness\n");
    printf("======================\n");
    printf("Sequence length (N): %d\n", N);
    printf("Head dimension (d):  %d\n", d);
    printf("Tile size:           %d\n", TILE_SIZE);
    printf("\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max threads per block:   %d\n\n", prop.maxThreadsPerBlock);

    // Allocate host memory
    size_t qkv_size = N * d * sizeof(float);
    size_t out_size = N * d * sizeof(float);

    float *h_Q = (float *)malloc(qkv_size);
    float *h_K = (float *)malloc(qkv_size);
    float *h_V = (float *)malloc(qkv_size);
    float *h_O = (float *)malloc(out_size);

    // Use fixed seed for reproducibility
    unsigned int seed = 42;
    fill_random(h_Q, N * d, seed);
    fill_random(h_K, N * d, seed + 1);
    fill_random(h_V, N * d, seed + 2);

    if (verbose) {
        print_sample("Q", h_Q, N, d, 4, 8);
        print_sample("K", h_K, N, d, 4, 8);
        print_sample("V", h_V, N, d, 4, 8);
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_O, out_size));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));

    // Warmup run (first CUDA launch has overhead)
    CUDA_CHECK(cudaMemset(d_O, 0, out_size));
    run_attention(d_Q, d_K, d_V, d_O, N, d);

    // Timed run
    GpuTimer timer;
    timer_create(&timer);

    int num_runs = 10;
    float total_ms = 0.0f;

    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaMemset(d_O, 0, out_size));
        timer_start(&timer);
        run_attention(d_Q, d_K, d_V, d_O, N, d);
        float ms = timer_stop(&timer);
        total_ms += ms;
        if (verbose)
            printf("Run %d: %.3f ms\n", run + 1, ms);
    }

    float avg_ms = total_ms / num_runs;
    printf("Average time over %d runs: %.3f ms\n\n", num_runs, avg_ms);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O, d_O, out_size, cudaMemcpyDeviceToHost));

    if (verbose)
        print_sample("Output O", h_O, N, d, 4, 8);

    // Sanity checks
    int nan_count = 0;
    int inf_count = 0;
    for (int i = 0; i < N * d; i++) {
        if (isnan(h_O[i])) nan_count++;
        if (isinf(h_O[i])) inf_count++;
    }
    if (nan_count > 0)
        printf("WARNING: %d NaN values in output!\n", nan_count);
    if (inf_count > 0)
        printf("WARNING: %d Inf values in output!\n", inf_count);
    if (nan_count == 0 && inf_count == 0)
        printf("Sanity check: no NaN or Inf values in output. Good.\n");

    // Save reference output
    if (save_ref) {
        FILE *f = fopen(REF_FILE, "wb");
        if (!f) {
            fprintf(stderr, "Error: cannot open %s for writing\n", REF_FILE);
            return 1;
        }
        fwrite(h_O, sizeof(float), N * d, f);
        fclose(f);
        printf("Reference output saved to %s\n", REF_FILE);
    }

    // Check against reference
    if (check_ref) {
        FILE *f = fopen(REF_FILE, "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open %s\n", REF_FILE);
            fprintf(stderr, "Run with --save-ref first (using naive implementation).\n");
            return 1;
        }
        float *ref_O = (float *)malloc(out_size);
        size_t n_read = fread(ref_O, sizeof(float), N * d, f);
        fclose(f);

        if ((int)n_read != N * d) {
            fprintf(stderr, "Error: reference file has wrong size\n");
            free(ref_O);
            return 1;
        }

        float max_diff = max_abs_diff(h_O, ref_O, N * d);
        int mismatches = count_mismatches(h_O, ref_O, N * d, TOLERANCE);

        printf("\nCorrectness check against %s:\n", REF_FILE);
        printf("  Max absolute difference: %e\n", max_diff);
        printf("  Mismatches (>%e):       %d / %d\n", TOLERANCE, mismatches, N * d);

        if (mismatches == 0)
            printf("  PASSED ✓\n");
        else {
            printf("  FAILED ✗\n");

            // Show first few mismatches
            printf("  First mismatches:\n");
            int shown = 0;
            for (int i = 0; i < N * d && shown < 10; i++) {
                if (fabsf(h_O[i] - ref_O[i]) > TOLERANCE) {
                    int r = i / d, c = i % d;
                    printf("    O[%d][%d]: got %e, expected %e (diff %e)\n",
                           r, c, h_O[i], ref_O[i], fabsf(h_O[i] - ref_O[i]));
                    shown++;
                }
            }
        }

        free(ref_O);
    }

    // Cleanup
    timer_destroy(&timer);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return 0;
}
