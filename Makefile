# Makefile for HW5: Attention
# CPSC 5600, Seattle University
#
# Targets:
#   make naive      - Build with naive implementation
#   make tiled      - Build with tiled implementation (student code)
#   make solution   - Build with reference solution (DO NOT DISTRIBUTE)
#   make test       - Build naive, save reference, build tiled, check
#   make profile    - Profile the tiled implementation with nvprof
#   make clean      - Remove build artifacts

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_86
# For Pascal (csp239): -arch=sm_61
# For cs3 RTX 3080:    -arch=sm_86

HARNESS = attention_test.cu
HEADER = attention.h

.PHONY: all naive tiled solution test profile clean

all: tiled

naive: naive_attention
	@echo "Built naive implementation"

tiled: tiled_attention
	@echo "Built tiled implementation"

solution: solution_attention
	@echo "Built reference solution"

naive_attention: $(HARNESS) naive_attention.cu $(HEADER)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(HARNESS) naive_attention.cu

tiled_attention: $(HARNESS) tiled_attention.cu $(HEADER)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(HARNESS) tiled_attention.cu

solution_attention: $(HARNESS) tiled_attention_solution.cu $(HEADER)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(HARNESS) tiled_attention_solution.cu

# Generate reference output from naive, then check tiled against it
test: naive_attention tiled_attention
	@echo "=== Generating reference output from naive implementation ==="
	./naive_attention --save-ref
	@echo ""
	@echo "=== Checking tiled implementation against reference ==="
	./tiled_attention --check

# Profile the tiled implementation
profile: tiled_attention
	nvprof ./tiled_attention

clean:
	rm -f naive_attention tiled_attention solution_attention
	rm -f reference_output.bin
