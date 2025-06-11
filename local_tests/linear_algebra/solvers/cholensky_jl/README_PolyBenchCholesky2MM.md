# PolyBench Cholesky Decomposition Benchmark for Julia

This module implements the Cholesky decomposition kernel from the PolyBench benchmark suite in Julia, with various optimization strategies.

## Overview

The Cholesky decomposition decomposes a positive-definite matrix A into the product of a lower triangular matrix L and its transpose: A = L × L^T.

The implementation follows the same algorithm as the original PolyBench C code:
- For each column j:
  - Update elements below the diagonal using previously computed columns
  - Compute the diagonal element
- Zero out the upper triangle

## Implementations

### 1. **Sequential (seq)**
- Direct translation of the C implementation
- Three nested loops following the mathematical definition
- Baseline for performance comparisons

### 2. **SIMD (simd)**
- Uses Julia's `@simd` macro for vectorization
- Optimizes inner loops for SIMD instructions
- Typically 2-4x faster than sequential

### 3. **Multithreaded (threads)**
- Parallelizes independent computations within each column
- Uses `@threads` for row updates below the diagonal
- Performance scales with number of threads

### 4. **BLAS (blas)**
- Uses Julia's built-in optimized `cholesky` function
- Leverages highly optimized LAPACK routines
- Usually the fastest single-node implementation

### 5. **Blocked (blocked)**
- Cache-optimized implementation with tiling
- Processes matrix in blocks to improve cache locality
- Tuned tile size of 64 for modern CPUs

### 6. **Distributed (distributed)**
- Uses `@distributed` macro for parallel computation
- Distributes row computations across workers
- Suitable for larger matrices

### 7. **Column-wise Distributed (dist_col)**
- More sophisticated distributed algorithm
- Minimizes communication by sending column data once
- Better scaling for distributed systems

## Key Differences from 2MM Benchmark

1. **Dependencies**: Cholesky has strong data dependencies - each column depends on all previous columns. This limits parallelization opportunities compared to matrix multiplication.

2. **In-place computation**: Cholesky modifies the input matrix in-place, while 2MM uses separate output matrices.

3. **Numerical stability**: Cholesky requires the input to be positive-definite. Non-positive-definite matrices will cause sqrt of negative numbers.

4. **Communication patterns**: In distributed implementations, Cholesky requires more frequent synchronization due to column dependencies.

## Performance Characteristics

### Sequential Complexity
- Time: O(n³/3) - about 1/3 the operations of matrix multiplication
- Space: O(1) - operates in-place

### Parallel Scalability
- Limited by sequential column processing
- Within each column, n-j independent row updates
- Communication overhead: O(n²) data movement per column

### Optimization Strategies
1. **Vectorization**: Inner loops computing dot products
2. **Blocking**: Improve cache usage with tiled access
3. **Parallelization**: Row updates within each column
4. **Distribution**: Partition rows across workers

## Usage Tips

1. **Matrix Preparation**: Ensure input matrix is symmetric positive-definite
2. **Thread Count**: Set `JULIA_NUM_THREADS` for best multithread performance
3. **Worker Processes**: Add workers before running distributed implementations
4. **Dataset Size**: Choose appropriate size for your system memory

## Verification

The module includes verification against Julia's built-in Cholesky decomposition. All implementations should produce results within numerical tolerance (typically 1e-10 to 1e-14).

## Common Issues

1. **Non-positive-definite matrices**: Will cause domain errors (sqrt of negative)
2. **Memory allocation**: Large matrices may require significant memory
3. **Load imbalance**: Column-by-column processing can cause uneven work distribution
4. **Communication overhead**: Distributed implementations may be slower for small matrices