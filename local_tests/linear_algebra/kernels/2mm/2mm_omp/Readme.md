# 2mm Benchmark with OpenMP on Windows

This project reproduces the 2mm benchmark from PolyBench/C using OpenMP parallelization on Windows, similar to the Julia implementation provided in the original files.

## What is 2mm?

The 2mm benchmark is a linear algebra kernel that consists of two matrix multiplications:
- Input: Matrices A(NI×NK), B(NK×NJ), C(NJ×NL), D(NI×NL), and scalars α and β
- Output: D = α×A×B×C + β×D

## Parallelization Strategies

This implementation includes 4 different versions:

1. **Sequential (baseline)**: The original non-parallel implementation
2. **Fine-grained parallelism**: Uses OpenMP to parallelize the outer loops
3. **Coarse-grained parallelism**: Divides work by assigning chunks of rows to each thread
4. **Improved implementation**: Uses dynamic scheduling and other optimizations

## Setup
- Windows OS
- GCC compiler with OpenMP support MinGW-w64 

## Building the Benchmark
1. Clone the repository
2. Open a command prompt in the project directory
3. Run the make command:
4. This will compile the 2mm benchmark with OpenMP support.

## Running the Benchmark
You can run the benchmark with different configurations:

```
# Run with default settings (4 threads, small dataset)
./2mm_openmp

# Run with custom number of threads (e.g., 8 threads)
./2mm_openmp 8

# Run with custom dataset size (0=small, 1=medium, 2=large)
./2mm_openmp 4 2

# Run all benchmarks with different thread counts
make benchmark
```

## Dataset Sizes

The benchmark includes three dataset sizes from PolyBench/C:
- **Small**: A(40×70), B(70×50), C(50×80), D(40×80)
- **Medium**: A(180×210), B(210×190), C(190×220), D(180×220)
- **Large**: A(800×1100), B(1100×900), C(900×1200), D(800×1200)

## Understanding the Output

The benchmark outputs:
- Matrix sizes being used
- Number of threads
- For each implementation:
  - Execution time (in seconds)
  - Speedup relative to sequential (higher is better)
  - Parallel efficiency percentage (how well the parallelism is utilized)

## Implementation Details
### Sequential (baseline)
Original implementation with no parallelism.

### Fine-grained Parallelism
Simply adds `#pragma omp parallel for` to the outer loops, letting OpenMP handle the distribution of iterations to threads.

### Coarse-grained Parallelism
More explicitly controls the work distribution by:
1. Calculating thread-specific row ranges
2. Each thread handles its own chunk of rows
3. Uses barriers to synchronize between matrix multiplication phases

### Improved Implementation
Combines multiple optimizations:
1. Uses dynamic scheduling for better load balancing
2. Uses temporary scalar variables for better cache utilization
3. Applies chunk size tuning (16 in this case)

## Comparing with Julia Implementation
This OpenMP implementation mirrors the approach taken in the Julia benchmark:
- Sequential implementation is the baseline
- Task-based approach in Julia corresponds to fine-grained OpenMP
- Distributed coarse-grained approach in Julia corresponds to coarse-grained OpenMP




# 2mm OpenMP Performance Analysis Guide
This guide explains the performance characteristics of the different parallelization strategies implemented for the 2mm benchmark and helps you understand the benchmark results.

## Understanding Matrix Multiplication Performance

The 2mm benchmark performs two matrix multiplications:
1. tmp = α⋅A⋅B
2. D = β⋅D + tmp⋅C

Each matrix multiplication has O(n³) computational complexity, making this a compute-intensive task that should benefit significantly from parallelization.

## Key Performance Factors

### 1. Memory Access Patterns

Matrix multiplication performance is heavily influenced by memory access patterns:

- **Temporal locality**: Reusing data already in cache
- **Spatial locality**: Accessing contiguous memory locations

Our implementations store matrices in row-major order, which means:
- Row-wise operations have good spatial locality
- Column-wise operations have poor spatial locality

### 2. Work Distribution

How work is distributed among threads affects performance:

- **Load balancing**: Ensuring each thread gets an equal amount of work
- **Granularity**: The size of work chunks assigned to each thread
- **Overhead**: The cost of thread management and synchronization

### 3. Resource Contention

Multiple threads competing for shared resources can cause:

- **Cache contention**: Threads invalidating each other's cache lines
- **Memory bandwidth limitations**: Parallel speed-up limited by memory throughput
- **False sharing**: Multiple threads accessing different data on the same cache line

## Analyzing the Implementations

### Sequential Implementation

The baseline sequential implementation performs two nested triple loops with O(n³) computational complexity.

**Performance characteristics:**
- Single-threaded execution
- No parallelization overhead
- Limited by single-core performance

### Fine-grained Parallelism

This implementation uses `#pragma omp parallel for` to parallelize the outer loops.

**Performance characteristics:**
- Easy to implement
- OpenMP handles the distribution of loop iterations
- Little control over work distribution
- Potential for unbalanced work distribution
- Good for uniformly-sized work units

### Coarse-grained Parallelism

This implementation manually divides rows among threads, with each thread processing its own chunk.

**Performance characteristics:**
- More control over work distribution
- Reduced parallelization overhead
- Better data locality (each thread works on a contiguous chunk)
- Requires explicit synchronization (barrier)
- Good for larger datasets where overhead is amortized

### Improved Implementation

This combines various optimizations to enhance performance.

**Performance characteristics:**
- Dynamic scheduling balances the workload automatically
- Scalar accumulation improves register usage
- Chunk size tuning reduces scheduling overhead
- Balance between parallelization and efficient memory access

## Interpreting Benchmark Results

### Speedup

Speedup is calculated as:
```
Speedup = Sequential_Time / Parallel_Time
```

Ideal speedup equals the number of threads, but this is rarely achieved due to:
- Amdahl's Law (not all code can be parallelized)
- Overhead of thread creation and management
- Memory bandwidth limitations
- Cache contention

### Efficiency

Efficiency measures how effectively we're using our threads:
```
Efficiency = (Speedup / Number_of_Threads) * 100%
```

Ideal efficiency is 100%, meaning perfect linear speedup.

### Expected Performance Patterns

1. **Small datasets**:
   - Overhead may outweigh benefits
   - Fine-grained might perform worse than sequential
   - Coarse-grained should perform better due to lower overhead

2. **Medium datasets**:
   - All parallel implementations should show speedup
   - Improved implementation should perform best

3. **Large datasets**:
   - Maximum benefit from parallelization
   - May become memory-bound rather than compute-bound
   - Cache efficiency becomes more important

4. **Scaling with threads**:
   - Diminishing returns with more threads
   - May see decreasing efficiency as threads increase
   - Performance may degrade with too many threads (overhead)

## Optimizing Further

If you want to improve performance further, consider:

1. **Blocking/Tiling**: Reorganize computations to improve cache utilization
2. **SIMD Vectorization**: Use compiler options or intrinsics for SIMD operations
3. **Memory Prefetching**: Hint the processor to load data before it's needed
4. **Loop Unrolling**: Reduce loop overhead by processing multiple elements per iteration
5. **Alternative Matrix Storage**: Using block or tile-based storage instead of row-major

## Limitations on Windows

Windows systems may show different performance characteristics than Linux:
- Thread creation is more expensive on Windows
- NUMA awareness may be less efficient
- Process/thread scheduling may differ

## Comparing with Julia Implementation

The Julia implementation used:
1. Sequential implementation (baseline)
2. Task-based implementation (similar to our fine-grained OpenMP)
3. Distributed implementation (similar to our coarse-grained OpenMP)

Key differences to consider:
- Julia's task system vs. OpenMP's thread pool
- Julia's JIT compilation vs. C's ahead-of-time compilation
- Memory management differences
- Different runtime environments

When comparing results, focus on relative speedups rather than absolute performance.
