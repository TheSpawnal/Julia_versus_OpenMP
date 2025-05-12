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



