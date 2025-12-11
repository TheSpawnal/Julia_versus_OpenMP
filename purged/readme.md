# Julia PolyBench Suite

State-of-the-art Julia implementations of 6 PolyBench kernels designed for systematic performance evaluation against OpenMP/C.

## Benchmarks

| Benchmark | Pattern | Characteristics | Parallelism |
|-----------|---------|-----------------|-------------|
| **2MM** | D = alpha*A*B*C + beta*D | Compute-intensive, regular | High |
| **3MM** | E=A*B; F=C*D; G=E*F | High arithmetic intensity | High |
| **Cholesky** | A = L*L^T decomposition | Dependency-heavy, triangular | Medium |
| **Correlation** | Pearson correlation matrix | Streaming, reductions | High |
| **Jacobi-2D** | 5-point stencil iteration | Memory-bound, low arithmetic intensity | High |
| **Nussinov** | RNA folding DP | Wavefront, irregular | Limited |

## Project Structure

```
julia_polybench/
├── README.md
├── Project.toml
├── src/
│   ├── PolyBenchJulia.jl       # Main module
│   ├── common/
│   │   ├── BenchCore.jl        # Core benchmarking utilities
│   │   ├── Metrics.jl          # Performance metrics collection
│   │   └── Config.jl           # Dataset configurations
│   └── kernels/
│       ├── TwoMM.jl            # 2MM implementations
│       ├── ThreeMM.jl          # 3MM implementations
│       ├── Cholesky.jl         # Cholesky implementations
│       ├── Correlation.jl      # Correlation implementations
│       ├── Jacobi2D.jl         # Jacobi-2D stencil implementations
│       └── Nussinov.jl         # Nussinov implementations
├── scripts/
│   ├── run_2mm.jl
│   ├── run_3mm.jl
│   ├── run_cholesky.jl
│   ├── run_correlation.jl
│   ├── run_jacobi2d.jl
│   └── run_nussinov.jl
├── test/
│   └── runtests.jl
└── results/                    # Output directory
```

## Quick Start

### Prerequisites
- Julia >= 1.10
- Packages: BenchmarkTools, Printf, LinearAlgebra, Statistics

### Running Individual Benchmarks

```bash
# Multi-threaded (recommended for most benchmarks)
julia -t 8 scripts/run_2mm.jl --dataset MEDIUM

# Multi-process (for distributed implementations)
julia scripts/run_cholesky.jl --distributed --workers 4 --dataset LARGE

# All strategies with profiling output
julia -t 8 scripts/run_correlation.jl --dataset SMALL --profile
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | MINI, SMALL, MEDIUM, LARGE, EXTRALARGE | MEDIUM |
| `--strategies` | Comma-separated list or "all" | all |
| `--iterations` | Number of timed iterations | 10 |
| `--warmup` | Number of warmup iterations | 5 |
| `--distributed` | Enable distributed computing | false |
| `--workers` | Number of worker processes | 4 |
| `--profile` | Generate profiling output | false |
| `--output` | Results output format (csv, json) | csv |

## Design Principles

### 1. Column-Major Optimization
All 2D array operations respect Julia's column-major layout:
```julia
# CORRECT: Column-major access
@inbounds for j in 1:n, i in 1:m
    A[i, j] = ...
end

# WRONG: Row-major access (cache-hostile)
@inbounds for i in 1:m, j in 1:n
    A[i, j] = ...
end
```

### 2. Zero-Allocation Hot Paths
Memory allocation in iteration loops must be zero:
```julia
# Pre-allocate outside loop
buffer = Vector{Float64}(undef, n)

for iter in 1:iterations
    # Use inplace operations
    copyto!(buffer, source)  # NOT: buffer = copy(source)
    fill!(result, 0.0)       # NOT: result = zeros(n)
end
```

### 3. View-Based Slicing
Array slices use @view to avoid copies:
```julia
@views col = A[:, j]         # SubArray, no allocation
col = A[:, j]                # WRONG: creates copy
```

### 4. Thread Scheduling Control
```julia
@threads :static for i in 1:n   # Predictable work distribution
@threads :dynamic for i in 1:n  # Load balancing (default)
```

### 5. Session Isolation
```julia
# Multi-threading session
julia -t 16  # nprocs() == 1, Threads.nthreads() == 16

# Multi-processing session  
julia -p 8   # nworkers() == 8, Threads.nthreads() == 1
```

## Metrics Reported

### Primary Metrics (Flame Graph Compatible)
- `min_time_ms`: Minimum execution time
- `median_time_ms`: Median execution time
- `mean_time_ms`: Mean execution time
- `std_time_ms`: Standard deviation
- `gflops`: Achieved GFLOP/s
- `speedup`: Relative to sequential baseline
- `efficiency`: Parallel efficiency (speedup/threads * 100)
- `allocations`: Memory allocations count
- `memory_mb`: Peak memory usage

### Output Format (CSV)
```csv
benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,allocs,memory_mb
2mm,MEDIUM,sequential,1,125.3,126.1,127.2,1.4,2.1,1.00,100.0,0,0.0
2mm,MEDIUM,threads_static,8,18.2,18.9,19.1,0.6,14.1,6.68,83.5,0,0.0
```

## Profiling for Flame Graphs

```bash
# Generate profile data
julia -t 8 scripts/run_2mm.jl --dataset LARGE --profile

# Results saved to: results/profile_2mm_LARGE_TIMESTAMP.jlprof
```

Use with ProfileView.jl or export to SVG:
```julia
using ProfileView
ProfileView.svgwrite("flamegraph.svg")
```

## DAS-5 Cluster Deployment

```bash
# Request interactive node
srun -N 1 -n 1 -c 32 --time=01:00:00 --pty bash

# Load Julia module
module load julia/1.10

# Run benchmark
julia -t 32 scripts/run_2mm.jl --dataset LARGE --output csv
```

## Implementation Strategies

### Linear Algebra Kernels (2MM, 3MM)
1. Sequential baseline
2. Threaded static scheduling
3. Threaded dynamic scheduling
4. Tiled/blocked (cache-optimized)
5. BLAS-accelerated
6. Task-based

### Cholesky Decomposition
1. Sequential (ijk order)
2. SIMD-optimized
3. Threaded trailing update
4. Blocked/tiled
5. Right-looking parallel
6. Left-looking parallel
7. Task-based recursive

### Correlation
1. Sequential baseline
2. Row-wise parallel
3. Column-major optimized
4. Tiled with cache blocking
5. Reduction-based parallel

### Jacobi-2D (Stencil)
1. Sequential baseline (double buffering)
2. Threaded row-wise parallel
3. Red-black (Gauss-Seidel) ordering
4. Tiled/blocked for cache optimization
5. Wavefront parallelization

### Nussinov (DP Wavefront)
1. Sequential baseline
2. Wavefront/anti-diagonal parallel
3. Tiled wavefront
4. Task-based with dependencies
5. Pipeline parallel
6. Hybrid coarse+fine grained

## Verification

All implementations are verified against:
1. Sequential baseline (numerical agreement)
2. Reference BLAS/LAPACK results (where applicable)
3. Relative error threshold: 1e-10

```bash
# Run verification tests
julia test/runtests.jl
```

