# Julia PolyBench Suite

State-of-the-art Julia implementations of PolyBench kernels for systematic multithreading performance evaluation against OpenMP/C.

## Benchmarks

| Benchmark | Pattern | Characteristics | Parallelism |
|-----------|---------|-----------------|-------------|
| **Cholesky** | A = L*L^T decomposition | Dependency-heavy, triangular | Medium |
| **Correlation** | Pearson correlation matrix | Streaming, reductions | High |
| **Jacobi-2D** | 5-point stencil iteration | Memory-bound, low arithmetic intensity | High 
| **Nussinov** | RNA folding DP | Wavefront, irregular | Limited |

## Design Principles
### Column-Major Optimization
```julia
# CORRECT: Column-major access (cache-friendly)
@inbounds for j in 1:n, i in 1:m
    A[i, j] = ...
end

# WRONG: Row-major access (cache-hostile)
@inbounds for i in 1:m, j in 1:n
    A[i, j] = ...
end
```

### Zero-Allocation Hot Paths
```julia
# Pre-allocate outside loop
buffer = Vector{Float64}(undef, n)

for iter in 1:iterations
    copyto!(buffer, source)  # NOT: buffer = copy(source)
    fill!(result, 0.0)       # NOT: result = zeros(n)
end
```

### Session Isolation
```bash
# Multi-threading (recommended)
julia -t 16 scripts/run_cholesky.jl --dataset LARGE

# Single-threaded baseline
julia -t 1 scripts/run_cholesky.jl --dataset LARGE
```

## Quick Start

### Running Benchmarks

```bash
# Cholesky decomposition
julia -t 8 scripts/run_cholesky.jl --dataset MEDIUM

# Correlation matrix
julia -t 8 scripts/run_correlation.jl --dataset MEDIUM

# Jacobi-2D stencil
julia -t 8 scripts/run_jacobi2d.jl --dataset MEDIUM

# Specific strategies only
julia -t 8 scripts/run_cholesky.jl --dataset LARGE --strategies sequential,blas,threads
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | MINI, SMALL, MEDIUM, LARGE, EXTRALARGE | MEDIUM |
| `--strategies` | Comma-separated list or "all" | all |
| `--iterations` | Number of timed iterations | 10 |
| `--warmup` | Number of warmup iterations | 5 |
| `--no-verify` | Skip result verification | false |
| `--output` | Output format (csv) | csv |

## Metrics Reported

### Primary Metrics (Flame Graph Compatible)
- `min_ms`: Minimum execution time (ms)
- `median_ms`: Median execution time
- `mean_ms`: Mean execution time
- `gflops`: Achieved GFLOP/s
- `speedup`: Relative to sequential baseline
- `efficiency`: Parallel efficiency (speedup/threads * 100)

### CSV Output Format
```csv
benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified
cholesky,MEDIUM,sequential,1,125.3,126.1,127.2,1.4,2.1,1.00,100.0,PASS
cholesky,MEDIUM,threads,8,18.2,18.9,19.1,0.6,14.1,6.68,83.5,PASS
```

## Implementation Strategies

### Cholesky Decomposition
1. Sequential (Cholesky-Banachiewicz)
2. SIMD-optimized dot products
3. Threaded (parallel trailing update)
4. BLAS-accelerated (LAPACK)
5. Tiled/blocked (cache-optimized)

### Correlation Matrix
1. Sequential baseline
2. Threaded (parallel over columns)
3. Column-major optimized (fused mean/stddev)
4. Tiled with cache blocking

### Jacobi-2D Stencil
1. Sequential (double buffering)
2. Threaded (row-wise parallel)
3. Tiled/blocked (cache optimization)
4. Red-black (Gauss-Seidel ordering)

## DAS-5 Cluster Deployment

```bash
# Request interactive node
srun -N 1 -n 1 -c 32 --time=01:00:00 --pty bash

# Load Julia module
module load julia/1.10

# Run benchmark
julia -t 32 scripts/run_cholesky.jl --dataset LARGE --output csv
```

## Project Structure

```
julia_polybench/
├── README.md
├── src/
│   ├── PolyBenchJulia.jl       # Main module
│   ├── common/
│   │   ├── Config.jl           # Dataset configurations
│   │   ├── Metrics.jl          # Performance metrics
│   │   └── BenchCore.jl        # Benchmarking utilities
│   └── kernels/
│       ├── Cholesky.jl
│       ├── Correlation.jl
│       └── Jacobi2D.jl
├── scripts/
│   ├── run_cholesky.jl
│   ├── run_correlation.jl
│   └── run_jacobi2d.jl
└── results/                    # Output directory
```

## Profiling for Flame Graphs

```julia
using Profile
using ProfileView

# Profile a benchmark
@profile run_benchmark(config)

# Export to SVG
ProfileView.svgwrite("flamegraph.svg")
```

## References

- PolyBench/C Benchmark Suite
- "Scientific Parallel Computing and Multithreading with Julia" (main.tex)
- Julia Manual: Multi-Threading
