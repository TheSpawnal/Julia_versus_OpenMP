# Julia PolyBench Suite

High-performance Julia implementations of PolyBench kernels for systematic multithreading benchmarking against OpenMP/C.

## Benchmarks

| Benchmark | Computation | Pattern | Parallelism Potential |
|-----------|-------------|---------|----------------------|
| **2MM** | D = alpha*A*B*C + beta*D | Chained GEMM | High (embarrassingly parallel) |
| **3MM** | G = (A*B)*(C*D) | Triple GEMM | High |
| **Cholesky** | A = L*L^T | Triangular decomposition | Medium (dependencies) |
| **Correlation** | Pearson correlation matrix | Streaming reductions | High |
| **Jacobi-2D** | 5-point stencil iteration | Memory-bound | High |
| **Nussinov** | RNA folding DP | Wavefront | Limited (irregular) |

---

## Design Principles

### 1. Column-Major Memory Layout

Julia stores matrices in column-major order. Inner loops must traverse columns for cache efficiency.

```julia
# CORRECT: Column-major (cache-friendly)
# Inner loop traverses consecutive memory addresses
@inbounds for j in 1:n      # outer: columns
    @simd for i in 1:m      # inner: rows (contiguous in memory)
        A[i, j] = ...
    end
end

# WRONG: Row-major (cache-hostile)
# Inner loop causes cache thrashing
@inbounds for i in 1:m      # outer: rows
    @simd for j in 1:n      # inner: columns (stride = m)
        A[i, j] = ...       # ~10-100x slower for large matrices
    end
end
```

### 2. Zero-Allocation Hot Paths

Allocations in hot loops destroy performance. Pre-allocate everything outside timed regions.

```julia
# WRONG: Allocation per iteration
for iter in 1:iterations
    result = zeros(n)       # ALLOCATES every iteration
    buffer = copy(source)   # ALLOCATES every iteration
end

# CORRECT: Pre-allocate and reuse
result = Vector{Float64}(undef, n)
buffer = similar(source)

for iter in 1:iterations
    fill!(result, 0.0)      # NO allocation
    copyto!(buffer, source) # NO allocation
end
```

### 3. SIMD and Bounds Check Elimination

Always use `@simd` on innermost loops and `@inbounds` to eliminate bounds checking.

```julia
@inbounds for j in 1:n
    for k in 1:m
        val = alpha * B[k, j]
        @simd for i in 1:p
            C[i, j] += A[i, k] * val
        end
    end
end
```

### 4. BLAS Thread Configuration

When using Julia threads, disable BLAS internal threading to prevent oversubscription.

```julia
using LinearAlgebra

if Threads.nthreads() > 1
    BLAS.set_num_threads(1)  # Julia handles parallelism
else
    BLAS.set_num_threads(Sys.CPU_THREADS)  # Let BLAS parallelize
end
```

### 5. Closure Capture Optimization

Avoid capturing scalars in threaded regions - hoist multiplications outside inner loops.

```julia
# SUBOPTIMAL: alpha captured by closure
@threads for j in 1:n
    @simd for i in 1:m
        C[i,j] += alpha * A[i,k] * B[k,j]  # alpha captured
    end
end

# OPTIMAL: Pre-compute outside SIMD loop
@threads for j in 1:n
    @inbounds for k in 1:p
        b_kj = alpha * B[k, j]  # Hoisted
        @simd for i in 1:m
            C[i,j] += A[i,k] * b_kj
        end
    end
end
```

---

## Metrics Mechanics

### Timing Strategy

```
                    JIT Compilation
                         |
    +--------------------+--------------------+
    |                                         |
    v                                         v
[First Call] -----> [Warmup Runs] -----> [Timed Runs]
    |                    |                    |
    |                    |                    v
    |                    |            Collect times[]
    |                    |                    |
    v                    v                    v
 Excluded           Excluded            Report Stats
                                        (min/median/mean/std)
```

**Warmup Protocol:**
1. Execute kernel `warmup` times (default: 5)
2. Call `GC.gc()` to clear garbage
3. Begin timed iterations

**Why Minimum Time?**
- Minimum represents "clean" execution without OS interference
- Median shows typical behavior
- Mean captures overall cost including outliers

### FLOP Calculation

Each kernel has a defined FLOP count based on operation count:

```julia
# 2MM: D = alpha*A*B*C + beta*D
# First multiplication: tmp[ni,nj] = A[ni,nk] * B[nk,nj]
#   -> ni * nj * nk multiplications + ni * nj * (nk-1) additions
# Second multiplication: D[ni,nl] = tmp[ni,nj] * C[nj,nl]
#   -> ni * nl * nj multiplications + ni * nl * (nj-1) additions
# Scaling: ni * nl multiplications (beta*D) + ni * nl additions

flops_2mm(ni, nj, nk, nl) = 2 * ni * nj * nk + 2 * ni * nj * nl
```

### Efficiency Calculation

Efficiency interpretation differs by strategy type:

```julia
# Non-threaded strategies (sequential, blas, simd)
# efficiency = speedup * 100
# BLAS at 18.35x speedup -> 1835% efficiency (leveraging vectorization)

# Threaded strategies (threads_static, threads_dynamic, tiled, tasks)
# efficiency = (speedup / threads) * 100
# 8x speedup on 8 threads -> 100% efficiency (perfect scaling)
# 6x speedup on 8 threads -> 75% efficiency (sublinear scaling)
```

### Verification Tolerance

Scale-aware tolerance accounts for floating-point accumulation errors:

```julia
# Error accumulates as O(sqrt(n)) for well-conditioned operations
scale_factor = sqrt(Float64(ni * nj * nk * nl))
tolerance = max(1e-10, 1e-14 * scale_factor)

# For LARGE dataset (ni=800, nj=900, nk=1100, nl=1200):
# scale_factor ~ 30,983
# tolerance ~ 3.1e-10
```

### CSV Output Format

```csv
benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified
2mm,LARGE,sequential,8,145.234,147.891,148.432,2.341,12.45,1.00,100.0,PASS
2mm,LARGE,threads_static,8,21.456,22.103,22.567,0.892,84.23,6.70,83.8,PASS
2mm,LARGE,blas,8,8.234,8.567,8.891,0.234,219.45,17.27,1727.2,PASS
```

---

## DAS-5 Cluster Deployment

### Node Specifications

| Component | Specification |
|-----------|---------------|
| CPU | Dual Intel E5-2630-v3 (8 cores each) |
| Total Cores | 16 |
| Memory | 64 GB |
| Partition | defq (default) |
| Max Job Time (daytime) | 15 minutes |

### Quick Commands

```bash
# Check cluster status
sinfo
squeue -u $USER

# Submit single benchmark
sbatch das5_single_node.slurm 2mm LARGE 16

# Submit scaling study
sbatch das5_scaling_study.slurm 2mm LARGE

# Interactive session
srun -N 1 -n 1 -c 16 --time=00:15:00 --partition=defq --pty bash

# Inside interactive session
module load prun
module load julia/1.10
export JULIA_NUM_THREADS=16
julia -t 16 scripts/run_2mm.jl --dataset LARGE
```

### SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=julia_polybench
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# CRITICAL: Source DAS-5 environment
. /etc/bashrc
. /etc/profile.d/lmod.sh

module load prun
module load julia/1.10

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

cd $HOME/Julia_versus_OpenMP/julia_polybench
julia -t $JULIA_NUM_THREADS scripts/run_2mm.jl --dataset LARGE
```

---

## Profiling for Flame Graphs

### Setup

```julia
using Pkg
Pkg.add("Profile")
Pkg.add("ProfileView")
Pkg.add("FlameGraphs")
Pkg.add("ProfileSVG")
```

### Profile a Benchmark

```julia
using Profile
using ProfileSVG

# Include kernel
include("src/kernels/TwoMM.jl")
using .TwoMM

# Setup data
ni, nj, nk, nl = 800, 900, 1100, 1200
A = rand(Float64, ni, nk)
B = rand(Float64, nk, nj)
C = rand(Float64, nj, nl)
D = rand(Float64, ni, nl)
tmp = zeros(Float64, ni, nj)
alpha, beta = 1.5, 1.2

# Warmup (CRITICAL: exclude JIT from profile)
kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)

# Clear any previous profile data
Profile.clear()

# Profile multiple runs for statistical significance
@profile for _ in 1:50
    fill!(tmp, 0.0)
    kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)
end

# Export flame graph
ProfileSVG.save("flamegraph_2mm_threads.svg")
```

### Command-Line Profiling

```bash
# Profile with allocation tracking
julia --track-allocation=user -t 16 scripts/run_2mm.jl --dataset LARGE

# View allocation results
cat src/kernels/TwoMM.jl.*.mem

# Profile specific function
julia -t 16 -e '
using Profile, ProfileSVG
include("scripts/run_2mm.jl")
# After warmup, profile is in flamegraph_*.svg
'
```

---

## Project Structure

```
julia_polybench/
├── README.md
├── das5_single_node.slurm       # Single-node SLURM script
├── das5_scaling_study.slurm     # Thread scaling study script
├── visualize_benchmarks.py      # Plot generation
├── src/
│   ├── PolyBenchJulia.jl        # Main module
│   ├── common/
│   │   ├── Config.jl            # Dataset configurations, FLOP formulas
│   │   ├── Metrics.jl           # BenchmarkResult, MetricsCollector
│   │   └── BenchCore.jl         # Timing utilities
│   └── kernels/
│       ├── TwoMM.jl             # 2MM implementations
│       ├── ThreeMM.jl           # 3MM implementations
│       ├── Cholesky.jl          # Cholesky decomposition
│       ├── Correlation.jl       # Correlation matrix
│       ├── Jacobi2D.jl          # Jacobi stencil
│       └── Nussinov.jl          # RNA folding
├── scripts/
│   ├── run_2mm.jl               # 2MM benchmark runner
│   ├── run_3mm.jl               # 3MM benchmark runner
│   ├── run_cholesky.jl
│   ├── run_correlation.jl
│   ├── run_jacobi2d.jl
│   └── run_nussinov.jl
└── results/                     # CSV output directory
```

---

## Command Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | MINI, SMALL, MEDIUM, LARGE, EXTRALARGE | MEDIUM |
| `--strategies` | Comma-separated list or "all" | all |
| `--iterations` | Number of timed iterations | 10 |
| `--warmup` | Number of warmup iterations | 5 |
| `--no-verify` | Skip result verification | false |
| `--output` | csv for file export | csv |

---

## References

- PolyBench/C 4.2.1 Benchmark Suite
- Julia Manual: Multi-Threading
- Intel Optimization Guide for Haswell
- BLAS Technical Reference
