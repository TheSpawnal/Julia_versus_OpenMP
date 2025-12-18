# Julia PolyBench Report 12172025

## Executive Summary

Deep analysis of `TheSpawnal/Julia_versus_OpenMP/julia_polybench/` reveals critical incoherences in shell scripts, SLURM configuration, metrics mechanics, and architectural patterns.

---

## 1. SHELL SCRIPTS - SLURM Alignment Issues

### 1.1 das5_single_node.sh - CRITICAL FIXES NEEDED

**Current Issues:**

| Line | Issue | Severity |
|------|-------|----------|
| `#SBATCH --cpus-per-task=32` | DAS-5 nodes have 16 cores (dual 8-core E5-2630-v3) | HIGH |
| `#SBATCH --partition=normal` | DAS-5 uses `defq` partition, not `normal` | HIGH |
| Missing `/etc/bashrc` source | Required for SLURM environment | HIGH |
| Missing `module load prun` | Required before SLURM on DAS-5 | MEDIUM |
| Missing 2mm/3mm benchmarks | Script only handles cholesky, correlation, jacobi2d | MEDIUM |

**DAS-5 Node Specs (per SLURM_PRUN_DAS5.txt):**
- Dual 8-core 2.4 GHz (E5-2630-v3) = 16 cores total
- 64 GB memory
- Default partition: `defq`
- Default max job time during day: 15 minutes

### 1.2 das5_deployment_guide.sh - Issues

| Issue | Current | Should Be |
|-------|---------|-----------|
| Partition name | `--partition=normal` | `--partition=defq` |
| CPU count | `--cpus-per-task=32` | `--cpus-per-task=16` |
| Missing bashrc | - | `. /etc/bashrc` |
| Missing lmod | - | `. /etc/profile.d/lmod.sh` |
| Job time | `--time=02:00:00` | `--time=00:15:00` (daytime) |

### 1.3 Missing Uniformized Shell Script Pattern

Current scripts lack consistent structure. Required pattern per DAS-5 docs:

```bash
#!/bin/bash
#SBATCH --time=00:15:00          # Max 15 min during daytime
#SBATCH -N 1                      # Nodes
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH --cpus-per-task=16        # Cores per task (max 16 on DAS-5)
#SBATCH --partition=defq          # Default queue

# CRITICAL: Source environment
. /etc/bashrc
. /etc/profile.d/lmod.sh

# Load modules
module load prun
module load julia/1.10

# Set thread environment
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

# Navigate and execute
cd $HOME/Julia_versus_OpenMP/julia_polybench
```

---

## 2. ARCHITECTURAL INCOHERENCES

### 2.1 Data Type Inconsistency

| File | Type Used | Should Be |
|------|-----------|-----------|
| `local_tests/.../PolyBench2MM_Improved.jl` | Float32 | Float64 |
| `julia_polybench/src/kernels/TwoMM.jl` | Float64 | Float64 |
| `local_tests/.../PolyBench3MM_improved.jl` | Float64 | Float64 |

**Impact:** Float32 precision causes verification failures when comparing with BLAS/OpenMP (typically double precision).

### 2.2 Function Signature Mismatch

```julia
# local_tests (WRONG ORDER)
kernel_2mm_seq!(alpha, beta, tmp, A, B, C, D)

# backbone (CORRECT - matches PolyBench C)
kernel_2mm_seq!(alpha, beta, A, B, tmp, C, D)
```

### 2.3 Missing Benchmarks in Shell Scripts

| Benchmark | das5_single_node.sh | run_*.jl exists |
|-----------|---------------------|-----------------|
| 2mm | NO | YES |
| 3mm | NO | YES |
| cholesky | YES | YES |
| correlation | YES | YES |
| jacobi2d | YES | YES |
| nussinov | NO | YES |

### 2.4 Strategy Naming Inconsistency

| Module | Strategies |
|--------|------------|
| local_tests/2mm | Sequential, Threads (static), BLAS, Tiled, Distributed |
| local_tests/3mm | seq, simd, threads, blas, tiled |
| backbone | sequential, threads_static, threads_dynamic, tiled, blas, tasks |

**Issue:** `simd` as separate strategy is redundant - sequential should ALWAYS use `@simd @inbounds`.

---

## 3. METRICS MECHANICS ISSUES

### 3.1 Efficiency Calculation Bug (FIXED in Metrics.jl)

The efficiency calculation is strategy-aware:

```julia
const NON_THREADED_STRATEGIES = Set(["sequential", "seq", "simd", "blas", "colmajor"])
const THREADED_STRATEGIES = Set(["threads", "threads_static", "threads_dynamic", "tiled", "tasks"])

# Non-threaded: efficiency = speedup * 100
# Threaded: efficiency = (speedup / threads) * 100
```

**Issue in Results:** CSV shows `efficiency=1834.6` for BLAS - this is CORRECT but confusing. BLAS achieves 18.35x speedup, so efficiency = 18.35 * 100 = 1835%. This indicates BLAS is leveraging optimizations beyond simple threading.

### 3.2 Timing Methodology

**Current approach (CORRECT):**
1. Warmup runs (not timed) - excludes JIT compilation
2. GC.gc() before timed runs
3. Multiple iterations with @elapsed
4. Report min/median/mean/std

**Issue:** Some scripts use `@belapsed` (BenchmarkTools), others use manual `@elapsed`. Should standardize.

### 3.3 Verification Tolerance

| Benchmark | Current Tolerance | Issue |
|-----------|-------------------|-------|
| 2mm (local) | 1e-5 (Float32) | Too tight for BLAS accumulation errors |
| 3mm | 1e-10 (Float64) | Appropriate |
| backbone | Scale-aware | CORRECT approach |

**Scale-aware formula (from run_2mm.jl):**
```julia
scale_factor = sqrt(Float64(ni) * Float64(nj) * Float64(nk) * Float64(nl))
tolerance = max(1e-10, 1e-14 * scale_factor)
```

---

## 4. MISSING COMPONENTS

### 4.1 Unified Runner Script

No `run_all.jl` exists despite being referenced in `das5_deployment_guide.sh`.

### 4.2 Profiling Integration

Scripts reference `--profile` flag but implementation is incomplete:

```julia
elseif ARGS[i] == "--profile"
    do_profile = true
    i += 1
```

No actual Profile.@profile invocation follows.

### 4.3 Flame Graph Export

`das5_deployment_guide.sh` shows example but no actual implementation:

```julia
ProfileView.svgwrite("flamegraph.svg")
```

This requires `ProfileView.jl` package which isn't in dependencies.

---

## 5. SCALING STUDY LOG ANALYSIS

The `scaling_study.log` file contains valuable data but is NOT in CSV format:

```
Strategy         |    Min(ms) | Median(ms) |   Mean(ms) |  GFLOP/s |  Speedup | Eff(%)
------------------------------------------------------------------------------------------
sequential       |      7.403 |      7.761 |      8.095 |     2.23 |     1.00x |  100.0
threads          |      0.859 |      0.891 |      0.896 |    19.25 |     8.62x |  215.5
```

**Issue:** Cannot be directly consumed by `visualize_benchmarks.py` which expects CSV format.

---

## 6. PRIORITY
### P0 - Critical (Before DAS-5)
1. Fix SLURM scripts for DAS-5 compatibility (16 cores, defq partition)
2. Add `/etc/bashrc` and `/etc/profile.d/lmod.sh` sourcing
3. Fix job time to 15 minutes max (daytime policy)

### P1 - High
4. Create unified `run_all.jl` script
5. Add 2mm/3mm to das5_single_node.sh
6. Standardize Float64 everywhere
7. Create scaling_study.log parser for CSV conversion

### P2 - Medium
8. Add ProfileView.jl to dependencies
9. Implement actual --profile flag functionality
10. Create flame graph export script
11. Fix efficiency display for BLAS (clarify in output)

### P3 - Low
12. Remove redundant "simd" strategy from 3mm
13. Unify @elapsed vs @belapsed usage
14. Add memory bandwidth metrics
