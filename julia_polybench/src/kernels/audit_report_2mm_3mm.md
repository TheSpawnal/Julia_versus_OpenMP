# Audit Report: 2MM and 3MM Julia Implementations

## Executive Summary

Deep analysis of `local_tests/linear_algebra/kernels/{2mm,3mm}/` reveals several critical issues affecting performance measurement accuracy, portability, and integration with the julia_polybench backbone.

---

## 1. Critical Issues

### 1.1 Data Type Inconsistency (SEVERITY: HIGH)

**Location**: `PolyBench2MM_Improved.jl`

```julia
# 2mm uses Float32
A = Matrix{Float32}(undef, ni, nk)
alpha[] = 1.5f0

# 3mm and backbone use Float64
A = Matrix{Float64}(undef, ni, nk)
alpha[] = 1.5
```

**Impact**: 
- Precision differences in verification tolerances
- BLAS performance varies between Float32/Float64
- Inconsistent comparison with OpenMP (which typically uses double)

**Fix**: Standardize on Float64 for fair comparison with C/OpenMP.

---

### 1.2 Function Signature Mismatch (SEVERITY: HIGH)

**Location**: `PolyBench2MM_Improved.jl` vs backbone `TwoMM.jl`

```julia
# local_tests (WRONG ORDER)
function kernel_2mm_seq!(alpha, beta, tmp, A, B, C, D)

# backbone (CORRECT - matches PolyBench C)
function kernel_2mm_seq!(alpha, beta, A, B, tmp, C, D)
```

**Impact**: Integration with backbone requires complete rewiring.

---

### 1.3 Missing @simd in Sequential (SEVERITY: MEDIUM)

**Location**: `PolyBench3MM_improved.jl`

```julia
# Current (suboptimal)
function kernel_3mm_seq!(E, A, B, F, C, D, G)
    for j = 1:nj
        for k = 1:nk
            for i = 1:ni
                E[i,j] += A[i,k] * B[k,j]  # No @simd, no @inbounds
            end
        end
    end
```

**Impact**: 
- Sequential baseline is artificially slow
- Inflates threading speedup numbers
- Unfair comparison with SIMD-enabled OpenMP

---

### 1.4 Threading Allocations (SEVERITY: MEDIUM)

**Location**: Both implementations

```julia
# Potential allocation due to closure capture
@threads :static for j = 1:nj
    @inbounds for k = 1:nk
        @simd for i = 1:ni
            tmp[i,j] += alpha * A[i,k] * B[k,j]  # alpha captured
        end
    end
end
```

**Diagnosis**: The debug function shows allocations in threaded kernels.

**Fix**: Pre-compute `b_kj = alpha * B[k,j]` outside SIMD loop.

---

### 1.5 Non-parallelized Tiled Implementation (SEVERITY: HIGH)

**Location**: `PolyBench2MM_Improved.jl`

```julia
# Current: Sequential tiling (defeats purpose)
@inbounds for jj = 1:tile_size:nj
    for kk = 1:tile_size:nk
        for ii = 1:tile_size:ni
            # ... inner loops
        end
    end
end
```

**Expected**: Outer tile loop should be parallelized.

---

## 2. Design Issues

### 2.1 Redundant SIMD Strategy

**Location**: `PolyBench3MM_improved.jl`

```julia
const strategies = ["seq", "simd", "threads", "blas", "tiled"]
```

**Issue**: `kernel_3mm_simd!` is just `kernel_3mm_seq!` with `@simd`. The sequential baseline should ALWAYS use `@simd @inbounds` - this is not a separate strategy.

**Fix**: Remove redundant "simd" strategy; integrate into sequential.

---

### 2.2 Missing Strategies

| Strategy | 2mm local | 3mm local | Backbone |
|----------|-----------|-----------|----------|
| sequential | Y | Y | Y |
| simd (redundant) | N | Y | N |
| threads_static | Y | Y | Y |
| threads_dynamic | N | N | Y |
| tiled | Y (no threads) | Y (no threads) | Y |
| blas | Y | Y | Y |
| tasks | N | N | Y |
| distributed | Y | Y | N |

---

### 2.3 Inconsistent Initialization Pattern

**2mm init**:
```julia
function init_arrays!(alpha::Ref{Float32}, beta::Ref{Float32}, tmp, A, B, C, D)
```

**3mm init**:
```julia
function init_arrays!(A, B, C, D, E, F, G)  # No alpha/beta
```

---

## 3. Performance Measurement Issues

### 3.1 Benchmark Setup Overhead

```julia
# Current: Reset inside benchmark
b = @benchmarkable begin
    copyto!($D, $D_original)
    fill!($tmp, 0.0f0)
    $impl_func($alpha[], $beta[], $tmp, $A, $B, $C, $D)
end
```

**Issue**: `copyto!` and `fill!` are measured, polluting kernel time.

**Fix**: Use `setup` parameter of @benchmarkable.

---

### 3.2 Missing Metrics

Current metrics:
- Time (min, median, mean)
- Memory
- Allocations

Missing critical metrics:
- **GFLOP/s** (primary performance metric)
- **Speedup** (relative to sequential)
- **Efficiency** (speedup / nthreads * 100%)
- **Memory bandwidth** (for roofline analysis)

---

## 4. Correctness Issues

### 4.1 BLAS Thread Interaction

```julia
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
    end
end
```

**Issue**: When testing BLAS strategy with single Julia thread, BLAS gets all cores. When testing with multiple Julia threads, BLAS gets 1 thread. This makes BLAS appear slower in multi-threaded runs.

**Fix**: For fair comparison, BLAS should use consistent thread count, or we test BLAS with dedicated thread count.

---

### 4.2 Verification Tolerance

```julia
# 2mm: Float32, tolerance = 1e-5
tolerance=1e-5

# 3mm: Float64, tolerance = 1e-10
tolerance=1e-10
```

**Issue**: Inconsistent precision requirements.

---

## 5. Integration Issues with julia_polybench

### 5.1 Module Structure

**local_tests**:
```julia
module PolyBench2MM_Improved
    # Everything in one file
end
```

**backbone expected**:
```julia
# src/kernels/TwoMM.jl
module TwoMM
    export init_2mm!, kernel_2mm_seq!, ...
    export STRATEGIES_2MM, get_kernel
end
```

### 5.2 Missing Components for Integration

1. `STRATEGIES_2MM` constant
2. `get_kernel(strategy::String)` dispatcher
3. Consistent function signatures
4. Float64 data types
5. Proper return types (`nothing` vs return value)

---

## 6. Recommendations

### Immediate Fixes (Before DAS-5 Testing)

1. **Standardize Float64** across all implementations
2. **Fix function signatures** to match backbone pattern
3. **Add @simd @inbounds** to sequential baseline
4. **Parallelize tiled strategy** outer loops
5. **Remove redundant SIMD strategy**
6. **Add GFLOP/s metric** to all benchmarks

### Refactoring Plan

1. Create `TwoMM.jl` module matching backbone structure
2. Create `ThreeMM.jl` module matching backbone structure
3. Implement all 6 strategies consistently
4. Add proper metrics collection
5. Create unified runner scripts

### Testing Protocol

1. Verify correctness with `verify_implementations()`
2. Check zero-allocation in hot paths with `@allocated`
3. Run scaling tests (1, 2, 4, 8, 16, 32 threads)
4. Profile with `@profile` for flame graphs
5. Validate against OpenMP reference

---

## 7. Summary Table

| Issue | Severity | 2mm | 3mm | Status |
|-------|----------|-----|-----|--------|
| Float32 vs Float64 | HIGH | X | - | Fix |
| Signature mismatch | HIGH | X | - | Fix |
| Missing @simd seq | MEDIUM | - | X | Fix |
| Thread allocations | MEDIUM | X | X | Fix |
| Non-parallel tiled | HIGH | X | X | Fix |
| Redundant SIMD | LOW | - | X | Remove |
| Missing strategies | MEDIUM | X | X | Add |
| Benchmark overhead | LOW | X | X | Fix |
| Missing GFLOP/s | MEDIUM | X | X | Add |