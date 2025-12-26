module ThreeMM
#=
PolyBench 3MM Kernel - Julia Implementation
Computation: G = (A * B) * (C * D)
    E = A * B
    F = C * D  
    G = E * F

Design Principles:
1. Zero-allocation hot paths (verified with @allocated)
2. Column-major optimized loop order (j outer, i innermost)
3. Consistent Float64 for fair OpenMP comparison
4. @simd @inbounds on all inner loops
5. Scalar hoisting outside SIMD loops

Strategies:
1. sequential      - Baseline with SIMD
2. threads_static  - Static scheduling over columns
3. threads_dynamic - Dynamic scheduling for load balance
4. tiled           - Cache-blocked with parallel outer loop
5. blas            - BLAS mul! (reference upper bound)
6. tasks           - Coarse-grained task parallelism

Author: SpawnAl / Falkor collaboration
=#

using LinearAlgebra
using Base.Threads

export init_3mm!, reset_3mm!
export kernel_3mm_seq!, kernel_3mm_threads_static!, kernel_3mm_threads_dynamic!
export kernel_3mm_tiled!, kernel_3mm_blas!, kernel_3mm_tasks!
export STRATEGIES_3MM, DATASETS_3MM, get_kernel
export flops_3mm, memory_3mm

# Strategy list for iteration
const STRATEGIES_3MM = [
    "sequential",
    "threads_static",
    "threads_dynamic",
    "tiled",
    "blas",
    "tasks"
]

# PolyBench standard dataset sizes
const DATASETS_3MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl, :nm), NTuple{5, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=20,   nl=22,   nm=24),
    "SMALL"      => (ni=40,   nj=50,   nk=60,   nl=70,   nm=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=200,  nl=210,  nm=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

#=============================================================================
 Initialization - PolyBench compatible
 Matrix dimensions:
   A: ni x nk
   B: nk x nj
   C: nj x nm
   D: nm x nl
   E: ni x nj (intermediate)
   F: nj x nl (intermediate)
   G: ni x nl (result)
=============================================================================#
function init_3mm!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}
)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # Column-major initialization (j outer, i inner for cache efficiency)
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = ((i - 1) * (j - 1) + 1) % ni / Float64(5 * ni)
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = ((i - 1) * j + 2) % nj / Float64(5 * nj)
    end
    
    @inbounds for j in 1:nm, i in 1:nj
        C[i, j] = (i - 1) * (j + 2) % nl / Float64(5 * nl)
    end
    
    @inbounds for j in 1:nl, i in 1:nm
        D[i, j] = ((i - 1) * (j + 1) + 2) % nk / Float64(5 * nk)
    end
    
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    
    return nothing
end

# Reset for re-benchmarking (call before each timed iteration)
function reset_3mm!(E::Matrix{Float64}, F::Matrix{Float64}, G::Matrix{Float64})
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline with SIMD
 - Column-major loop order (j outer, i innermost)
 - @simd on innermost loop for vectorization
 - @inbounds for bounds-check elimination
 - Scalar hoisting to avoid repeated memory access in SIMD loop
=============================================================================#
function kernel_3mm_seq!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = B[k, j]
            @simd for i in 1:ni
                E[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # F = C * D
    @inbounds for j in 1:nl
        for k in 1:nm
            d_kj = D[k, j]
            @simd for i in 1:nj
                F[i, j] += C[i, k] * d_kj
            end
        end
    end
    
    # G = E * F
    @inbounds for j in 1:nl
        for k in 1:nj
            f_kj = F[k, j]
            @simd for i in 1:ni
                G[i, j] += E[i, k] * f_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded with static scheduling
 - Parallelizes over columns (j loop)
 - Zero allocations in hot path (verified)
 - Best for uniform workload
=============================================================================#
function kernel_3mm_threads_static!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    @threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = B[k, j]
            @simd for i in 1:ni
                E[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # F = C * D
    @threads :static for j in 1:nl
        @inbounds for k in 1:nm
            d_kj = D[k, j]
            @simd for i in 1:nj
                F[i, j] += C[i, k] * d_kj
            end
        end
    end
    
    # G = E * F
    @threads :static for j in 1:nl
        @inbounds for k in 1:nj
            f_kj = F[k, j]
            @simd for i in 1:ni
                G[i, j] += E[i, k] * f_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Threaded with dynamic scheduling
 - Work-stealing for load balance
 - Better for irregular workloads or NUMA systems
=============================================================================#
function kernel_3mm_threads_dynamic!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B
    @threads :dynamic for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = B[k, j]
            @simd for i in 1:ni
                E[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # F = C * D
    @threads :dynamic for j in 1:nl
        @inbounds for k in 1:nm
            d_kj = D[k, j]
            @simd for i in 1:nj
                F[i, j] += C[i, k] * d_kj
            end
        end
    end
    
    # G = E * F
    @threads :dynamic for j in 1:nl
        @inbounds for k in 1:nj
            f_kj = F[k, j]
            @simd for i in 1:ni
                G[i, j] += E[i, k] * f_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Tiled/Blocked with parallel outer loop
 - Cache optimization through blocking
 - Tile size tuned for L2 cache (~256KB per core typical)
 - Outer tile loop parallelized
=============================================================================#
function kernel_3mm_tiled!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64};
    tile_size::Int=64
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    ts = tile_size
    
    # E = A * B (tiled with parallel outer loop)
    @threads :static for jj in 1:ts:nj
        j_end = min(jj + ts - 1, nj)
        @inbounds for kk in 1:ts:nk
            k_end = min(kk + ts - 1, nk)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        b_kj = B[k, j]
                        @simd for i in ii:i_end
                            E[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # F = C * D (tiled with parallel outer loop)
    @threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        @inbounds for kk in 1:ts:nm
            k_end = min(kk + ts - 1, nm)
            for ii in 1:ts:nj
                i_end = min(ii + ts - 1, nj)
                for j in jj:j_end
                    for k in kk:k_end
                        d_kj = D[k, j]
                        @simd for i in ii:i_end
                            F[i, j] += C[i, k] * d_kj
                        end
                    end
                end
            end
        end
    end
    
    # G = E * F (tiled with parallel outer loop)
    @threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        @inbounds for kk in 1:ts:nj
            k_end = min(kk + ts - 1, nj)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        f_kj = F[k, j]
                        @simd for i in ii:i_end
                            G[i, j] += E[i, k] * f_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: BLAS (reference upper bound)
 - Uses optimized BLAS mul! routines
 - Reference upper bound for performance
 - Note: BLAS threading configured externally
=============================================================================#
function kernel_3mm_blas!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}
)
    # E = A * B
    mul!(E, A, B)
    
    # F = C * D
    mul!(F, C, D)
    
    # G = E * F
    mul!(G, E, F)
    
    return nothing
end

#=============================================================================
 Strategy 6: Task-based parallelism
 - Coarse-grained tasks over column blocks
 - Explicit task spawning for control
 - E and F can be computed in parallel before G
=============================================================================#
function kernel_3mm_tasks!(
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64};
    num_tasks::Int=nthreads()
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nm = size(C)
    _, nl = size(D)
    
    # E = A * B (parallel tasks)
    chunk_j = max(1, cld(nj, num_tasks))
    @sync begin
        for task_id in 1:num_tasks
            j_start = (task_id - 1) * chunk_j + 1
            j_end = min(task_id * chunk_j, nj)
            j_start > nj && continue
            
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nk
                        b_kj = B[k, j]
                        @simd for i in 1:ni
                            E[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # F = C * D (parallel tasks)
    chunk_l = max(1, cld(nl, num_tasks))
    @sync begin
        for task_id in 1:num_tasks
            j_start = (task_id - 1) * chunk_l + 1
            j_end = min(task_id * chunk_l, nl)
            j_start > nl && continue
            
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nm
                        d_kj = D[k, j]
                        @simd for i in 1:nj
                            F[i, j] += C[i, k] * d_kj
                        end
                    end
                end
            end
        end
    end
    
    # G = E * F (parallel tasks, after E and F complete)
    @sync begin
        for task_id in 1:num_tasks
            j_start = (task_id - 1) * chunk_l + 1
            j_end = min(task_id * chunk_l, nl)
            j_start > nl && continue
            
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nj
                        f_kj = F[k, j]
                        @simd for i in 1:ni
                            G[i, j] += E[i, k] * f_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Kernel Dispatcher - FIXED: accepts AbstractString to handle SubString
=============================================================================#
function get_kernel(strategy::AbstractString)
    s = lowercase(String(strategy))  # Normalize
    
    if s == "sequential" || s == "seq"
        return kernel_3mm_seq!
    elseif s == "threads_static" || s == "threads" || s == "static"
        return kernel_3mm_threads_static!
    elseif s == "threads_dynamic" || s == "dynamic"
        return kernel_3mm_threads_dynamic!
    elseif s == "tiled" || s == "blocked"
        return kernel_3mm_tiled!
    elseif s == "blas"
        return kernel_3mm_blas!
    elseif s == "tasks"
        return kernel_3mm_tasks!
    else
        error("Unknown strategy: $strategy. Available: $(join(STRATEGIES_3MM, ", "))")
    end
end

#=============================================================================
 FLOPs Calculation
 E = A*B: 2*ni*nj*nk
 F = C*D: 2*nj*nl*nm
 G = E*F: 2*ni*nl*nj
=============================================================================#
function flops_3mm(ni, nj, nk, nl, nm)
    return 2*ni*nj*nk + 2*nj*nl*nm + 2*ni*nl*nj
end

#=============================================================================
 Memory Calculation (bytes)
 A: ni*nk, B: nk*nj, C: nj*nm, D: nm*nl
 E: ni*nj, F: nj*nl, G: ni*nl
=============================================================================#
function memory_3mm(ni, nj, nk, nl, nm)
    return (ni*nk + nk*nj + nj*nm + nm*nl + ni*nj + nj*nl + ni*nl) * sizeof(Float64)
end

end # module ThreeMM
