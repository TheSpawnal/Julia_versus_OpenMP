

module TwoMM
#=
PolyBench 2MM Kernel - Julia Implementation
Computation: D = alpha * A * B * C + beta * D

Design Principles:
1. Zero-allocation hot paths (verified with @allocated)
2. Column-major optimized loop order
3. Consistent Float64 for fair OpenMP comparison
4. @simd @inbounds on all inner loops

Strategies:
1. sequential     - Baseline with SIMD
2. threads_static - Static scheduling over columns
3. threads_dynamic- Dynamic scheduling for load balance
4. tiled          - Cache-blocked with parallel outer loop
5. blas           - BLAS mul! (reference upper bound)
6. tasks          - Coarse-grained task parallelism
=#

using LinearAlgebra
using Base.Threads

export init_2mm!, reset_2mm!
export kernel_2mm_seq!, kernel_2mm_threads_static!, kernel_2mm_threads_dynamic!
export kernel_2mm_tiled!, kernel_2mm_blas!, kernel_2mm_tasks!
export STRATEGIES_2MM, get_kernel

const STRATEGIES_2MM = [
    "sequential",
    "threads_static",
    "threads_dynamic",
    "tiled",
    "blas",
    "tasks"
]

const DATASETS_2MM = Dict(
    "MINI" => (ni=16, nj=18, nk=22, nl=24),
    "SMALL" => (ni=40, nj=50, nk=70, nl=80),
    "MEDIUM" => (ni=180, nj=190, nk=210, nl=220),
    "LARGE" => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

#=============================================================================
 Initialization - PolyBench compatible
=============================================================================#
function init_2mm!(
    alpha::Ref{Float64}, beta::Ref{Float64},
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    nj = size(B, 2)
    nl = size(C, 2)
    
    alpha[] = 1.5
    beta[] = 1.2
    
    # Column-major initialization
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = ((i - 1) * (j - 1) + 1) % ni / Float64(ni)
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = (i - 1) * j % nj / Float64(nj)
    end
    
    @inbounds for j in 1:nl, i in 1:nj
        C[i, j] = ((i - 1) * (j + 2) + 1) % nl / Float64(nl)
    end
    
    @inbounds for j in 1:nl, i in 1:ni
        D[i, j] = (i - 1) * (j + 1) % nk / Float64(nk)
    end
    
    fill!(tmp, 0.0)
    
    return nothing
end

# Reset for re-benchmarking (in-place)
function reset_2mm!(tmp::Matrix{Float64}, D::Matrix{Float64}, D_orig::Matrix{Float64})
    fill!(tmp, 0.0)
    copyto!(D, D_orig)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline with SIMD
 - Column-major loop order (j outer, i innermost)
 - @simd on innermost loop for vectorization
 - @inbounds for bounds-check elimination
=============================================================================#
function kernel_2mm_seq!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = alpha * B[k, j]  # Hoist scalar outside SIMD loop
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @inbounds for j in 1:nl
        # Scale D column by beta
        @simd for i in 1:ni
            D[i, j] *= beta
        end
        # Accumulate tmp * C
        for k in 1:nj
            c_kj = C[k, j]  # Hoist scalar
            @simd for i in 1:ni
                D[i, j] += tmp[i, k] * c_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded with static scheduling
 - Parallelizes over columns (j loop)
 - Zero allocations in hot path
 - Best for uniform workload
=============================================================================#
function kernel_2mm_threads_static!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @threads :static for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Threaded with dynamic scheduling
 - Work-stealing for load balance
 - Better for irregular workloads
=============================================================================#
function kernel_2mm_threads_dynamic!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # tmp = alpha * A * B
    @threads :dynamic for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @threads :dynamic for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Tiled/Blocked with parallel outer loop
 - Cache optimization through blocking
 - Tile size tuned for L2 cache (~256KB)
 - Outer tile loop parallelized
=============================================================================#
function kernel_2mm_tiled!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64};
    tile_size::Int=64
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    ts = tile_size
    
    # tmp = alpha * A * B (tiled, parallelized over column tiles)
    @threads :static for jj in 1:ts:nj
        j_end = min(jj + ts - 1, nj)
        for kk in 1:ts:nk
            k_end = min(kk + ts - 1, nk)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                
                @inbounds for j in jj:j_end
                    for k in kk:k_end
                        b_kj = alpha * B[k, j]
                        @simd for i in ii:i_end
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # D = beta * D + tmp * C (tiled, parallelized)
    @threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        
        # Scale D columns in this tile
        @inbounds for j in jj:j_end
            @simd for i in 1:ni
                D[i, j] *= beta
            end
        end
        
        # Accumulate tmp * C for this tile
        for kk in 1:ts:nj
            k_end = min(kk + ts - 1, nj)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                
                @inbounds for j in jj:j_end
                    for k in kk:k_end
                        c_kj = C[k, j]
                        @simd for i in ii:i_end
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: BLAS-accelerated
 - Uses optimized BLAS mul! routines
 - Reference upper bound for performance
 - Note: BLAS threading is separate from Julia threading
=============================================================================#
function kernel_2mm_blas!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    # tmp = alpha * A * B + 0 * tmp
    mul!(tmp, A, B, alpha, 0.0)
    
    # D = 1.0 * tmp * C + beta * D
    mul!(D, tmp, C, 1.0, beta)
    
    return nothing
end

#=============================================================================
 Strategy 6: Task-based parallelism
 - Coarse-grained tasks over column blocks
 - Explicit task spawning for control
 - Good for heterogeneous systems
=============================================================================#
function kernel_2mm_tasks!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64};
    num_tasks::Int=nthreads()
)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    # First multiplication: tmp = alpha * A * B
    chunk_j = max(1, cld(nj, num_tasks))
    
    @sync begin
        for task_id in 1:num_tasks
            j_start = (task_id - 1) * chunk_j + 1
            j_end = min(task_id * chunk_j, nj)
            
            j_start > nj && continue
            
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nk
                        b_kj = alpha * B[k, j]
                        @simd for i in 1:ni
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    # Second multiplication: D = beta * D + tmp * C
    chunk_l = max(1, cld(nl, num_tasks))
    
    @sync begin
        for task_id in 1:num_tasks
            j_start = (task_id - 1) * chunk_l + 1
            j_end = min(task_id * chunk_l, nl)
            
            j_start > nl && continue
            
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    @simd for i in 1:ni
                        D[i, j] *= beta
                    end
                    for k in 1:nj
                        c_kj = C[k, j]
                        @simd for i in 1:ni
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Kernel Dispatcher
=============================================================================#
function get_kernel(strategy::String)
    kernels = Dict(
        "sequential" => kernel_2mm_seq!,
        "threads_static" => kernel_2mm_threads_static!,
        "threads_dynamic" => kernel_2mm_threads_dynamic!,
        "tiled" => kernel_2mm_tiled!,
        "blas" => kernel_2mm_blas!,
        "tasks" => kernel_2mm_tasks!
    )
    return get(kernels, strategy, kernel_2mm_seq!)
end

#=============================================================================
 FLOPs Calculation
=============================================================================#
function flops_2mm(ni, nj, nk, nl)
    # tmp = alpha * A * B: 2*ni*nj*nk (mul + add per element)
    # D = beta*D + tmp*C: 2*ni*nl*nj + 2*ni*nl (scale + add)
    return 2*ni*nj*nk + 2*ni*nl*nj + 2*ni*nl
end

#=============================================================================
 Memory Calculation (bytes)
=============================================================================#
function memory_2mm(ni, nj, nk, nl)
    # A: ni*nk, B: nk*nj, tmp: ni*nj, C: nj*nl, D: ni*nl
    return (ni*nk + nk*nj + ni*nj + nj*nl + ni*nl) * sizeof(Float64)
end

end # module TwoMM