module TwoMM

using LinearAlgebra
using Base.Threads

export init_2mm!, kernel_2mm_seq!, kernel_2mm_threads_static!, kernel_2mm_threads_dynamic!
export kernel_2mm_tiled!, kernel_2mm_blas!, kernel_2mm_tasks!
export STRATEGIES_2MM

const STRATEGIES_2MM = [
    "sequential",
    "threads_static",
    "threads_dynamic", 
    "tiled",
    "blas",
    "tasks"
]

# Initialize arrays following PolyBench pattern
# Column-major friendly initialization
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
    
    # Column-major order: iterate columns first (outer), rows second (inner)
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = ((i - 1) * (j - 1) + 1) % ni / ni
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = (i - 1) * j % nj / nj
    end
    
    @inbounds for j in 1:nl, i in 1:nj
        C[i, j] = ((i - 1) * (j + 2) + 1) % nl / nl
    end
    
    @inbounds for j in 1:nl, i in 1:ni
        D[i, j] = (i - 1) * (j + 1) % nk / nk
    end
    
    fill!(tmp, 0.0)
    
    return nothing
end

# Reset arrays for next iteration (in-place)
function reset_2mm!(tmp::Matrix{Float64}, D::Matrix{Float64}, D_orig::Matrix{Float64})
    fill!(tmp, 0.0)
    copyto!(D, D_orig)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Column-major optimized loop order
 - Proper cache access pattern for Julia arrays
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
    # Column-major: for each column j of tmp, compute as sum over k
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = alpha * B[k, j]
            for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @inbounds for j in 1:nl
        # Scale D column
        for i in 1:ni
            D[i, j] *= beta
        end
        # Add tmp * C contribution
        for k in 1:nj
            c_kj = C[k, j]
            for i in 1:ni
                D[i, j] += tmp[i, k] * c_kj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded with static scheduling
 - Parallelizes outer loop over columns
 - Zero allocations in hot path
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
    
    # tmp = alpha * A * B (parallelize over columns of tmp)
    @threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C (parallelize over columns of D)
    @threads :static for j in 1:nl
        @inbounds begin
            for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Threaded with dynamic scheduling
 - Better load balancing for irregular workloads
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
            for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    # D = beta * D + tmp * C
    @threads :dynamic for j in 1:nl
        @inbounds begin
            for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Tiled/Blocked for cache optimization
 - Block size tuned for L2 cache
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
    
    # tmp = alpha * A * B (tiled)
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
    
    # D = beta * D + tmp * C (tiled)
    @threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        
        @inbounds for j in jj:j_end
            for i in 1:ni
                D[i, j] *= beta
            end
        end
        
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
 - Uses optimized BLAS routines
=============================================================================#
function kernel_2mm_blas!(
    alpha::Float64, beta::Float64,
    A::Matrix{Float64}, B::Matrix{Float64},
    tmp::Matrix{Float64}, C::Matrix{Float64},
    D::Matrix{Float64}
)
    # tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)
    
    # D = beta * D + tmp * C  
    mul!(D, tmp, C, 1.0, beta)
    
    return nothing
end

#=============================================================================
 Strategy 6: Task-based parallelism
 - Coarse-grained tasks over column blocks
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
                        for i in 1:ni
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
                    for i in 1:ni
                        D[i, j] *= beta
                    end
                    for k in 1:nj
                        c_kj = C[k, j]
                        for i in 1:ni
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

# Get kernel function by strategy name
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

end # module
