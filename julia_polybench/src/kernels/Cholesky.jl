module Cholesky

using LinearAlgebra
using Base.Threads
using Printf

export STRATEGIES_CHOLESKY, DATASETS_CHOLESKY
export init_cholesky!, verify_cholesky, get_kernel
export kernel_cholesky_seq!, kernel_cholesky_simd!
export kernel_cholesky_threads!, kernel_cholesky_blas!, kernel_cholesky_tiled!

const STRATEGIES_CHOLESKY = [
    "sequential",
    "simd",
    "threads",
    "blas",
    "tiled"
]

const DATASETS_CHOLESKY = Dict(
    "MINI" => (n=40,),
    "SMALL" => (n=120,),
    "MEDIUM" => (n=400,),
    "LARGE" => (n=2000,),
    "EXTRALARGE" => (n=4000,)
)

# Initialize matrix as positive-definite following PolyBench specification
function init_cholesky!(A::Matrix{Float64})
    n = size(A, 1)
    
    # Initialize lower triangular part (column-major)
    @inbounds for j in 1:n
        for i in j:n
            if i == j
                A[i, j] = 1.0
            elseif i > j
                A[i, j] = Float64((-j % n) / n + 1)
            else
                A[i, j] = 0.0
            end
        end
    end
    
    # Make positive semi-definite: A = B * B^T
    B = copy(A)
    
    # Column-major matrix multiplication
    @inbounds for j in 1:n
        for i in 1:n
            sum_val = 0.0
            for k in 1:n
                sum_val += B[i, k] * B[j, k]
            end
            A[i, j] = sum_val
        end
    end
    
    return nothing
end

# Verify Cholesky decomposition: A = L * L^T
function verify_cholesky(A_orig::Matrix{Float64}, L::Matrix{Float64}; tol::Float64=1e-6)
    n = size(A_orig, 1)
    max_err = 0.0
    
    @inbounds for j in 1:n
        for i in j:n
            sum_val = 0.0
            for k in 1:min(i, j)
                sum_val += L[i, k] * L[j, k]
            end
            err = abs(A_orig[i, j] - sum_val)
            max_err = max(max_err, err)
        end
    end
    
    return max_err < tol, max_err
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Cholesky-Banachiewicz algorithm (row-by-row)
=============================================================================#
function kernel_cholesky_seq!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for i in 1:n
        # Off-diagonal elements
        for j in 1:(i-1)
            sum_val = 0.0
            for k in 1:(j-1)
                sum_val += A[i, k] * A[j, k]
            end
            A[i, j] -= sum_val
            A[i, j] /= A[j, j]
        end
        
        # Diagonal element
        sum_val = 0.0
        for k in 1:(i-1)
            sum_val += A[i, k] * A[i, k]
        end
        A[i, i] -= sum_val
        A[i, i] = sqrt(A[i, i])
    end
    
    # Zero out upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: SIMD optimized
 - Vectorized dot products
=============================================================================#
function kernel_cholesky_simd!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for i in 1:n
        # Off-diagonal elements with SIMD
        for j in 1:(i-1)
            sum_val = 0.0
            @simd for k in 1:(j-1)
                sum_val += A[i, k] * A[j, k]
            end
            A[i, j] -= sum_val
            A[i, j] /= A[j, j]
        end
        
        # Diagonal with SIMD
        sum_val = 0.0
        @simd for k in 1:(i-1)
            sum_val += A[i, k] * A[i, k]
        end
        A[i, i] -= sum_val
        A[i, i] = sqrt(A[i, i])
    end
    
    # Zero out upper triangular
    @inbounds for j in 2:n
        @simd for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Threaded (parallel trailing update)
 - Zero-allocation design with pre-allocated thread-local storage
=============================================================================#
function kernel_cholesky_threads!(A::Matrix{Float64})
    n = size(A, 1)
    nt = nthreads()
    
    # Pre-allocate thread-local accumulators ONCE
    thread_sums = zeros(Float64, nt)
    
    @inbounds for i in 1:n
        # Off-diagonal: parallel only for large enough inner loops
        for j in 1:(i-1)
            if j > 64 && nt > 1
                # Parallel reduction
                fill!(thread_sums, 0.0)
                chunk = cld(j-1, nt)
                
                @threads :static for tid in 1:nt
                    k_start = (tid - 1) * chunk + 1
                    k_end = min(tid * chunk, j - 1)
                    local_sum = 0.0
                    for k in k_start:k_end
                        local_sum += A[i, k] * A[j, k]
                    end
                    thread_sums[tid] = local_sum
                end
                
                total = 0.0
                for t in 1:nt
                    total += thread_sums[t]
                end
                A[i, j] -= total
            else
                sum_val = 0.0
                for k in 1:(j-1)
                    sum_val += A[i, k] * A[j, k]
                end
                A[i, j] -= sum_val
            end
            A[i, j] /= A[j, j]
        end
        
        # Diagonal element
        if i > 64 && nt > 1
            fill!(thread_sums, 0.0)
            chunk = cld(i-1, nt)
            
            @threads :static for tid in 1:nt
                k_start = (tid - 1) * chunk + 1
                k_end = min(tid * chunk, i - 1)
                local_sum = 0.0
                for k in k_start:k_end
                    local_sum += A[i, k] * A[i, k]
                end
                thread_sums[tid] = local_sum
            end
            
            total = 0.0
            for t in 1:nt
                total += thread_sums[t]
            end
            A[i, i] -= total
        else
            sum_val = 0.0
            for k in 1:(i-1)
                sum_val += A[i, k] * A[i, k]
            end
            A[i, i] -= sum_val
        end
        A[i, i] = sqrt(A[i, i])
    end
    
    # Zero out upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: BLAS-accelerated
 - Uses Julia's optimized LAPACK cholesky
=============================================================================#
function kernel_cholesky_blas!(A::Matrix{Float64})
    n = size(A, 1)
    
    try
        # Use LAPACK cholesky (lower triangular)
        F = cholesky!(Hermitian(A, :L))
        
        # Zero upper triangular
        @inbounds for j in 2:n
            for i in 1:(j-1)
                A[i, j] = 0.0
            end
        end
    catch e
        # Fallback to sequential if LAPACK fails
        @warn "BLAS Cholesky failed, using sequential: $e"
        kernel_cholesky_seq!(A)
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: Tiled/Blocked (cache-optimized)
 - Process in blocks for better cache utilization
=============================================================================#
function kernel_cholesky_tiled!(A::Matrix{Float64}; tile_size::Int=64)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for kk in 1:ts:n
        k_end = min(kk + ts - 1, n)
        
        # Factor diagonal block
        for i in kk:k_end
            for j in kk:(i-1)
                sum_val = 0.0
                for k in kk:(j-1)
                    sum_val += A[i, k] * A[j, k]
                end
                A[i, j] -= sum_val
                A[i, j] /= A[j, j]
            end
            
            sum_val = 0.0
            for k in kk:(i-1)
                sum_val += A[i, k] * A[i, k]
            end
            A[i, i] -= sum_val
            A[i, i] = sqrt(A[i, i])
        end
        
        # Update trailing blocks
        if k_end < n
            # Column panel update
            for ii in (k_end+1):ts:n
                i_end = min(ii + ts - 1, n)
                
                for i in ii:i_end
                    for j in kk:k_end
                        sum_val = 0.0
                        for k in kk:(j-1)
                            sum_val += A[i, k] * A[j, k]
                        end
                        A[i, j] -= sum_val
                        A[i, j] /= A[j, j]
                    end
                end
            end
        end
    end
    
    # Zero upper triangular
    @inbounds for j in 2:n
        for i in 1:(j-1)
            A[i, j] = 0.0
        end
    end
    
    return nothing
end

# Get kernel by strategy name
function get_kernel(strategy::AbstractString)
    kernels = Dict(
        "sequential" => kernel_cholesky_seq!,
        "simd" => kernel_cholesky_simd!,
        "threads" => kernel_cholesky_threads!,
        "blas" => kernel_cholesky_blas!,
        "tiled" => kernel_cholesky_tiled!
    )
    return get(kernels, String(strategy), kernel_cholesky_seq!)
end

end # module