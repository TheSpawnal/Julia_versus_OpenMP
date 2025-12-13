module Cholesky

using LinearAlgebra
using Base.Threads
using Printf

export DATASET_SIZES, flops_cholesky
export init_array!, verify_result
export kernel_cholesky_seq!, kernel_cholesky_simd!
export kernel_cholesky_threads!, kernel_cholesky_blas!, kernel_cholesky_tiled!
export get_kernel

# Dataset sizes following PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => 40,
    "SMALL" => 120,
    "MEDIUM" => 400,
    "LARGE" => 2000,
    "EXTRALARGE" => 4000
)

# FLOPs for Cholesky: n^3/3
function flops_cholesky(n::Int)
    return Float64(n)^3 / 3
end

# Initialize symmetric positive-definite matrix
# Uses A = B * B^T construction for guaranteed SPD
function init_array!(A::Matrix{Float64})
    n = size(A, 1)
    
    # Build lower triangular + diagonal
    @inbounds for j in 1:n
        for i in j:n
            if i == j
                A[i, j] = 1.0
            else
                A[i, j] = (-(j - 1) % n) / n + 1.0
            end
        end
    end
    
    # Make symmetric
    @inbounds for j in 1:n
        for i in 1:(j-1)
            A[i, j] = A[j, i]
        end
    end
    
    # Make positive definite: A = B * B^T
    B = copy(A)
    fill!(A, 0.0)
    
    # Column-major friendly A = B * B^T
    @inbounds for j in 1:n
        for k in 1:n
            b_jk = B[j, k]
            for i in j:n
                A[i, j] += B[i, k] * b_jk
            end
        end
    end
    
    # Symmetrize (copy lower to upper)
    @inbounds for j in 1:n
        for i in 1:(j-1)
            A[i, j] = A[j, i]
        end
    end
    
    return nothing
end

# Verify Cholesky result: ||A - L*L^T|| / ||A||
function verify_result(A_orig::Matrix{Float64}, L::Matrix{Float64})
    n = size(A_orig, 1)
    
    # Compute L * L^T
    LLT = zeros(n, n)
    @inbounds for j in 1:n
        for k in 1:j
            l_jk = L[j, k]
            for i in j:n
                LLT[i, j] += L[i, k] * l_jk
            end
        end
    end
    
    # Symmetrize
    @inbounds for j in 1:n
        for i in 1:(j-1)
            LLT[i, j] = LLT[j, i]
        end
    end
    
    # Relative error
    diff_norm = norm(A_orig - LLT)
    orig_norm = norm(A_orig)
    
    return diff_norm / max(orig_norm, 1e-10)
end

#=============================================================================
 Strategy 1: Sequential baseline (Cholesky-Banachiewicz)
 - Standard lower-triangular Cholesky
 - Column-major optimized
=============================================================================#
function kernel_cholesky_seq!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2))
        sum_sq = 0.0
        for k in 1:(j-1)
            sum_sq += A[j, k] * A[j, k]
        end
        A[j, j] = sqrt(A[j, j] - sum_sq)
        
        # Column elements below diagonal
        for i in (j+1):n
            sum_prod = 0.0
            for k in 1:(j-1)
                sum_prod += A[i, k] * A[j, k]
            end
            A[i, j] = (A[i, j] - sum_prod) / A[j, j]
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

#=============================================================================
 Strategy 2: SIMD-optimized dot products
 - Uses @simd for inner loops
=============================================================================#
function kernel_cholesky_simd!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element with SIMD reduction
        sum_sq = 0.0
        @simd for k in 1:(j-1)
            sum_sq += A[j, k] * A[j, k]
        end
        A[j, j] = sqrt(A[j, j] - sum_sq)
        
        # Column elements below diagonal
        diag_inv = 1.0 / A[j, j]
        for i in (j+1):n
            sum_prod = 0.0
            @simd for k in 1:(j-1)
                sum_prod += A[i, k] * A[j, k]
            end
            A[i, j] = (A[i, j] - sum_prod) * diag_inv
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

#=============================================================================
 Strategy 3: Right-looking blocked (PROPER parallelization)
 - Parallelizes the TRAILING MATRIX UPDATE (the only parallelizable part)
 - Respects Cholesky data dependencies
=============================================================================#
function kernel_cholesky_threads!(A::Matrix{Float64}; tile_size::Int=64)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for kk in 1:ts:n
        k_end = min(kk + ts - 1, n)
        
        # 1. Factor diagonal block (sequential - must be serial due to dependencies)
        for j in kk:k_end
            sum_sq = 0.0
            for k in 1:(j-1)
                sum_sq += A[j, k] * A[j, k]
            end
            A[j, j] = sqrt(A[j, j] - sum_sq)
            
            for i in (j+1):k_end
                sum_prod = 0.0
                for k in 1:(j-1)
                    sum_prod += A[i, k] * A[j, k]
                end
                A[i, j] = (A[i, j] - sum_prod) / A[j, j]
            end
        end
        
        # 2. Solve panel below diagonal block: A21 = A21 / L11^T
        if k_end < n
            for j in kk:k_end
                diag_inv = 1.0 / A[j, j]
                for i in (k_end+1):n
                    sum_prod = 0.0
                    for k in kk:(j-1)
                        sum_prod += A[i, k] * A[j, k]
                    end
                    A[i, j] = (A[i, j] - sum_prod) * diag_inv
                end
            end
        end
        
        # 3. UPDATE TRAILING MATRIX - THIS IS THE PARALLEL PART!
        # A22 = A22 - L21 * L21^T
        # Each column of the trailing matrix can be updated independently
        if k_end < n
            @threads :static for j in (k_end+1):n
                # Update column j of trailing matrix
                for i in j:n
                    sum_update = 0.0
                    @simd for k in kk:k_end
                        sum_update += A[i, k] * A[j, k]
                    end
                    A[i, j] -= sum_update
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

#=============================================================================
 Strategy 4: BLAS-accelerated (LAPACK)
 - Uses optimized library implementation
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
        @warn "BLAS Cholesky failed, using sequential fallback"
        kernel_cholesky_seq!(A)
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: Tiled/Blocked (cache-optimized, single-threaded)
 - Process in blocks for better cache utilization
 - No threading - pure cache optimization
=============================================================================#
function kernel_cholesky_tiled!(A::Matrix{Float64}; tile_size::Int=64)
    n = size(A, 1)
    ts = tile_size
    
    @inbounds for kk in 1:ts:n
        k_end = min(kk + ts - 1, n)
        
        # Factor diagonal block
        for j in kk:k_end
            # Compute diagonal element
            sum_sq = 0.0
            for k in 1:(j-1)
                sum_sq += A[j, k] * A[j, k]
            end
            A[j, j] = sqrt(A[j, j] - sum_sq)
            
            # Compute column below diagonal within block
            diag_inv = 1.0 / A[j, j]
            for i in (j+1):k_end
                sum_prod = 0.0
                for k in 1:(j-1)
                    sum_prod += A[i, k] * A[j, k]
                end
                A[i, j] = (A[i, j] - sum_prod) * diag_inv
            end
        end
        
        # Update panel below diagonal block
        if k_end < n
            for j in kk:k_end
                diag_inv = 1.0 / A[j, j]
                for i in (k_end+1):n
                    sum_prod = 0.0
                    for k in kk:(j-1)
                        sum_prod += A[i, k] * A[j, k]
                    end
                    A[i, j] = (A[i, j] - sum_prod) * diag_inv
                end
            end
            
            # Update trailing matrix A22 = A22 - L21 * L21^T
            # Process in tiles for cache efficiency
            for jj in (k_end+1):ts:n
                j_end = min(jj + ts - 1, n)
                for ii in jj:ts:n
                    i_end = min(ii + ts - 1, n)
                    
                    # Update tile [ii:i_end, jj:j_end]
                    for j in jj:j_end
                        for i in max(ii, j):i_end
                            sum_update = 0.0
                            @simd for k in kk:k_end
                                sum_update += A[i, k] * A[j, k]
                            end
                            A[i, j] -= sum_update
                        end
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

# Kernel selector
function get_kernel(name::AbstractString)
    name_lower = lowercase(String(name))
    
    kernels = Dict(
        "sequential" => kernel_cholesky_seq!,
        "seq" => kernel_cholesky_seq!,
        "simd" => kernel_cholesky_simd!,
        "threads" => kernel_cholesky_threads!,
        "blas" => kernel_cholesky_blas!,
        "tiled" => kernel_cholesky_tiled!,
    )
    
    if haskey(kernels, name_lower)
        return kernels[name_lower]
    else
        available = join(keys(kernels), ", ")
        error("Unknown kernel: $name. Available: $available")
    end
end

end # module