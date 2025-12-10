module Cholesky

using LinearAlgebra
using Base.Threads

export init_cholesky!, kernel_cholesky_seq!, kernel_cholesky_threads!
export kernel_cholesky_blocked!, kernel_cholesky_blas!, kernel_cholesky_rightlooking!
export kernel_cholesky_leftlooking!, kernel_cholesky_tasks!
export STRATEGIES_CHOLESKY

const STRATEGIES_CHOLESKY = [
    "sequential",
    "threads",
    "blocked",
    "blas",
    "rightlooking",
    "leftlooking",
    "tasks"
]

# Initialize symmetric positive-definite matrix for Cholesky
# Following PolyBench pattern: A = B * B^T for guaranteed SPD
function init_cholesky!(A::Matrix{Float64})
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

# Reset matrix for next iteration
function reset_cholesky!(A::Matrix{Float64}, A_orig::Matrix{Float64})
    copyto!(A, A_orig)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline (ijk variant)
 - Standard Cholesky with column-major optimization
=============================================================================#
function kernel_cholesky_seq!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element
        for k in 1:(j-1)
            A[j, j] -= A[j, k] * A[j, k]
        end
        A[j, j] = sqrt(A[j, j])
        
        # Column elements below diagonal
        for i in (j+1):n
            for k in 1:(j-1)
                A[i, j] -= A[i, k] * A[j, k]
            end
            A[i, j] /= A[j, j]
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded trailing update
 - Parallelizes the update of columns below current panel
=============================================================================#
function kernel_cholesky_threads!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element (sequential)
        for k in 1:(j-1)
            A[j, j] -= A[j, k] * A[j, k]
        end
        A[j, j] = sqrt(A[j, j])
        
        ajj = A[j, j]
        
        # Parallelize update of column j below diagonal
        if n - j >= nthreads()
            @threads :static for i in (j+1):n
                @inbounds begin
                    for k in 1:(j-1)
                        A[i, j] -= A[i, k] * A[j, k]
                    end
                    A[i, j] /= ajj
                end
            end
        else
            # Sequential for small remaining work
            for i in (j+1):n
                for k in 1:(j-1)
                    A[i, j] -= A[i, k] * A[j, k]
                end
                A[i, j] /= ajj
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Blocked/Tiled Cholesky
 - Panel factorization followed by trailing update
=============================================================================#
function kernel_cholesky_blocked!(A::Matrix{Float64}; block_size::Int=64)
    n = size(A, 1)
    bs = min(block_size, n)
    
    @inbounds for jj in 1:bs:n
        j_end = min(jj + bs - 1, n)
        
        # Factor diagonal block
        for j in jj:j_end
            for k in 1:(j-1)
                A[j, j] -= A[j, k] * A[j, k]
            end
            A[j, j] = sqrt(A[j, j])
            
            ajj = A[j, j]
            for i in (j+1):j_end
                for k in 1:(j-1)
                    A[i, j] -= A[i, k] * A[j, k]
                end
                A[i, j] /= ajj
            end
        end
        
        # Update trailing panel (parallel)
        if j_end < n
            @threads :static for i in (j_end+1):n
                @inbounds for j in jj:j_end
                    for k in 1:(j-1)
                        A[i, j] -= A[i, k] * A[j, k]
                    end
                    A[i, j] /= A[j, j]
                end
            end
            
            # Update trailing submatrix (parallel)
            @threads :static for jnew in (j_end+1):n
                @inbounds for k in jj:j_end
                    a_jk = A[jnew, k]
                    for i in jnew:n
                        A[i, jnew] -= A[i, k] * a_jk
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: BLAS-accelerated (using LinearAlgebra.cholesky!)
=============================================================================#
function kernel_cholesky_blas!(A::Matrix{Float64})
    # Use LAPACK potrf! via LinearAlgebra
    # This modifies A in-place, storing L in lower triangle
    LAPACK.potrf!('L', A)
    return nothing
end

#=============================================================================
 Strategy 5: Right-looking Cholesky
 - Update trailing matrix after each column
=============================================================================#
function kernel_cholesky_rightlooking!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Diagonal element
        A[j, j] = sqrt(A[j, j])
        ajj = A[j, j]
        
        # Scale column j
        for i in (j+1):n
            A[i, j] /= ajj
        end
        
        # Rank-1 update of trailing submatrix (parallel)
        if j < n
            @threads :static for k in (j+1):n
                @inbounds begin
                    a_kj = A[k, j]
                    for i in k:n
                        A[i, k] -= A[i, j] * a_kj
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 6: Left-looking Cholesky (parallel)
 - Each column computation independent once previous columns done
=============================================================================#
function kernel_cholesky_leftlooking!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Update column j from all previous columns (parallel inner loop)
        if j > 1 && n > 64  # Only parallelize for larger matrices
            # Parallel update of A[j:n, j]
            @threads :static for i in j:n
                @inbounds begin
                    sum_val = A[i, j]
                    for k in 1:(j-1)
                        sum_val -= A[i, k] * A[j, k]
                    end
                    A[i, j] = sum_val
                end
            end
        else
            for i in j:n
                for k in 1:(j-1)
                    A[i, j] -= A[i, k] * A[j, k]
                end
            end
        end
        
        # Diagonal and scale
        A[j, j] = sqrt(A[j, j])
        ajj = A[j, j]
        
        for i in (j+1):n
            A[i, j] /= ajj
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 7: Task-based recursive Cholesky
=============================================================================#
function kernel_cholesky_tasks!(A::Matrix{Float64}; min_block::Int=128)
    n = size(A, 1)
    
    if n <= min_block
        # Base case: sequential Cholesky
        kernel_cholesky_seq!(A)
        return nothing
    end
    
    # Split matrix into blocks
    mid = div(n, 2)
    
    # A = [A11  A12]
    #     [A21  A22]
    
    @views A11 = A[1:mid, 1:mid]
    @views A21 = A[(mid+1):n, 1:mid]
    @views A22 = A[(mid+1):n, (mid+1):n]
    
    # Factor A11
    kernel_cholesky_tasks!(A11, min_block=min_block)
    
    # Solve A21 * L11^T = A21 -> A21 = A21 / L11^T
    # Using triangular solve: A21 = A21 * inv(L11^T)
    L11 = LowerTriangular(A11)
    rdiv!(A21, L11')
    
    # Update A22 = A22 - A21 * A21^T
    # This is a rank-k update
    @threads :static for j in 1:(n-mid)
        @inbounds for k in 1:mid
            a_jk = A21[j, k]
            for i in j:(n-mid)
                A22[i, j] -= A21[i, k] * a_jk
            end
        end
    end
    
    # Factor A22
    kernel_cholesky_tasks!(A22, min_block=min_block)
    
    return nothing
end

# Helper: in-place right division A = A / B (where B is triangular)
function rdiv!(A::AbstractMatrix, B::UpperTriangular)
    n, m = size(A)
    @inbounds for i in 1:n
        for j in 1:m
            for k in 1:(j-1)
                A[i, j] -= A[i, k] * B.data[k, j]
            end
            A[i, j] /= B.data[j, j]
        end
    end
    return A
end

# Get kernel by strategy name
function get_kernel(strategy::String)
    kernels = Dict(
        "sequential" => kernel_cholesky_seq!,
        "threads" => kernel_cholesky_threads!,
        "blocked" => kernel_cholesky_blocked!,
        "blas" => kernel_cholesky_blas!,
        "rightlooking" => kernel_cholesky_rightlooking!,
        "leftlooking" => kernel_cholesky_leftlooking!,
        "tasks" => kernel_cholesky_tasks!
    )
    return get(kernels, strategy, kernel_cholesky_seq!)
end



end # module
