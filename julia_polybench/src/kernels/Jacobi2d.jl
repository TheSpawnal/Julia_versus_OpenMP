module Jacobi2D

using Base.Threads
using Printf

export STRATEGIES_JACOBI2D, DATASETS_JACOBI2D
export init_jacobi2d!, get_kernel
export kernel_jacobi2d_seq!, kernel_jacobi2d_threads!
export kernel_jacobi2d_tiled!, kernel_jacobi2d_redblack!

const STRATEGIES_JACOBI2D = [
    "sequential",
    "threads",
    "tiled",
    "redblack"
]

const DATASETS_JACOBI2D = Dict(
    "MINI" => (n=30, tsteps=20),
    "SMALL" => (n=90, tsteps=40),
    "MEDIUM" => (n=250, tsteps=100),
    "LARGE" => (n=1300, tsteps=500),
    "EXTRALARGE" => (n=2800, tsteps=1000)
)

# Initialize grids following PolyBench pattern
function init_jacobi2d!(A::Matrix{Float64}, B::Matrix{Float64})
    n = size(A, 1)
    
    # Column-major initialization
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = Float64((i - 1) * (j - 1) + 2) / n
        B[i, j] = Float64((i - 1) * (j - 1) + 3) / n
    end
    
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Double buffering with pointer swap
=============================================================================#
function kernel_jacobi2d_seq!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    for t in 1:tsteps
        # A -> B (5-point stencil)
        @inbounds for j in 2:(n-1)
            for i in 2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # B -> A
        @inbounds for j in 2:(n-1)
            for i in 2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded row-wise parallel
 - Parallelizes over rows (inner dimension for column-major)
=============================================================================#
function kernel_jacobi2d_threads!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    for t in 1:tsteps
        # A -> B
        @threads :static for j in 2:(n-1)
            @inbounds for i in 2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
            end
        end
        
        # B -> A
        @threads :static for j in 2:(n-1)
            @inbounds for i in 2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Tiled/Blocked for cache optimization
=============================================================================#
function kernel_jacobi2d_tiled!(
    A::Matrix{Float64},
    B::Matrix{Float64},
    tsteps::Int;
    tile_size::Int=64
)
    n = size(A, 1)
    ts = tile_size
    
    for t in 1:tsteps
        # A -> B (tiled)
        @threads :static for jj in 2:ts:(n-1)
            j_end = min(jj + ts - 1, n - 1)
            for ii in 2:ts:(n-1)
                i_end = min(ii + ts - 1, n - 1)
                
                @inbounds for j in jj:j_end
                    for i in ii:i_end
                        B[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
                    end
                end
            end
        end
        
        # B -> A (tiled)
        @threads :static for jj in 2:ts:(n-1)
            j_end = min(jj + ts - 1, n - 1)
            for ii in 2:ts:(n-1)
                i_end = min(ii + ts - 1, n - 1)
                
                @inbounds for j in jj:j_end
                    for i in ii:i_end
                        A[i, j] = 0.2 * (B[i, j] + B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1])
                    end
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Red-Black (Gauss-Seidel) ordering
 - Better convergence, allows in-place update
=============================================================================#
function kernel_jacobi2d_redblack!(A::Matrix{Float64}, B::Matrix{Float64}, tsteps::Int)
    n = size(A, 1)
    
    # Use A as primary, B as temporary
    copyto!(B, A)
    
    for t in 1:tsteps
        # Red phase (checkerboard: (i+j) % 2 == 0)
        @threads :static for j in 2:(n-1)
            @inbounds for i in 2:(n-1)
                if (i + j) % 2 == 0
                    A[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
                end
            end
        end
        
        # Black phase (checkerboard: (i+j) % 2 == 1)
        @threads :static for j in 2:(n-1)
            @inbounds for i in 2:(n-1)
                if (i + j) % 2 == 1
                    A[i, j] = 0.2 * (A[i, j] + A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1])
                end
            end
        end
    end
    
    # Copy result to B for consistency with other strategies
    copyto!(B, A)
    
    return nothing
end

# Get kernel by strategy name
function get_kernel(strategy::AbstractString)
    kernels = Dict(
        "sequential" => kernel_jacobi2d_seq!,
        "threads" => kernel_jacobi2d_threads!,
        "tiled" => kernel_jacobi2d_tiled!,
        "redblack" => kernel_jacobi2d_redblack!
    )
    return get(kernels, String(strategy), kernel_jacobi2d_seq!)
end

end # module
