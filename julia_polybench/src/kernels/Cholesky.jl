# Corrected Cholesky Decomposition with Proper Parallelization
# 
# The original threading implementation was 2500x SLOWER than sequential
# because Cholesky has inherent sequential dependencies along the diagonal.
#
# This implementation uses blocked (right-looking) algorithm where only
# the trailing matrix update is parallelized.

module PolyBenchCholesky_Corrected

using LinearAlgebra
using Base.Threads
using BenchmarkTools
using Printf
using Statistics

export kernel_cholesky_seq!, kernel_cholesky_simd!
export kernel_cholesky_blocked_parallel!, kernel_cholesky_blas!
export kernel_cholesky_recursive_parallel!
export main, verify_implementations, run_benchmarks

# Dataset sizes
const DATASET_SIZES = Dict(
    "MINI" => 40,
    "SMALL" => 120,
    "MEDIUM" => 400,
    "LARGE" => 2000,
    "EXTRALARGE" => 4000
)

# Initialize positive-definite matrix (A = B * B' + n*I for stability)
function init_array!(A::Matrix{Float64})
    n = size(A, 1)
    
    # Create a random matrix and make it positive definite
    @inbounds for j in 1:n
        for i in 1:j
            A[i,j] = Float64(-j % n) / n + 1.0
            A[j,i] = A[i,j]
        end
        A[j,j] = 1.0  # Will be overwritten
    end
    
    # Make diagonally dominant (ensures positive definiteness)
    @inbounds for i in 1:n
        A[i,i] = Float64(n)
    end
    
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline (column-major optimized)
=============================================================================#
function kernel_cholesky_seq!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Update diagonal element
        for k in 1:j-1
            A[j,j] -= A[j,k]^2
        end
        A[j,j] = sqrt(A[j,j])
        
        # Update column below diagonal
        for i in j+1:n
            for k in 1:j-1
                A[i,j] -= A[i,k] * A[j,k]
            end
            A[i,j] /= A[j,j]
        end
    end
    
    # Zero out upper triangle (L is lower triangular)
    @inbounds for j in 2:n
        for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: SIMD-optimized (vectorize inner loops)
=============================================================================#
function kernel_cholesky_simd!(A::Matrix{Float64})
    n = size(A, 1)
    
    @inbounds for j in 1:n
        # Update diagonal - use SIMD for accumulation
        s = 0.0
        @simd for k in 1:j-1
            s += A[j,k]^2
        end
        A[j,j] -= s
        A[j,j] = sqrt(A[j,j])
        
        # Update column below diagonal - SIMD on inner reduction
        for i in j+1:n
            s = 0.0
            @simd for k in 1:j-1
                s += A[i,k] * A[j,k]
            end
            A[i,j] = (A[i,j] - s) / A[j,j]
        end
    end
    
    # Zero upper triangle
    @inbounds for j in 2:n
        @simd for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: BLAS (uses LAPACK's potrf)
=============================================================================#
function kernel_cholesky_blas!(A::Matrix{Float64})
    # LAPACK's dpotrf computes Cholesky in-place
    # 'L' means compute lower triangular L where A = L*L'
    LAPACK.potrf!('L', A)
    
    # Zero upper triangle for consistency
    n = size(A, 1)
    @inbounds for j in 2:n
        for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Blocked Parallel (RIGHT-LOOKING algorithm)
 
 This is the CORRECT way to parallelize Cholesky!
 
 The key insight: Only the trailing matrix update can be parallelized.
 The diagonal block factorization MUST be sequential.
 
 Algorithm:
 1. Factor diagonal block (sequential)
 2. Solve triangular system for panel (can use BLAS)
 3. Update trailing matrix (PARALLEL - this is where we win)
=============================================================================#
function kernel_cholesky_blocked_parallel!(A::Matrix{Float64}; block_size::Int=64)
    n = size(A, 1)
    
    for jb in 1:block_size:n
        jb_end = min(jb + block_size - 1, n)
        
        # 1. Factor diagonal block A[jb:jb_end, jb:jb_end] - SEQUENTIAL
        @inbounds for j in jb:jb_end
            # Update diagonal
            for k in jb:j-1
                A[j,j] -= A[j,k]^2
            end
            A[j,j] = sqrt(A[j,j])
            
            # Update column within block
            for i in j+1:jb_end
                for k in jb:j-1
                    A[i,j] -= A[i,k] * A[j,k]
                end
                A[i,j] /= A[j,j]
            end
        end
        
        # 2. Update panel below diagonal block - SEQUENTIAL (triangular solve)
        if jb_end < n
            @inbounds for j in jb:jb_end
                for i in jb_end+1:n
                    for k in jb:j-1
                        A[i,j] -= A[i,k] * A[j,k]
                    end
                    A[i,j] /= A[j,j]
                end
            end
        end
        
        # 3. Update trailing matrix - PARALLEL (this is the big win)
        # A[jb_end+1:n, jb_end+1:n] -= A[jb_end+1:n, jb:jb_end] * A[jb_end+1:n, jb:jb_end]'
        if jb_end < n
            remaining = n - jb_end
            
            # Parallelize over columns of trailing matrix
            @threads :static for j_offset in 1:remaining
                j = jb_end + j_offset
                @inbounds for i in j:n  # Only lower triangle
                    s = 0.0
                    @simd for k in jb:jb_end
                        s += A[i,k] * A[j,k]
                    end
                    A[i,j] -= s
                end
            end
        end
    end
    
    # Zero upper triangle
    @inbounds for j in 2:n
        for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: Alternative - Task-based with coarser granularity
=============================================================================#
function kernel_cholesky_tasks!(A::Matrix{Float64}; block_size::Int=128)
    n = size(A, 1)
    
    for jb in 1:block_size:n
        jb_end = min(jb + block_size - 1, n)
        
        # Factor diagonal block (sequential)
        factor_block!(A, jb, jb_end)
        
        if jb_end < n
            # Solve panel (sequential - uses BLAS trsm internally if available)
            solve_panel!(A, jb, jb_end, n)
            
            # Update trailing matrix with tasks
            num_blocks = cld(n - jb_end, block_size)
            @sync begin
                for ib in 1:num_blocks
                    i_start = jb_end + 1 + (ib - 1) * block_size
                    i_end = min(i_start + block_size - 1, n)
                    
                    Threads.@spawn update_trailing_block!(A, jb, jb_end, i_start, i_end, n)
                end
            end
        end
    end
    
    # Zero upper triangle
    @inbounds for j in 2:n
        for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

# Helper functions for task-based version
function factor_block!(A, jb, jb_end)
    @inbounds for j in jb:jb_end
        for k in jb:j-1
            A[j,j] -= A[j,k]^2
        end
        A[j,j] = sqrt(A[j,j])
        
        for i in j+1:jb_end
            for k in jb:j-1
                A[i,j] -= A[i,k] * A[j,k]
            end
            A[i,j] /= A[j,j]
        end
    end
end

function solve_panel!(A, jb, jb_end, n)
    @inbounds for j in jb:jb_end
        for i in jb_end+1:n
            for k in jb:j-1
                A[i,j] -= A[i,k] * A[j,k]
            end
            A[i,j] /= A[j,j]
        end
    end
end

function update_trailing_block!(A, jb, jb_end, i_start, i_end, n)
    @inbounds for j in i_start:n
        for i in max(j, i_start):i_end
            s = 0.0
            @simd for k in jb:jb_end
                s += A[i,k] * A[j,k]
            end
            A[i,j] -= s
        end
    end
end

#=============================================================================
 Strategy 6: Recursive blocked (for very large matrices)
=============================================================================#
function kernel_cholesky_recursive_parallel!(A::Matrix{Float64}; min_block::Int=64)
    n = size(A, 1)
    cholesky_recursive!(A, 1, n, min_block)
    
    # Zero upper triangle
    @inbounds for j in 2:n
        for i in 1:j-1
            A[i,j] = 0.0
        end
    end
    
    return nothing
end

function cholesky_recursive!(A, start_idx, end_idx, min_block)
    n = end_idx - start_idx + 1
    
    if n <= min_block
        # Base case: use sequential for small blocks
        @inbounds for j in start_idx:end_idx
            for k in start_idx:j-1
                A[j,j] -= A[j,k]^2
            end
            A[j,j] = sqrt(A[j,j])
            
            for i in j+1:end_idx
                for k in start_idx:j-1
                    A[i,j] -= A[i,k] * A[j,k]
                end
                A[i,j] /= A[j,j]
            end
        end
        return
    end
    
    mid = start_idx + div(n, 2) - 1
    
    # 1. Recursively factor top-left
    cholesky_recursive!(A, start_idx, mid, min_block)
    
    # 2. Solve for off-diagonal block (sequential)
    @inbounds for j in start_idx:mid
        for i in mid+1:end_idx
            for k in start_idx:j-1
                A[i,j] -= A[i,k] * A[j,k]
            end
            A[i,j] /= A[j,j]
        end
    end
    
    # 3. Update bottom-right (PARALLEL)
    @threads :static for j in mid+1:end_idx
        @inbounds for i in j:end_idx
            s = 0.0
            @simd for k in start_idx:mid
                s += A[i,k] * A[j,k]
            end
            A[i,j] -= s
        end
    end
    
    # 4. Recursively factor bottom-right
    cholesky_recursive!(A, mid+1, end_idx, min_block)
end

#=============================================================================
 Verification and Benchmarking
=============================================================================#
function verify_cholesky(A_original, L)
    # Compute L * L' and compare to original
    n = size(A_original, 1)
    reconstructed = L * L'
    
    max_error = 0.0
    @inbounds for j in 1:n
        for i in j:n  # Only lower triangle matters for symmetric
            max_error = max(max_error, abs(A_original[i,j] - reconstructed[i,j]))
        end
    end
    
    return max_error
end

function verify_implementations(dataset="SMALL"; tolerance=1e-10)
    n = DATASET_SIZES[dataset]
    
    # Create reference matrix
    A_ref = Matrix{Float64}(undef, n, n)
    init_array!(A_ref)
    
    # Get BLAS reference result
    A_blas = copy(A_ref)
    kernel_cholesky_blas!(A_blas)
    
    implementations = Dict(
        "Sequential" => kernel_cholesky_seq!,
        "SIMD" => kernel_cholesky_simd!,
        "Blocked Parallel" => kernel_cholesky_blocked_parallel!,
        "Tasks" => kernel_cholesky_tasks!,
        "Recursive" => kernel_cholesky_recursive_parallel!
    )
    
    println("Verifying Cholesky implementations on $dataset (n=$n)...")
    println("-"^60)
    @printf("%-20s | %15s | %10s\n", "Implementation", "Max Error", "Status")
    println("-"^60)
    
    for (name, func) in implementations
        A_test = copy(A_ref)
        func(A_test)
        
        error = maximum(abs.(A_blas - A_test))
        status = error < tolerance ? "PASS" : "FAIL"
        
        @printf("%-20s | %15.2e | %10s\n", name, error, status)
    end
end

function run_benchmarks(dataset_name="MEDIUM"; samples=10, seconds=5)
    n = DATASET_SIZES[dataset_name]
    
    # Compute FLOPs for Cholesky: n^3/3 + n^2/2 (approximately)
    flops = Float64(n)^3 / 3.0
    
    println("\n", "="^70)
    println("CHOLESKY DECOMPOSITION BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("Dataset: $dataset_name (n=$n)")
    println("Memory: $(round(n * n * 8 / 1024^2, digits=2)) MB")
    println("FLOPs: $(round(flops / 1e9, digits=2)) GFLOP")
    println()
    
    implementations = [
        ("sequential", kernel_cholesky_seq!),
        ("simd", kernel_cholesky_simd!),
        ("blocked_parallel", kernel_cholesky_blocked_parallel!),
        ("tasks", kernel_cholesky_tasks!),
        ("blas", kernel_cholesky_blas!),
    ]
    
    # Reference matrix
    A_ref = Matrix{Float64}(undef, n, n)
    init_array!(A_ref)
    
    # Baseline
    A_test = copy(A_ref)
    baseline_trial = @benchmark kernel_cholesky_seq!($A_test) setup=(A_test = copy($A_ref)) samples=samples seconds=seconds
    baseline_ns = minimum(baseline_trial).time
    
    println("-"^70)
    @printf("%-18s | %10s | %10s | %10s | %8s | %8s\n",
            "Strategy", "Min(ms)", "Med(ms)", "Mean(ms)", "GFLOP/s", "Speedup")
    println("-"^70)
    
    for (name, func) in implementations
        A_test = copy(A_ref)
        trial = @benchmark $func($A_test) setup=(A_test = copy($A_ref)) samples=samples seconds=seconds
        
        min_time = minimum(trial).time / 1e6
        med_time = median(trial).time / 1e6
        mean_time = mean(trial).time / 1e6
        gflops = flops / (minimum(trial).time)  # GFLOP/s
        speedup = baseline_ns / minimum(trial).time
        
        @printf("%-18s | %10.3f | %10.3f | %10.3f | %8.2f | %7.2fx\n",
                name, min_time, med_time, mean_time, gflops, speedup)
    end
    
    println("-"^70)
end

function main(; datasets=["SMALL", "MEDIUM"])
    # Configure BLAS
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    end
    
    println("Configuration:")
    println("  Julia threads: $(Threads.nthreads())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    
    # Verify first
    verify_implementations("SMALL")
    
    # Run benchmarks
    for dataset in datasets
        run_benchmarks(dataset)
    end
end

end # module
