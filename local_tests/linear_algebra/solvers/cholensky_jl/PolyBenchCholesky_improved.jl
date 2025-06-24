module PolyBenchCholesky_improved

# PolyBench Cholesky Decomposition - RACE CONDITION FREE VERSION
# 
# CRITICAL FIX: Cholesky has extremely strict data dependencies that make
# inner-loop parallelization unsafe. This version uses conservative threading
# that respects the algorithm's inherent sequential nature.

using BenchmarkTools
using Distributed
using LinearAlgebra
using Statistics
using Base.Threads
using Printf
using SharedArrays

# Configure BLAS threads
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        @info "Set BLAS threads to 1 (Julia using $(Threads.nthreads()) threads)"
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
        @info "Set BLAS threads to $(Sys.CPU_THREADS)"
    end
end

# Dataset sizes
const DATASET_SIZES = Dict(
    "MINI" => 40,
    "SMALL" => 120, 
    "MEDIUM" => 400,
    "LARGE" => 2000,
    "EXTRALARGE" => 4000
)

# FIXED: Initialize matrix exactly as in PolyBench C code
function init_array!(A)
    n = size(A, 1)
    
    # Step 1: Initialize according to PolyBench specification
    @inbounds for i = 1:n
        for j = 1:i
            A[i,j] = Float64((-j % n) / n + 1)
        end
        for j = (i+1):n
            A[i,j] = 0.0
        end
        A[i,i] = 1.0
    end
    
    # Step 2: Make positive semi-definite A = B*B^T
    B = zeros(Float64, n, n)
    
    # B = A * A^T computation (column-major optimized)
    @inbounds for r = 1:n
        for s = 1:n
            for t = 1:n
                B[r,s] += A[r,t] * A[s,t]
            end
        end
    end
    
    # Copy result back
    copyto!(A, B)
    
    return nothing
end

# 1. Reference Sequential Implementation (exactly from cholesky.c)
function kernel_cholesky_seq!(A)
    n = size(A, 1)
    
    @inbounds for i = 1:n
        # j < i: Update L[i,j] elements
        for j = 1:(i-1)
            for k = 1:(j-1)
                A[i,j] -= A[i,k] * A[j,k]
            end
            A[i,j] /= A[j,j]
        end
        
        # i == j case: Compute diagonal element
        for k = 1:(i-1)
            A[i,i] -= A[i,k] * A[i,k]
        end
        A[i,i] = sqrt(A[i,i])
    end
    
    return A
end

# 2. SIMD version (safe - no threading)
function kernel_cholesky_simd!(A)
    n = size(A, 1)
    
    @inbounds for i = 1:n
        for j = 1:(i-1)
            sum_val = 0.0
            @simd for k = 1:(j-1)
                sum_val += A[i,k] * A[j,k]
            end
            A[i,j] -= sum_val
            A[i,j] /= A[j,j]
        end
        
        sum_val = 0.0
        @simd for k = 1:(i-1)
            sum_val += A[i,k] * A[i,k]
        end
        A[i,i] -= sum_val
        A[i,i] = sqrt(A[i,i])
    end
    
    return A
end

# 3. SAFE Threading Implementation - Block-Level Parallelization Only
function kernel_cholesky_threads!(A)
    n = size(A, 1)
    nthreads = Threads.nthreads()
    
    # CRITICAL INSIGHT: Cholesky's inner loops have READ-AFTER-WRITE dependencies
    # that make fine-grained parallelization extremely dangerous.
    # We can only safely parallelize at the block level or with careful synchronization.
    
    # Conservative approach: Use threading only for independent preparatory work
    # The main algorithm remains sequential to avoid race conditions
    
    if nthreads > 1 && n >= 200
        # For large matrices, we can use block-wise approach
        return kernel_cholesky_blocked_threads!(A)
    else
        # For smaller matrices or single thread, use sequential
        return kernel_cholesky_seq!(A)
    end
end

# 4. Block-wise threading implementation (safe parallelization)
function kernel_cholesky_blocked_threads!(A)
    n = size(A, 1)
    nthreads = Threads.nthreads()
    block_size = max(32, n ÷ (2*nthreads))  # Conservative block size
    
    @inbounds for block_start = 1:block_size:n
        block_end = min(block_start + block_size - 1, n)
        
        # SEQUENTIAL: Factor the diagonal block (cannot be parallelized)
        for i = block_start:block_end
            for j = block_start:(i-1)
                for k = block_start:(j-1)
                    A[i,j] -= A[i,k] * A[j,k]
                end
                A[i,j] /= A[j,j]
            end
            
            for k = block_start:(i-1)
                A[i,i] -= A[i,k] * A[i,k]
            end
            A[i,i] = sqrt(A[i,i])
        end
        
        # PARALLEL: Update sub-diagonal blocks (these are independent)
        if block_end < n
            sub_blocks = collect((block_end+1):block_size:n)
            
            @threads :static for block_idx in 1:length(sub_blocks)
                sub_start = sub_blocks[block_idx]
                sub_end = min(sub_start + block_size - 1, n)
                
                # Update this sub-block
                for j = block_start:block_end
                    for i = sub_start:sub_end
                        if i > j  # Lower triangular only
                            for k = block_start:(j-1)
                                A[i,j] -= A[i,k] * A[j,k]
                            end
                            A[i,j] /= A[j,j]
                        end
                    end
                end
            end
        end
    end
    
    return A
end

# 5. BLAS Implementation (corrected)
function kernel_cholesky_blas!(A)
    try
        # Make a copy to avoid modifying input during error checking
        A_test = copy(A)
        
        # Check if matrix is positive definite first
        F = cholesky!(Hermitian(A_test, :L))
        
        # If successful, apply to original matrix
        F = cholesky!(Hermitian(A, :L))
        
        # Zero upper triangular part
        n = size(A, 1)
        @inbounds for j = 1:n
            for i = 1:(j-1)
                A[i,j] = 0.0
            end
        end
        
    catch e
        @warn "BLAS Cholesky failed: $e, using sequential fallback"
        return kernel_cholesky_seq!(A)
    end
    
    return A
end

# 6. Cache-optimized tiled implementation
function kernel_cholesky_tiled!(A; tile_size=64)
    n = size(A, 1)
    
    @inbounds for tile_k = 1:tile_size:n
        tile_end = min(tile_k + tile_size - 1, n)
        
        # Factor diagonal tile (sequential)
        for i = tile_k:tile_end
            for j = tile_k:(i-1)
                for k = tile_k:(j-1)
                    A[i,j] -= A[i,k] * A[j,k]
                end
                A[i,j] /= A[j,j]
            end
            
            for k = tile_k:(i-1)
                A[i,i] -= A[i,k] * A[i,k]
            end
            A[i,i] = sqrt(A[i,i])
        end
        
        # Update sub-diagonal tiles
        for tile_i = (tile_end+1):tile_size:n
            tile_i_end = min(tile_i + tile_size - 1, n)
            
            for j = tile_k:tile_end
                for i = tile_i:tile_i_end
                    if i > j
                        for k = tile_k:(j-1)
                            A[i,j] -= A[i,k] * A[j,k]
                        end
                        A[i,j] /= A[j,j]
                    end
                end
            end
        end
    end
    
    return A
end

# 7. SharedArray implementation for distributed
function kernel_cholesky_shared!(n)
    A = SharedArray{Float64}(n, n)
    init_array!(A)
    
    # Cholesky is inherently sequential, limited distributed benefit
    # This is mainly for demonstration
    return kernel_cholesky_seq!(sdata(A))
end

# Benchmark with zero allocations
function benchmark_cholesky(n, impl_func, impl_name; 
                           samples=BenchmarkTools.DEFAULT_PARAMETERS.samples,
                           seconds=BenchmarkTools.DEFAULT_PARAMETERS.seconds)
    
    A_original = Matrix{Float64}(undef, n, n)
    A_work = Matrix{Float64}(undef, n, n)
    
    init_array!(A_original)
    
    b = @benchmarkable begin
        copyto!($A_work, $A_original)
        $impl_func($A_work)
    end samples=samples seconds=seconds evals=1
    
    return run(b)
end

# Run benchmarks
function run_benchmarks(dataset_name; show_memory=true)
    n = DATASET_SIZES[dataset_name]
    
    println("\n" * "="^80)
    println("Dataset: $dataset_name (n=$n)")
    println("Julia version: $(VERSION)")
    println("Number of threads: $(Threads.nthreads())")
    println("Number of processes: $(nprocs())")
    println("="^80)
    
    implementations = Dict(
        "Sequential" => kernel_cholesky_seq!,
        "SIMD" => kernel_cholesky_simd!,
        "Threads" => kernel_cholesky_threads!,
        "BLAS" => kernel_cholesky_blas!,
        "Tiled" => kernel_cholesky_tiled!
    )
    
    results = Dict{String, Any}()
    
    println("\nImplementation        | Min Time (ms) | Median (ms) | Mean (ms) | Memory (MB) | Allocs")
    println("-"^80)
    
    for (name, func) in implementations
        try
            trial = benchmark_cholesky(n, func, name)
            results[name] = trial
            
            min_time = minimum(trial).time / 1e6
            median_time = median(trial).time / 1e6
            mean_time = mean(trial).time / 1e6
            memory_mb = trial.memory / 1024^2
            allocs = trial.allocs
            
            @printf("%-20s | %13.3f | %11.3f | %9.3f | %11.2f | %6d\n", 
                    name, min_time, median_time, mean_time, memory_mb, allocs)
        catch e
            println("ERROR in $name: $e")
            continue
        end
    end
    
    if haskey(results, "Sequential")
        seq_time = minimum(results["Sequential"]).time
        println("\nSpeedups (relative to Sequential):")
        println("-"^40)
        for (name, trial) in results
            if name != "Sequential"
                speedup = seq_time / minimum(trial).time
                @printf("%-20s | %6.2fx\n", name, speedup)
            end
        end
    end
    
    return results
end

# Verify correctness with detailed error checking
function verify_implementations(dataset="SMALL"; tolerance=1e-12)
    n = DATASET_SIZES[dataset]
    
    println("Verifying implementation correctness...")
    println("Dataset: $dataset (n=$n)")
    println("-"^50)
    
    # Create reference matrix
    A_ref = Matrix{Float64}(undef, n, n)
    init_array!(A_ref)
    
    # Verify matrix is positive definite
    try
        eigenvals = eigvals(Hermitian(A_ref, :L))
        min_eigval = minimum(eigenvals)
        if min_eigval <= 0
            println("ERROR: Matrix is not positive definite (min eigenvalue: $min_eigval)")
            return false
        end
        println("✓ Input matrix is positive definite (min eigenvalue: $(round(min_eigval, digits=6)))")
    catch e
        println("ERROR checking positive definiteness: $e")
        return false
    end
    
    # Generate reference solution
    A_ref_copy = copy(A_ref)
    kernel_cholesky_seq!(A_ref_copy)
    
    if any(isnan, A_ref_copy) || any(isinf, A_ref_copy)
        println("ERROR: Reference solution contains NaN or Inf!")
        return false
    end
    
    implementations = Dict(
        "SIMD" => kernel_cholesky_simd!,
        "Threads" => kernel_cholesky_threads!,
        "BLAS" => kernel_cholesky_blas!,
        "Tiled" => kernel_cholesky_tiled!,
    )
    
    println("Implementation        | Max Absolute Error | Status")
    println("-"^50)
    
    all_passed = true
    for (name, func) in implementations
        try
            A_test = copy(A_ref)
            func(A_test)
            
            if any(isnan, A_test) || any(isinf, A_test)
                println("$(lpad(name, 20)) | $(lpad("NaN/Inf detected", 17)) | FAIL")
                all_passed = false
                continue
            end
            
            max_error = maximum(abs.(A_ref_copy - A_test))
            status = max_error < tolerance ? "PASS" : "FAIL"
            
            @printf("%-20s | %17.2e | %s\n", name, max_error, status)
            
            if status == "FAIL"
                all_passed = false
            end
        catch e
            println("$(lpad(name, 20)) | $(lpad("Exception", 17)) | FAIL")
            println("    Error: $e")
            all_passed = false
        end
    end
    
    if all_passed
        println("\n✅ All implementations passed verification!")
    else
        println("\n❌ Some implementations failed verification!")
    end
    
    return all_passed
end

# Debug allocations
function debug_threading_allocations(dataset="SMALL")
    println("\n" * "="^80)
    println("DEBUGGING ALLOCATION PATTERNS")
    println("="^80)
    
    n = DATASET_SIZES[dataset]
    A_original = Matrix{Float64}(undef, n, n)
    init_array!(A_original)
    
    println("Testing allocation patterns for $dataset (n=$n)...")
    
    implementations = [
        ("Sequential", kernel_cholesky_seq!),
        ("SIMD", kernel_cholesky_simd!),
        ("Threads", kernel_cholesky_threads!),
        ("BLAS", kernel_cholesky_blas!)
    ]
    
    for (name, func) in implementations
        try
            A_test = copy(A_original)
            allocs = @allocated func(A_test)
            println("$name allocations: $allocs bytes")
            
            # Also verify no domain errors
            if any(isnan, A_test) || any(isinf, A_test)
                println("  ⚠️  $name produced NaN/Inf values!")
            else
                println("  ✓ $name completed successfully")
            end
        catch e
            println("$name: ERROR - $e")
        end
    end
    
    return nothing
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM"])
    println("PolyBench Cholesky - RACE CONDITION FREE VERSION")
    println("="^60)
    
    configure_blas_threads()
    
    println("\n📋 IMPORTANT NOTES:")
    println("- Cholesky decomposition has strict data dependencies")
    println("- Limited parallelization opportunities compared to matrix multiplication")
    println("- Threading focuses on block-level parallelism to avoid race conditions")
    println("- Expected threading speedup: 1.0x-1.3x (not 8x like matrix multiply)")
    
    if implementation == "all"
        println("\n🔍 VERIFYING CORRECTNESS...")
        verify_passed = verify_implementations()
        
        println("\n🐛 DEBUGGING ALLOCATIONS...")
        debug_threading_allocations()
        
        if verify_passed
            println("\n🚀 RUNNING BENCHMARKS...")
            for dataset in datasets
                if dataset in keys(DATASET_SIZES)
                    try
                        run_benchmarks(dataset)
                    catch e
                        println("ERROR running $dataset: $e")
                        continue
                    end
                else
                    println("Unknown dataset: $dataset")
                end
            end
        else
            println("\n❌ Skipping benchmarks due to verification failures")
        end
        
        println("\n📊 PERFORMANCE EXPECTATIONS:")
        println("="^40)
        println("Sequential:  1.0x   (baseline)")
        println("SIMD:       1.1-1.2x (modest vectorization gains)")
        println("Threads:    1.0-1.3x (limited by algorithm dependencies)")
        println("BLAS:       2-5x     (highly optimized implementation)")
        println("Tiled:      1.1-1.5x (cache optimization)")
        println("\nNote: Cholesky ≠ Matrix Multiplication in parallelization potential!")
        
    else
        error("Specific implementation mode not implemented")
    end
end

export main, verify_implementations, run_benchmarks, debug_threading_allocations

end # module