module PolyBenchCholesky

# PolyBench Cholesky Decomposition Benchmark for Julia
#
# This module implements the Cholesky decomposition kernel from PolyBench
# with comprehensive optimizations addressing the unique challenges of this
# inherently sequential algorithm.
#
# The kernel computes the Cholesky decomposition A = LL^T of a positive-definite matrix A,
# where L is a lower triangular matrix.
#
# Usage Examples:
# 
# 1. Basic Benchmarking
# ```julia
# # Start Julia with threads: julia -t 16
# include("PolyBenchCholesky.jl")
# using .PolyBenchCholesky
# 
# # Run all implementations
# PolyBenchCholesky.main()
# 
# # Verify correctness first
# PolyBenchCholesky.verify_implementations("SMALL")
# ```
#
# 2. Distributed Computing
# ```julia
# # Terminal: julia -t 1  # Single thread per process
# using Distributed
# addprocs(8)  # Add 8 worker processes
# 
# @everywhere include("PolyBenchCholesky.jl")
# @everywhere using .PolyBenchCholesky
# 
# # Run with distributed implementations
# PolyBenchCholesky.main(distributed=true, datasets=["LARGE"])
# ```
#
# 3. Advanced Performance Analysis
# ```julia
# # Comprehensive analysis across multiple datasets
# results = PolyBenchCholesky.performance_analysis(["SMALL", "MEDIUM", "LARGE"])
# 
# # Cache optimization analysis
# best_tile = PolyBenchCholesky.optimize_tile_size("MEDIUM")
# 
# # System configuration check
# PolyBenchCholesky.check_system_configuration()
# ```


using BenchmarkTools
using Distributed
using LinearAlgebra
using Statistics
using Base.Threads
using Printf
using SharedArrays

# Configure BLAS threads based on Julia configuration
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        @info "Set BLAS threads to 1 (Julia using $(Threads.nthreads()) threads)"
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
        @info "Set BLAS threads to $(Sys.CPU_THREADS)"
    end
end

# Dataset sizes according to PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => 40,
    "SMALL" => 120,
    "MEDIUM" => 400,
    "LARGE" => 2000,
    "EXTRALARGE" => 4000
)

# Initialize matrix following PolyBench specification
function init_array!(A)
    n = size(A, 1)
    
    # COLUMN-MAJOR OPTIMIZATION: Initialize by columns for better cache performance
    # Julia stores A[:,j] (column j) contiguously in memory
    
    # Initialize lower triangular part (column by column for cache efficiency)
    @inbounds for j = 1:n  # Outer loop over columns (cache-friendly)
        for i = j:n        # Inner loop over rows in each column
            if i == j
                A[i,j] = 1.0
            elseif i > j
                A[i,j] = Float64((-j % n) / n + 1)
            else  # i < j (upper triangular)
                A[i,j] = 0.0
            end
        end
    end
    
    # Make the matrix positive semi-definite: A = B*B^T
    # This computation is inherently matrix multiplication, also optimize for column-major
    B = similar(A)
    B .= 0.0
    
    # Column-major optimized matrix multiplication: B = A * A^T
    @inbounds for j = 1:n      # Result column
        for i = 1:n            # Result row  
            local_sum = 0.0
            for k = 1:n        # Inner product
                local_sum += A[i,k] * A[j,k]  # Note: A[j,k] for transpose
            end
            B[i,j] = local_sum
        end
    end
    
    # Copy back to A (column by column)
    A .= B
    
    return nothing
end

# 1. Sequential Implementation (baseline) - Following PolyBench C reference
function kernel_cholesky_seq!(A)
    n = size(A, 1)
    
    @inbounds for i = 1:n
        # j < i: Update L[i,j] elements
        for j = 1:(i-1)
            # Compute dot product: A[i,j] -= sum(A[i,k] * A[j,k] for k=1:j-1)
            for k = 1:(j-1)
                A[i,j] -= A[i,k] * A[j,k]
            end
            # Divide by diagonal element
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

# 2. SIMD Optimized Implementation
function kernel_cholesky_simd!(A)
    n = size(A, 1)
    
    @inbounds for i = 1:n
        # j < i: Update L[i,j] elements with SIMD
        for j = 1:(i-1)
            dot_product = 0.0
            @simd for k = 1:(j-1)
                dot_product += A[i,k] * A[j,k]
            end
            A[i,j] -= dot_product
            A[i,j] /= A[j,j]
        end
        
        # i == j case: Compute diagonal element with SIMD
        dot_product = 0.0
        @simd for k = 1:(i-1)
            dot_product += A[i,k] * A[i,k]
        end
        A[i,i] -= dot_product
        A[i,i] = sqrt(A[i,i])
    end
    
    return A
end

# 3. Multithreaded Implementation (optimized for Julia's column-major layout)
function kernel_cholesky_threads!(A)
    n = size(A, 1)
    
    # IMPORTANT: Julia uses COLUMN-MAJOR memory layout
    # A[i,j] then A[i+1,j] (same column) = cache-friendly
    # A[i,j] then A[i,j+1] (same row) = NOT cache-friendly
    
    @inbounds for i = 1:n
        # j < i: Update L[i,j] elements
        # Note: We access A[i,k] and A[j,k] which are row-wise accesses (not optimal)
        # But this is inherent to the Cholesky algorithm structure
        for j = 1:(i-1)
            if j > 32  # Only parallelize for larger inner loops to amortize overhead
                # Parallel reduction for dot product
                # We parallelize over k dimension, but each thread still does row-wise access
                dot_product = zeros(Threads.nthreads()) # //bug, this allocates every iteration.
                chunk_size = max(1, (j-1) ÷ Threads.nthreads())
                
                @threads :static for tid = 1:Threads.nthreads()
                    start_k = 1 + (tid-1) * chunk_size
                    end_k = min(start_k + chunk_size - 1, j-1)
                    if start_k <= end_k
                        local_sum = 0.0
                        for k = start_k:end_k
                            local_sum += A[i,k] * A[j,k]
                        end
                        dot_product[tid] = local_sum
                    end
                end
                A[i,j] -= sum(dot_product)
            else
                # Sequential for small loops to avoid threading overhead
                local_sum = 0.0
                for k = 1:(j-1)
                    local_sum += A[i,k] * A[j,k]
                end
                A[i,j] -= local_sum
            end
            A[i,j] /= A[j,j]
        end
        
        # i == j case: Compute diagonal element
        if i > 32
            # Parallel reduction for diagonal computation
            dot_product = zeros(Threads.nthreads())
            chunk_size = max(1, (i-1) ÷ Threads.nthreads())
            
            @threads :static for tid = 1:Threads.nthreads()
                start_k = 1 + (tid-1) * chunk_size
                end_k = min(start_k + chunk_size - 1, i-1)
                if start_k <= end_k
                    local_sum = 0.0
                    for k = start_k:end_k
                        local_sum += A[i,k] * A[i,k]
                    end
                    dot_product[tid] = local_sum
                end
            end
            A[i,i] -= sum(dot_product)
        else
            local_sum = 0.0
            for k = 1:(i-1)
                local_sum += A[i,k] * A[i,k]
            end
            A[i,i] -= local_sum
        end
        A[i,i] = sqrt(A[i,i])
    end
    
    return A
end

# 4. BLAS Implementation using LinearAlgebra
function kernel_cholesky_blas!(A)
    # Use Julia's built-in Cholesky decomposition
    # Note: Julia's cholesky returns Upper triangular by default
    # We need lower triangular, so we use cholesky with Val(false)
    
    try
        F = cholesky!(Hermitian(A, :L))
        # The result is stored in the lower triangular part
        # Zero out the upper triangular part
        n = size(A, 1)
        @inbounds for i = 1:n
            for j = (i+1):n
                A[i,j] = 0.0
            end
        end
    catch e
        # If BLAS cholesky fails (matrix not positive definite), 
        # fall back to sequential
        @warn "BLAS Cholesky failed, falling back to sequential: $e"
        return kernel_cholesky_seq!(A)
    end
    
    return A
end

# 5. Tiled Implementation optimized for Julia's column-major layout
function kernel_cholesky_tiled!(A; tile_size=64)
    n = size(A, 1)
    
    # COLUMN-MAJOR OPTIMIZATION: 
    # Julia stores A[:,j] (column j) contiguously in memory
    # Tiling should prioritize column-wise access patterns when possible
    
    # Block-wise Cholesky decomposition with column-major awareness
    @inbounds for kk = 1:tile_size:n
        end_k = min(kk + tile_size - 1, n)
        
        # Diagonal block: Standard Cholesky on this block
        # This part is inherently row-wise due to algorithm structure
        for i = kk:end_k
            for j = kk:(i-1)
                # Dot product computation - unavoidably row-wise access
                local_sum = 0.0
                for k = kk:(j-1)
                    local_sum += A[i,k] * A[j,k]
                end
                A[i,j] -= local_sum
                A[i,j] /= A[j,j]
            end
            
            # Diagonal element computation
            local_sum = 0.0
            for k = kk:(i-1)
                local_sum += A[i,k] * A[i,k]
            end
            A[i,i] -= local_sum
            A[i,i] = sqrt(A[i,i])
        end
        
        # Update blocks below diagonal block
        # COLUMN-MAJOR OPTIMIZATION: Process by columns to improve cache locality
        for jj = kk:tile_size:(end_k-1)
            j_start = jj
            j_end = min(jj + tile_size - 1, end_k-1)
            
            for ii = (end_k+1):tile_size:n
                i_start = ii
                i_end = min(ii + tile_size - 1, n)
                
                # Process this tile column by column (cache-friendly for Julia)
                for j = j_start:j_end
                    for i = i_start:i_end
                        if i > j  # Only lower triangular
                            # This inner loop is still row-wise, but we've organized
                            # the outer loops to be more cache-friendly
                            local_sum = 0.0
                            for k = kk:(j-1)
                                local_sum += A[i,k] * A[j,k]
                            end
                            A[i,j] -= local_sum
                            A[i,j] /= A[j,j]
                        end
                    end
                end
            end
        end
    end
    
    return A
end

# 6. Column-wise Distributed Implementation (Julia column-major optimized)
function kernel_cholesky_distributed!(A)
    n = size(A, 1)
    nw = nworkers()
    
    if nw <= 1
        @warn "No workers available for distributed implementation, falling back to sequential"
        return kernel_cholesky_seq!(A)
    end
    
    # COLUMN-MAJOR OPTIMIZATION: Distribute work column-wise rather than row-wise
    # since Julia stores columns contiguously in memory
    
    @inbounds for i = 1:n
        # Sequential part that must be done in order (inherent to Cholesky)
        for j = 1:(i-1)
            local_sum = 0.0
            for k = 1:(j-1)
                local_sum += A[i,k] * A[j,k]
            end
            A[i,j] -= local_sum
            A[i,j] /= A[j,j]
        end
        
        # Diagonal element computation
        local_sum = 0.0
        for k = 1:(i-1)
            local_sum += A[i,k] * A[i,k]
        end
        A[i,i] -= local_sum
        A[i,i] = sqrt(A[i,i])
        
        # Limited parallel opportunity: Update future columns
        # This is more of a demonstration since Cholesky has limited parallelism
        if i < n - 20  # Only if there's substantial remaining work
            
            # Distribute remaining columns across workers (column-major friendly)
            remaining_cols = (i+1):min(i+10, n)  # Limited lookahead
            cols_per_worker = max(1, length(remaining_cols) ÷ nw)
            
            @sync begin
                for (idx, w) in enumerate(workers())
                    start_col = i + 1 + (idx-1) * cols_per_worker
                    end_col = min(i + idx * cols_per_worker, min(i+10, n))
                    
                    if start_col <= end_col && start_col <= n
                        @async begin
                            # Send column data to worker (cache-friendly)
                            cols_data = A[:, start_col:end_col]
                            result = remotecall_fetch(w, cols_data, i) do local_cols, completed_row
                                # Prepare columns for next iterations
                                # (Limited work available due to algorithm constraints)
                                return local_cols
                            end
                            A[:, start_col:end_col] .= result
                        end
                    end
                end
            end
        end
    end
    
    return A
end

# 7. SharedArray implementation for multi-process shared memory
function kernel_cholesky_shared!(n)
    # Create shared array
    A = SharedArray{Float64}(n, n)
    
    # Initialize array
    init_array!(A)
    
    # Distributed Cholesky with shared memory
    @sync @distributed for i = 1:n
        # Most of the work is still sequential due to dependencies
        # This is more of a demonstration than an efficient implementation
        if myid() == workers()[1]  # Only first worker does the actual computation
            for j = 1:(i-1)
                for k = 1:(j-1)
                    A[i,j] -= A[i,k] * A[j,k]
                end
                A[i,j] /= A[j,j]
            end
            
            for k = 1:(i-1)
                A[i,i] -= A[i,k] * A[i,k]
            end
            A[i,i] = sqrt(A[i,i])
        end
    end
    
    return sdata(A)
end

# Benchmark runner with proper memory pre-allocation
function benchmark_cholesky(n, impl_func, impl_name; 
                           samples=BenchmarkTools.DEFAULT_PARAMETERS.samples,
                           seconds=BenchmarkTools.DEFAULT_PARAMETERS.seconds)
    
    # Pre-allocate matrix
    A = Matrix{Float64}(undef, n, n)
    
    # Initialize matrix once
    init_array!(A)
    
    # Create benchmark with setup that reinitializes the matrix
    b = @benchmarkable begin
        init_array!($A)
        $impl_func($A)
    end samples=samples seconds=seconds evals=1
    
    # Run benchmark
    result = run(b)
    
    return result
end

# Run comprehensive benchmarks
function run_benchmarks(dataset_name; show_memory=true)
    n = DATASET_SIZES[dataset_name]
    
    println("\n" * "="^80)
    println("Dataset: $dataset_name (n=$n)")
    println("Julia version: $(VERSION)")
    println("Number of threads: $(Threads.nthreads())")
    println("Number of workers: $(nworkers())")
    println("="^80)
    
    implementations = Dict(
        "Sequential" => kernel_cholesky_seq!,
        "SIMD" => kernel_cholesky_simd!,
        "Threads" => kernel_cholesky_threads!,
        "BLAS" => kernel_cholesky_blas!,
        "Tiled" => kernel_cholesky_tiled!
    )
    
    # Only add distributed implementation if workers are available
    if nworkers() > 1
        implementations["Distributed"] = kernel_cholesky_distributed!
    end
    
    results = Dict{String, Any}()
    
    println("\nImplementation        | Min Time (ms) | Median (ms) | Mean (ms) | Memory (MB) | Allocs")
    println("-"^80)
    
    for (name, func) in implementations
        trial = benchmark_cholesky(n, func, name)
        results[name] = trial
        
        min_time = minimum(trial).time / 1e6  # ns to ms
        median_time = median(trial).time / 1e6
        mean_time = mean(trial).time / 1e6
        memory_mb = trial.memory / 1024^2
        allocs = trial.allocs
        
        @printf("%-20s | %13.3f | %11.3f | %9.3f | %11.2f | %6d\n", 
                name, min_time, median_time, mean_time, memory_mb, allocs)
    end
    
    # Calculate speedups relative to sequential
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

# Verify correctness of implementations
function verify_implementations(dataset="SMALL"; tolerance=1e-10)
    n = DATASET_SIZES[dataset]
    
    println("Verifying implementation correctness...")
    println("Dataset: $dataset (n=$n)")
    println("-"^50)
    
    # Generate reference solution using BLAS
    A_ref = Matrix{Float64}(undef, n, n)
    init_array!(A_ref)
    A_ref_copy = copy(A_ref)
    kernel_cholesky_blas!(A_ref_copy)
    
    implementations = Dict(
        "Sequential" => kernel_cholesky_seq!,
        "SIMD" => kernel_cholesky_simd!,
        "Threads" => kernel_cholesky_threads!,
        "Tiled" => kernel_cholesky_tiled!,
    )
    
    if nworkers() > 1
        implementations["Distributed"] = kernel_cholesky_distributed!
    end
    
    println("Implementation        | Max Absolute Error | Status")
    println("-"^50)
    
    all_passed = true
    for (name, func) in implementations
        A_test = copy(A_ref)
        func(A_test)
        
        max_error = maximum(abs.(A_ref_copy - A_test))
        status = max_error < tolerance ? "PASS" : "FAIL"
        
        @printf("%-20s | %17.2e | %s\n", name, max_error, status)
        
        if status == "FAIL"
            all_passed = false
        end
    end
    
    if all_passed
        println("\n✅ implementations passed verification!")
    else
        println("\n❌Some implementations failed verification!")
    end
    
    return all_passed
end

# Performance analysis function
function performance_analysis(datasets=["SMALL", "MEDIUM", "LARGE"]; detailed=true)
    println("\n" * "="^80)
    println("CHOLESKY DECOMPOSITION PERFORMANCE ANALYSIS")
    println("="^80)
    
    # System information
    println("System Configuration:")
    println("  Julia version: $(VERSION)")
    println("  Threads: $(Threads.nthreads())")
    println("  Workers: $(nworkers())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  CPU threads: $(Sys.CPU_THREADS)")
    
    all_results = Dict()
    
    for dataset in datasets
        println("\n" * "-"^60)
        println("Analyzing dataset: $dataset")
        println("-"^60)
        
        results = run_benchmarks(dataset, show_memory=detailed)
        all_results[dataset] = results
        
        if detailed
            # Algorithm complexity analysis
            n = DATASET_SIZES[dataset]
            theoretical_ops = n^3 / 3  # Cholesky is O(n³/3)
            
            println("\nAlgorithmic Analysis:")
            println("  Matrix size: $n × $n")
            println("  Theoretical operations: $(round(theoretical_ops / 1e6, digits=2)) MFLOPs")
            
            if haskey(results, "Sequential")
                seq_time = minimum(results["Sequential"]).time / 1e9  # Convert to seconds
                gflops = theoretical_ops / seq_time / 1e9
                println("  Sequential performance: $(round(gflops, digits=2)) GFLOP/s")
            end
            
            # Memory analysis
            memory_required = n * n * 8 / 1024^2  # MB for Float64
            println("  Memory required: $(round(memory_required, digits=2)) MB")
            
            # Column-major access pattern analysis
            println("\nColumn-Major Layout Analysis:")
            println("  ⚠️  Cholesky algorithm challenges for Julia:")
            println("     - Inner loops access A[i,k] and A[j,k] (row-wise)")
            println("     - Row-wise access is cache-unfriendly in column-major layout")
            println("     - Cache miss ratio higher than column-wise algorithms")
            println("     - Tiling helps but cannot completely solve the issue")
            println("     - BLAS implementations use specialized optimizations")
            
            # Estimate cache performance
            cache_line_size = 64  # bytes, typical
            elements_per_line = cache_line_size ÷ 8  # Float64 = 8 bytes
            println("  Cache analysis:")
            println("     - Elements per cache line: $elements_per_line")
            println("     - Row access stride: $(n) elements ($(n*8) bytes)")
            println("     - Cache efficiency: $(round(100*elements_per_line/n, digits=1))% per cache line")
        end
    end
    
    return all_results
end

# Tile size optimization
function optimize_tile_size(dataset="MEDIUM"; tile_sizes=[32, 64, 128, 256, 512])
    println("\n" * "="^80)
    println("TILE SIZE OPTIMIZATION FOR CHOLESKY")
    println("="^80)
    
    n = DATASET_SIZES[dataset]
    
    println("Dataset: $dataset (n=$n)")
    println("Tile Size | Time (ms) | Relative Performance")
    println("----------|-----------|--------------------")
    
    best_time = Inf
    best_tile = 0
    
    for tile_size in tile_sizes
        if tile_size > n
            continue  # Skip if tile size is larger than matrix
        end
        
        # Run multiple trials for accuracy
        times = Float64[]
        for _ in 1:5
            A = Matrix{Float64}(undef, n, n)
            init_array!(A)
            
            time_ns = @elapsed kernel_cholesky_tiled!(A; tile_size=tile_size)
            push!(times, time_ns * 1000)  # Convert to ms
        end
        
        time_ms = minimum(times)  # Take best time
        relative_perf = best_time == Inf ? 1.0 : best_time / (time_ms / 1000)
        
        println("$(lpad(tile_size, 9)) | $(lpad(round(time_ms, digits=2), 9)) | $(lpad(round(relative_perf, digits=2), 18))x")
        
        if time_ms < best_time * 1000
            best_time = time_ms / 1000
            best_tile = tile_size
        end
    end
    
    println("\nOptimal tile size: $best_tile")
    println("Best time: $(round(best_time * 1000, digits=2)) ms")
    
    return best_tile
end

# System configuration check
function check_system_configuration()
    println("\n" * "="^80)
    println("CHOLESKY SYSTEM CONFIGURATION CHECK")
    println("="^80)
    
    println("\n📋 Current Configuration:")
    println("  Julia version: $(VERSION)")
    println("  Julia threads: $(Threads.nthreads())")
    println("  Julia processes: $(nprocs())")
    println("  Julia workers: $(nworkers())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  CPU threads: $(Sys.CPU_THREADS)")
    println("  Available memory: $(round(Sys.total_memory() / 1024^3, digits=1)) GB")
    
    # Memory analysis for different datasets
    println("\n💾 MEMORY REQUIREMENTS:")
    println("="^60)
    available_gb = Sys.total_memory() / 1024^3
    
    for (dataset, n) in DATASET_SIZES
        memory_req = n * n * 8 / 1024^3  # GB for Float64
        
        status = memory_req < available_gb * 0.5 ? "✅" : 
                memory_req < available_gb * 0.8 ? "⚠️" : "❌"
        
        println("  $status $dataset ($n×$n): $(round(memory_req, digits=2)) GB")
    end
    
    # Cholesky-specific recommendations
    println("\n🎯 CHOLESKY-SPECIFIC RECOMMENDATIONS:")
    println("="^60)
    println("  1. ⚠️  JULIA COLUMN-MAJOR LAYOUT CONSIDERATIONS:")
    println("     - A[i,j] then A[i+1,j] (same column) = cache-friendly ✅")
    println("     - A[i,j] then A[i,j+1] (same row) = cache-unfriendly ❌")
    println("     - Cholesky inherently does row-wise access A[i,k] - unavoidable")
    println("  2. 🔄 Limited parallelization due to data dependencies")
    println("  3. 🧱 Focus on cache optimization (tiling) over parallelization")
    println("  4. ⚡ BLAS implementation likely fastest for large matrices")
    println("  5. 📦 Consider blocking algorithms with column-major awareness")
    println("  6. ✅ Ensure input matrix is positive definite")
    println("  7. 🎛️  Tile sizes should consider cache line alignment")
    
    # Performance expectations
    println("\n📊 PERFORMANCE EXPECTATIONS (with column-major considerations):")
    println("="^60)
    println("  Sequential:    1.0x   (baseline)")
    println("  SIMD:         1.2-1.5x (modest improvement despite row access)")
    println("  Threads:      1.1-1.3x (limited by dependencies + cache misses)")
    println("  BLAS:         2-10x   (highly optimized for column-major)")
    println("  Tiled:        1.2-2.0x (cache optimization, column-aware blocking)")
    println("  Distributed:  0.8-1.1x (overhead + communication dominates)")
    println("\n  📌 Note: Row-wise access pattern in Cholesky limits cache efficiency")
    println("     compared to column-wise algorithms like matrix multiplication")
    
    return nothing
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM", "LARGE"])
    println("PolyBench Cholesky Decomposition Benchmark")
    println("="^60)
    
    # Configure BLAS threads appropriately
    configure_blas_threads()
    
    if implementation == "all"
        # Verify implementations first
        verify_implementations()
        println()
        
        # Run benchmarks for specified datasets
        for dataset in datasets
            if dataset in keys(DATASET_SIZES)
                run_benchmarks(dataset)
            else
                println("Unknown dataset: $dataset")
            end
        end
    else
        # Run specific implementation
        n = DATASET_SIZES["LARGE"]
        
        println("Matrix size: $n×$n")
        println("Memory required: $(round(n*n*8/1024^3, digits=2)) GB")
        
        impl_funcs = Dict(
            "seq" => kernel_cholesky_seq!,
            "simd" => kernel_cholesky_simd!,
            "threads" => kernel_cholesky_threads!,
            "blas" => kernel_cholesky_blas!,
            "tiled" => kernel_cholesky_tiled!
        )
        
        if distributed && nworkers() > 0
            impl_funcs["distributed"] = kernel_cholesky_distributed!
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            trial = benchmark_cholesky(n, func, implementation)
            
            println("\n===== Performance Benchmark =====")
            min_time = minimum(trial).time / 1e6  # ns to ms
            mean_time = mean(trial).time / 1e6
            median_time = median(trial).time / 1e6
            @printf("%s | %.3f | %.3f | %.3f\n", 
                    implementation, min_time, mean_time, median_time)
            
            return trial
        else
            error("Unknown implementation: $implementation")
        end
    end
end

# Export public functions
export main, verify_implementations, run_benchmarks, performance_analysis,
       optimize_tile_size, check_system_configuration

end # module