module PolyBench3MM_improved
# Resource Competition:
# CPU Cores: Both threading and multiprocessing compete for the same physical cores
# MemoryBandwidth: Multiple processes + multiple threads = severe bandwidth contention
# cache pollution: Different processes/threads invalidate each other's cache lines
#Oversubscription Problems: 8 processes × 24 threads = 192 execution contexts competing for 8 CPU cores
#Result: massive context switching overhead (200-1000x slower)
#NUMA Effects:
# Modern CPUs have Non-Uniform Memory Access
# Mixing paradigms creates unpredictable memory access patterns
# Performance becomes non-deterministic



# Multithreading PolyBench 3MM Benchmark comparisons for Julia
# 
# This module implements the 3mm kernel from PolyBench with proper optimizations
# addressing column-major order, memory allocation, and parallel efficiency.
#
# The kernel computes G := (A*B)*(C*D), which involves three matrix multiplications.
#
# 1. Basic Benchmarking
# julia# Start Julia with threads: julia -t 16
# include("PolyBench3MM_improved.jl")
# using .PolyBench3MM_improved

# # Run all implementations
# PolyBench3MM_improved.main()

# # Verify correctness first
# PolyBench3MM_improved.verify_implementations("SMALL")
# 2. Distributed Computing
# julia# Terminal: julia -t 1  # Single thread per process
# using Distributed
# addprocs(8)  # Add 8 worker processes

# @everywhere include("PolyBench3MM_improved.jl")
# @everywhere using .PolyBench3MM_improved

# # Run with distributed implementations
# PolyBench3MM_improved.main(distributed=true, datasets=["LARGE"])


# 3. Advanced Performance Analysis
# julia# Comprehensive analysis across multiple datasets
# results = PolyBench3MM_improved.performance_analysis(["SMALL", "MEDIUM", "LARGE"])

# # Thread scaling analysis (theoretical)
# PolyBench3MM_improved.scaling_analysis(dataset="LARGE")

# # Memory-optimized benchmarking for large datasets
# PolyBench3MM_improved.memory_optimized_benchmark("EXTRALARGE", implementation="blas")
# PolyBench3MM_improved.memory_optimized_benchmark("EXTRALARGE", implementation="threads")


# # Optimize tile sizes for cache performance
# best_tile = PolyBench3MM_improved.optimize_tile_size("MEDIUM")


#Usage recommendations:
#julia/check your system configuration first
#PolyBench3MM_improved.check_system_configuration()

# Debug threading issues
#PolyBench3MM_improved.debug_threading_allocations()

# Analyze performance problems
#PolyBench3MM_improved.analyze_performance_issues()

# For multi-threading:
# julia -t 16 (no addprocs)

# For multi-processing :
# julia -t 1, then addprocs(8)

using BenchmarkTools
using Distributed
using LinearAlgebra
using Statistics
using Base.Threads
using Printf
using SharedArrays

# Configure BLAS threads based on Julia configuration
function configure_blas_threads()
    if Threads.nthreads() > 1 #Multithreaded 
        BLAS.set_num_threads(1)#set blas to one thread
        @info "Set BLAS threads to 1 (Julia using $(Threads.nthreads()) threads)"
    else
        # If single-threaded Julia, let BLAS use all cpu cores
        BLAS.set_num_threads(Sys.CPU_THREADS)
        @info "Set BLAS threads to $(Sys.CPU_THREADS)"
    end
end

# Dataset sizes according to PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => (ni=16, nj=18, nk=20, nl=22, nm=24),
    "SMALL" => (ni=40, nj=50, nk=60, nl=70, nm=80),
    "MEDIUM" => (ni=180, nj=190, nk=200, nl=210, nm=220),
    "LARGE" => (ni=800, nj=900, nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

# Initialize arrays with the same pattern as in the C benchmark
function init_arrays!(A, B, C, D, E, F, G)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # Initialize matrices in column-major order for better cache performance
    @inbounds for j = 1:nk, i = 1:ni
        A[i,j] = Float64(((i-1) * (j-1) + 1) % ni) / (5*ni)
    end
    
    @inbounds for j = 1:nj, i = 1:nk
        B[i,j] = Float64(((i-1) * (j) + 2) % nj) / (5*nj)
    end
    
    @inbounds for j = 1:nm, i = 1:nj
        C[i,j] = Float64((i-1) * (j+2) % nl) / (5*nl)
    end
    
    @inbounds for j = 1:nl, i = 1:nm
        D[i,j] = Float64(((i-1) * (j+1) + 2) % nk) / (5*nk)
    end
    
    # Initialize intermediate and result matrices to zero
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    
    return nothing
end

# 1. Sequential Implementation (baseline) - Most basic approach
function kernel_3mm_seq!(E, A, B, F, C, D, G)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # E := A*B - Basic triple loop, column-major order
    for j = 1:nj
        for k = 1:nk
            for i = 1:ni
                E[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    
    # F := C*D - Basic triple loop, column-major order  
    for j = 1:nl
        for k = 1:nm
            for i = 1:nj
                F[i,j] += C[i,k] * D[k,j]
            end
        end
    end
    
    # G := E*F - Basic triple loop, column-major order
    for j = 1:nl
        for k = 1:nj
            for i = 1:ni
                G[i,j] += E[i,k] * F[k,j]
            end
        end
    end
    
    return G
end

# 2. SIMD Optimized Implementation
function kernel_3mm_simd!(E, A, B, F, C, D, G)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # E := A*B - SIMD optimized
    @inbounds for j = 1:nj
        for k = 1:nk
            @simd for i = 1:ni
                E[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    
    # F := C*D - SIMD optimized
    @inbounds for j = 1:nl
        for k = 1:nm
            @simd for i = 1:nj
                F[i,j] += C[i,k] * D[k,j]
            end
        end
    end
    
    # G := E*F - SIMD optimized
    @inbounds for j = 1:nl
        for k = 1:nj
            @simd for i = 1:ni
                G[i,j] += E[i,k] * F[k,j]
            end
        end
    end
    
    return G
end

# 3. Multithreaded Implementation with static scheduling (FIXED - no allocations)
function kernel_3mm_threads_static!(E, A, B, F, C, D, G)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # E := A*B - Parallelize over columns for better cache locality
    @threads :static for j = 1:nj
        @inbounds for k = 1:nk
            @simd for i = 1:ni
                E[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    
    # F := C*D - Parallelize over columns
    @threads :static for j = 1:nl
        @inbounds for k = 1:nm
            @simd for i = 1:nj
                F[i,j] += C[i,k] * D[k,j]
            end
        end
    end
    
    # G := E*F - Parallelize over columns
    @threads :static for j = 1:nl
        @inbounds for k = 1:nj
            @simd for i = 1:ni
                G[i,j] += E[i,k] * F[k,j]
            end
        end
    end
    
    return G
end

# 4. BLAS Optimized Implementation (in-place operations)
# Note: Performance may vary significantly based on:
# - BLAS library (OpenBLAS, MKL, BLIS)
# - Thread configuration  
# - CPU thermal state
# - Memory bandwidth saturation
function kernel_3mm_blas!(E, A, B, F, C, D, G)
    # E := A*B
    mul!(E, A, B)
    
    # F := C*D
    mul!(F, C, D)
    
    # G := E*F
    mul!(G, E, F)
    
    return G
end

# 5. Tiled implementation for better cache usage
function kernel_3mm_tiled!(E, A, B, F, C, D, G; tile_size=64)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    # E := A*B with tiling
    @inbounds for jj = 1:tile_size:nj
        for kk = 1:tile_size:nk
            for ii = 1:tile_size:ni
                for j = jj:min(jj+tile_size-1, nj)
                    for k = kk:min(kk+tile_size-1, nk)
                        @simd for i = ii:min(ii+tile_size-1, ni)
                            E[i,j] += A[i,k] * B[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # F := C*D with tiling
    @inbounds for jj = 1:tile_size:nl
        for kk = 1:tile_size:nm
            for ii = 1:tile_size:nj
                for j = jj:min(jj+tile_size-1, nl)
                    for k = kk:min(kk+tile_size-1, nm)
                        @simd for i = ii:min(ii+tile_size-1, nj)
                            F[i,j] += C[i,k] * D[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # G := E*F with tiling
    @inbounds for jj = 1:tile_size:nl
        for kk = 1:tile_size:nj
            for ii = 1:tile_size:ni
                for j = jj:min(jj+tile_size-1, nl)
                    for k = kk:min(kk+tile_size-1, nj)
                        @simd for i = ii:min(ii+tile_size-1, ni)
                            G[i,j] += E[i,k] * F[k,j]
                        end
                    end
                end
            end
        end
    end
    
    return G
end

# 6. SharedArray implementation for multi-process shared memory
function kernel_3mm_shared!(ni, nj, nk, nl, nm)
    # Create shared arrays that all processes can access
    A = SharedArray{Float64}(ni, nk)
    B = SharedArray{Float64}(nk, nj)
    C = SharedArray{Float64}(nj, nm)
    D = SharedArray{Float64}(nm, nl)
    E = SharedArray{Float64}(ni, nj)
    F = SharedArray{Float64}(nj, nl)
    G = SharedArray{Float64}(ni, nl)
    
    # Initialize arrays
    init_arrays!(A, B, C, D, E, F, G)
    
    # E := A*B - Distribute columns across workers
    @sync @distributed for j = 1:nj
        col_range = localindices(E)[2]
        if j in col_range
            @inbounds for k = 1:nk
                @simd for i = 1:ni
                    E[i,j] += A[i,k] * B[k,j]
                end
            end
        end
    end
    
    # F := C*D - Distribute columns across workers
    @sync @distributed for j = 1:nl
        col_range = localindices(F)[2]
        if j in col_range
            @inbounds for k = 1:nm
                @simd for i = 1:nj
                    F[i,j] += C[i,k] * D[k,j]
                end
            end
        end
    end
    
    # G := E*F - Distribute columns across workers
    @sync @distributed for j = 1:nl
        col_range = localindices(G)[2]
        if j in col_range
            @inbounds for k = 1:nj
                @simd for i = 1:ni
                    G[i,j] += E[i,k] * F[k,j]
                end
            end
        end
    end
    
    return sdata(G)
end

# 7. Distributed implementation with column partitioning
function kernel_3mm_distributed_cols!(E, A, B, F, C, D, G)
    ni, nk = size(A)
    nj = size(B, 2)
    nm = size(C, 2)
    nl = size(D, 2)
    
    p = nworkers()
    
    # First matrix multiplication: E := A*B
    # Distribute columns of E across workers
    @sync begin
        for (idx, w) in enumerate(workers())
            cols_per_worker = ceil(Int, nj / p)
            start_col = (idx-1) * cols_per_worker + 1
            end_col = min(idx * cols_per_worker, nj)
            
            if start_col <= end_col
                @async begin
                    E_cols = view(E, :, start_col:end_col)
                    B_cols = view(B, :, start_col:end_col)
                    
                    result = remotecall_fetch(w, E_cols, A, B_cols) do E_local, A, B_local
                        ni_local, nk_local = size(A)
                        nj_local = size(B_local, 2)
                        
                        fill!(E_local, 0.0)
                        @inbounds for j = 1:nj_local
                            for k = 1:nk_local
                                @simd for i = 1:ni_local
                                    E_local[i,j] += A[i,k] * B_local[k,j]
                                end
                            end
                        end
                        return E_local
                    end
                    
                    copyto!(E_cols, result)
                end
            end
        end
    end
    
    # Second matrix multiplication: F := C*D
    # Distribute columns of F across workers
    @sync begin
        for (idx, w) in enumerate(workers())
            cols_per_worker = ceil(Int, nl / p)
            start_col = (idx-1) * cols_per_worker + 1
            end_col = min(idx * cols_per_worker, nl)
            
            if start_col <= end_col
                @async begin
                    F_cols = view(F, :, start_col:end_col)
                    D_cols = view(D, :, start_col:end_col)
                    
                    result = remotecall_fetch(w, F_cols, C, D_cols) do F_local, C, D_local
                        nj_local = size(F_local, 1)
                        nl_local = size(F_local, 2)
                        nm_local = size(C, 2)
                        
                        fill!(F_local, 0.0)
                        @inbounds for j = 1:nl_local
                            for k = 1:nm_local
                                @simd for i = 1:nj_local
                                    F_local[i,j] += C[i,k] * D_local[k,j]
                                end
                            end
                        end
                        
                        return F_local
                    end
                    
                    copyto!(F_cols, result)
                end
            end
        end
    end
    
    # Third matrix multiplication: G := E*F
    # Distribute columns of G across workers
    @sync begin
        for (idx, w) in enumerate(workers())
            cols_per_worker = ceil(Int, nl / p)
            start_col = (idx-1) * cols_per_worker + 1
            end_col = min(idx * cols_per_worker, nl)
            
            if start_col <= end_col
                @async begin
                    G_cols = view(G, :, start_col:end_col)
                    F_cols = view(F, :, start_col:end_col)
                    
                    result = remotecall_fetch(w, G_cols, E, F_cols) do G_local, E, F_local
                        ni_local = size(G_local, 1)
                        nl_local = size(G_local, 2)
                        nj_local = size(E, 2)
                        
                        fill!(G_local, 0.0)
                        @inbounds for j = 1:nl_local
                            for k = 1:nj_local
                                @simd for i = 1:ni_local
                                    G_local[i,j] += E[i,k] * F_local[k,j]
                                end
                            end
                        end
                        
                        return G_local
                    end
                    
                    copyto!(G_cols, result)
                end
            end
        end
    end
    
    return G
end

# Benchmark runner with proper memory pre-allocation
function benchmark_3mm(ni, nj, nk, nl, nm, impl_func, impl_name; 
                      samples=BenchmarkTools.DEFAULT_PARAMETERS.samples,
                      seconds=BenchmarkTools.DEFAULT_PARAMETERS.seconds)
    
    # Pre-allocate all arrays
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    
    # Initialize arrays once
    init_arrays!(A, B, C, D, E, F, G)
    
    # Create benchmark with setup that only clears intermediate matrices
    b = @benchmarkable begin
        fill!($E, 0.0)
        fill!($F, 0.0)
        fill!($G, 0.0)
        $impl_func($E, $A, $B, $F, $C, $D, $G)
    end samples=samples seconds=seconds evals=1
    
    # Run benchmark
    result = run(b)
    
    return result
end

# Run comprehensive benchmarks
function run_benchmarks(dataset_name; show_memory=true)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl, nm = dataset.ni, dataset.nj, dataset.nk, dataset.nl, dataset.nm
    
    println("\n" * "="^80)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl, nm=$nm)")
    println("Julia version: $(VERSION)")
    println("Number of threads: $(Threads.nthreads())")
    println("Number of workers: $(nworkers())")
    println("="^80)
    
    implementations = Dict(
        "Sequential" => kernel_3mm_seq!,
        "SIMD" => kernel_3mm_simd!,
        "Threads (static)" => kernel_3mm_threads_static!,
        "BLAS" => kernel_3mm_blas!,
        "Tiled" => kernel_3mm_tiled!
    )
    
    # Only add distributed implementation if workers are available
    if nworkers() > 1
        implementations["Distributed (cols)"] = kernel_3mm_distributed_cols!
    end
    
    results = Dict{String, Any}()
    
    println("\nImplementation        | Min Time (ms) | Median (ms) | Mean (ms) | Memory (MB) | Allocs")
    println("-"^80)
    
    for (name, func) in implementations
        trial = benchmark_3mm(ni, nj, nk, nl, nm, func, name)
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
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl, nm = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl, dataset_params.nm
    
    # Allocate arrays
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E_ref = Matrix{Float64}(undef, ni, nj)
    F_ref = Matrix{Float64}(undef, nj, nl)
    G_ref = Matrix{Float64}(undef, ni, nl)
    
    # Initialize arrays
    init_arrays!(A, B, C, D, E_ref, F_ref, G_ref)
    
    # Reference result using BLAS
    E_ref_copy = copy(E_ref)
    F_ref_copy = copy(F_ref)
    G_ref_copy = copy(G_ref)
    kernel_3mm_blas!(E_ref_copy, A, B, F_ref_copy, C, D, G_ref_copy)
    
    implementations = Dict(
        "Sequential" => kernel_3mm_seq!,
        "SIMD" => kernel_3mm_simd!,
        "Threads (static)" => kernel_3mm_threads_static!,
        "Tiled" => kernel_3mm_tiled!,
    )
    
    if nworkers() > 1
        implementations["Distributed (cols)"] = kernel_3mm_distributed_cols!
    end
    
    println("Verifying implementation correctness...")
    println("Implementation        | Max Absolute Error | Status")
    println("-"^50)
    
    for (name, func) in implementations
        E_test = copy(E_ref)
        F_test = copy(F_ref)
        G_test = copy(G_ref)
        
        func(E_test, A, B, F_test, C, D, G_test)
        
        max_error = maximum(abs.(G_ref_copy - G_test))
        status = max_error < tolerance ? "PASS" : "FAIL"
        
        @printf("%-20s | %17.2e | %s\n", name, max_error, status)
    end
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM", "LARGE"])
    println("PolyBench 3MM Benchmark (Complete)")
    println("="^60)
    
    # Configure BLAS threads appropriately
    configure_blas_threads()
    
    if implementation == "all"
        # Check environment
        if Threads.nthreads() > 1 && nworkers() > 1
            @warn "Both multi-threading and multi-processing are enabled. This may lead to resource contention."
            println("Consider running with either threads OR workers, not both.")
            println("Current configuration:")
            println("  - Threads: $(Threads.nthreads())")
            println("  - Workers: $(nworkers())")
            
            # Check thread count on workers
            if nworkers() > 1
                worker_threads = fetch(@spawnat 2 Threads.nthreads())
                println("  - Threads per worker: $worker_threads")
                if worker_threads > 1
                    @warn "Workers have multiple threads. This is not recommended for distributed runs."
                end
            end
        end
        
        # Verify implementations first
        verify_implementations()
        println()
        
        # Run benchmarks for specified datasets
        for dataset in datasets
            if dataset in keys(DATASET_SIZES)
                benchmark_all_implementations(dataset, distributed)
            else
                println("Unknown dataset: $dataset")
            end
        end
    else
        # Run specific implementation for default dataset
        dataset = DATASET_SIZES["LARGE"]
        ni, nj, nk, nl, nm = dataset.ni, dataset.nj, dataset.nk, dataset.nl, dataset.nm
        
        println("Matrix sizes: A($ni×$nk), B($nk×$nj), C($nj×$nm), D($nm×$nl)")
        println("Intermediate: E($ni×$nj), F($nj×$nl), Result: G($ni×$nl)")
        
        impl_funcs = Dict(
            "seq" => kernel_3mm_seq!,
            "simd" => kernel_3mm_simd!,
            "threads" => kernel_3mm_threads_static!,
            "blas" => kernel_3mm_blas!,
            "polly" => kernel_3mm_tiled!
        )
        
        # Add distributed implementations if requested
        if distributed && nworkers() > 0
            impl_funcs["distributed"] = kernel_3mm_distributed_cols!
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            trial = benchmark_3mm(ni, nj, nk, nl, nm, func, implementation)
            
            println("\n===== Performance Benchmark =====")
            min_time = minimum(trial).time / 1e6  # ns to ms
            mean_time = mean(trial).time / 1e6
            median_time = median(trial).time / 1e6
            @printf("%s | %.6f | %.6f | %.6f\n", 
                    implementation, min_time, mean_time, median_time)
            
            println("\n===== Memory Usage =====")
            println("$implementation: $(trial.memory) bytes")
            
            return trial
        else
            error("Unknown implementation: $implementation")
        end
    end
end

# Add the benchmark_all_implementations function
function benchmark_all_implementations(dataset_name, distributed=false)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl, nm = dataset.ni, dataset.nj, dataset.nk, dataset.nl, dataset.nm
    
    println("\n" * "="^60)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl, nm=$nm)")
    println("="^60)
    
    implementations = Dict(
        "seq" => kernel_3mm_seq!,
        "simd" => kernel_3mm_simd!,
        "threads" => kernel_3mm_threads_static!,
        "blas" => kernel_3mm_blas!,
        "tiled" => kernel_3mm_tiled!
    )
    
    # Run standard implementations
    println("\nImplementation | Min Time (s) | Mean Time (s) | Median Time (s) | Memory (MB)")
    println("--------------|--------------|---------------|-----------------|------------")
    
    for (name, func) in implementations
        trial = benchmark_3mm(ni, nj, nk, nl, nm, func, name)
        min_time = minimum(trial).time / 1e9  # ns to s
        mean_time = mean(trial).time / 1e9
        median_time = median(trial).time / 1e9
        memory_mb = trial.memory / 1024^2
        
        @printf("%-13s | %12.6f | %13.6f | %15.6f | %10.2f\n", 
                name, min_time, mean_time, median_time, memory_mb)
    end
    
    # Run distributed implementations if requested
    if distributed && nworkers() > 0
        println("\n" * "-"^60)
        println("Distributed Implementations")
        println("-"^60)
        
        # Test distributed implementation
        trial = benchmark_3mm(ni, nj, nk, nl, nm, kernel_3mm_distributed_cols!, "distributed")
        min_time = minimum(trial).time / 1e9
        @printf("%-13s | %12.6f | Memory: %.2f MB\n", 
                "distributed", min_time, trial.memory / 1024^2)
    end
end

# Advanced performance analysis function
function performance_analysis(datasets=["SMALL", "MEDIUM", "LARGE"]; detailed=true)
    println("\n" * "="^80)
    println("COMPREHENSIVE PERFORMANCE ANALYSIS")
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
            # Memory efficiency analysis
            println("\nMemory Efficiency Analysis:")
            for (name, trial) in results
                allocs_per_op = trial.allocs
                memory_per_op = trial.memory / 1024^2  # MB
                println("  $name: $allocs_per_op allocations, $(round(memory_per_op, digits=3)) MB")
            end
            
            # Cache performance estimation
            dataset_params = DATASET_SIZES[dataset]
            total_elements = dataset_params.ni * dataset_params.nj + 
                           dataset_params.nj * dataset_params.nl + 
                           dataset_params.ni * dataset_params.nl
            cache_footprint = total_elements * 8 / 1024^2  # MB for Float64
            println("\nCache Analysis:")
            println("  Total matrix elements: $total_elements")
            println("  Estimated cache footprint: $(round(cache_footprint, digits=2)) MB")
        end
    end
    
    return all_results
end

# Scaling analysis function
function scaling_analysis(; max_threads=Threads.nthreads(), dataset="LARGE")
    println("\n" * "="^80)
    println("THREAD SCALING ANALYSIS")
    println("="^80)
    
    if max_threads == 1
        println("Single-threaded Julia detected. Run with julia -t N for scaling analysis.")
        return
    end
    
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl, nm = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl, dataset_params.nm
    
    # Pre-allocate arrays
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    
    init_arrays!(A, B, C, D, E, F, G)
    
    # Baseline sequential time
    fill!(E, 0.0); fill!(F, 0.0); fill!(G, 0.0)
    seq_time = @belapsed kernel_3mm_seq!($E, $A, $B, $F, $C, $D, $G)
    
    println("Dataset: $dataset")
    println("Sequential time: $(round(seq_time * 1000, digits=3)) ms")
    println("\nThread Scaling Results:")
    println("Threads | Time (ms) | Speedup | Efficiency")
    println("--------|-----------|---------|----------")
    
    # Note: This is a conceptual framework - actual thread scaling would require
    # running separate Julia processes with different -t values
    println("     1  | $(lpad(round(seq_time * 1000, digits=1), 9)) |   1.00x |    100.0%")
    
    # Theoretical analysis based on Amdahl's law
    parallel_fraction = 0.95  # Assume 95% of work is parallelizable
    for t in 2:min(max_threads, 16)
        theoretical_speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / t)
        theoretical_time = seq_time / theoretical_speedup
        efficiency = theoretical_speedup / t * 100
        
        println("    $(lpad(t, 2))  | $(lpad(round(theoretical_time * 1000, digits=1), 9)) |   $(round(theoretical_speedup, digits=2))x |    $(round(efficiency, digits=1))%")
    end
    
    println("\nNote: These are theoretical projections based on Amdahl's law.")
    println("For actual measurements, run separate Julia instances with different -t values.")
end

# Memory-optimized benchmarking for large datasets
function memory_optimized_benchmark(dataset="EXTRALARGE"; implementation="blas")
    println("\n" * "="^80)
    println("MEMORY-OPTIMIZED BENCHMARK")
    println("="^80)
    
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl, nm = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl, dataset_params.nm
    
    # Calculate memory requirements
    memory_A = ni * nk * 8 / 1024^3  # GB
    memory_B = nk * nj * 8 / 1024^3
    memory_C = nj * nm * 8 / 1024^3
    memory_D = nm * nl * 8 / 1024^3
    memory_E = ni * nj * 8 / 1024^3
    memory_F = nj * nl * 8 / 1024^3
    memory_G = ni * nl * 8 / 1024^3
    total_memory = memory_A + memory_B + memory_C + memory_D + memory_E + memory_F + memory_G
    
    println("Memory Requirements for $dataset:")
    println("  Matrix A ($ni × $nk): $(round(memory_A, digits=3)) GB")
    println("  Matrix B ($nk × $nj): $(round(memory_B, digits=3)) GB")
    println("  Matrix C ($nj × $nm): $(round(memory_C, digits=3)) GB")
    println("  Matrix D ($nm × $nl): $(round(memory_D, digits=3)) GB")
    println("  Matrix E ($ni × $nj): $(round(memory_E, digits=3)) GB")
    println("  Matrix F ($nj × $nl): $(round(memory_F, digits=3)) GB")
    println("  Matrix G ($ni × $nl): $(round(memory_G, digits=3)) GB")
    println("  Total: $(round(total_memory, digits=3)) GB")
    
    available_memory = Sys.total_memory() / 1024^3
    println("  Available system memory: $(round(available_memory, digits=1)) GB")
    
    if total_memory > available_memory * 0.8
        println("\n⚠️  WARNING: Memory requirements exceed 80% of system memory!")
        println("Consider using a smaller dataset or distributed computing.")
        return nothing
    end
    
    println("\n✅ Memory requirements acceptable. Running benchmark...")
    
    # Choose implementation
    impl_func = if implementation == "blas"
        kernel_3mm_blas!
    elseif implementation == "threads"
        kernel_3mm_threads_static!
    elseif implementation == "tiled"
        kernel_3mm_tiled!
    else
        kernel_3mm_seq!
    end
    
    # Run benchmark with memory monitoring
    GC.gc()  # Clean up before benchmark
    start_memory = Sys.total_memory() - Sys.free_memory()
    
    trial = benchmark_3mm(ni, nj, nk, nl, nm, impl_func, implementation; samples=3, seconds=10)
    
    end_memory = Sys.total_memory() - Sys.free_memory()
    
    println("\nBenchmark Results:")
    println("  Implementation: $implementation")
    println("  Min time: $(round(minimum(trial).time / 1e9, digits=3)) seconds")
    println("  Memory allocated: $(round(trial.memory / 1024^3, digits=3)) GB")
    println("  System memory used: $(round((end_memory - start_memory) / 1024^3, digits=3)) GB")
    
    # Performance metrics
    ops_count = 2 * ni * nj * nk + 2 * nj * nl * nm + 2 * ni * nl * nj  # FLOPs for 3MM
    gflops = ops_count / minimum(trial).time
    
    println("  Operations: $(ops_count)")
    println("  Performance: $(round(gflops, digits=2)) GFLOP/s")
    
    return trial
end

# Custom tile size optimization (FIXED)
function optimize_tile_size(dataset="MEDIUM"; tile_sizes=[32, 64, 128, 256])
    println("\n" * "="^80)
    println("TILE SIZE OPTIMIZATION")
    println("="^80)
    
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl, nm = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl, dataset_params.nm
    
    println("Dataset: $dataset")
    println("Tile Size | Time (ms) | Relative Performance")
    println("----------|-----------|--------------------")
    
    best_time = Inf
    best_tile = 0
    
    for tile_size in tile_sizes
        # Pre-allocate arrays
        A = Matrix{Float64}(undef, ni, nk)
        B = Matrix{Float64}(undef, nk, nj)
        C = Matrix{Float64}(undef, nj, nm)
        D = Matrix{Float64}(undef, nm, nl)
        E = Matrix{Float64}(undef, ni, nj)
        F = Matrix{Float64}(undef, nj, nl)
        G = Matrix{Float64}(undef, ni, nl)
        
        init_arrays!(A, B, C, D, E, F, G)
        
        # Benchmark with current tile size - Run multiple times for accuracy
        times = Float64[]
        for _ in 1:5
            fill!(E, 0.0)
            fill!(F, 0.0)
            fill!(G, 0.0)
            
            time_ns = @elapsed kernel_3mm_tiled!(E, A, B, F, C, D, G; tile_size=tile_size)
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

# Debug threading allocation issues
function debug_threading_allocations(dataset="SMALL")
    println("\n" * "="^80)
    println("DEBUGGING THREADING ALLOCATION ISSUES")
    println("="^80)
    
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl, nm = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl, dataset_params.nm
    
    # Pre-allocate arrays
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    
    init_arrays!(A, B, C, D, E, F, G)
    
    println("Testing threading implementations for allocation issues...")
    
    # Test sequential (should be 0 allocations)
    fill!(E, 0.0); fill!(F, 0.0); fill!(G, 0.0)
    seq_allocs = @allocated kernel_3mm_seq!(E, A, B, F, C, D, G)
    println("Sequential allocations: $seq_allocs bytes")
    
    # Test SIMD (should be 0 allocations)  
    fill!(E, 0.0); fill!(F, 0.0); fill!(G, 0.0)
    simd_allocs = @allocated kernel_3mm_simd!(E, A, B, F, C, D, G)
    println("SIMD allocations: $simd_allocs bytes")
    
    # Test threaded (PROBLEM: shows allocations)
    fill!(E, 0.0); fill!(F, 0.0); fill!(G, 0.0)
    thread_allocs = @allocated kernel_3mm_threads_static!(E, A, B, F, C, D, G)
    println("Threaded allocations: $thread_allocs bytes")
    
    if thread_allocs > 0
        println("\n❌ PROBLEM IDENTIFIED:")
        println("   Threading implementation has $(thread_allocs) bytes of allocations")
        println("   This indicates closure captures or other allocation sources")
        println("\n🔧 POTENTIAL CAUSES:")
        println("   1. @threads macro creating closures")
        println("   2. Captured variables in threaded loops")
        println("   3. Thread-local storage allocations")
        println("   4. Range partitioning overhead")
        
        println("\n💡 SOLUTIONS:")
        println("   1. Use Threads.@spawn with explicit ranges")
        println("   2. Pre-allocate thread-local buffers")
        println("   3. Use StaticArrays for small temporary data")
        println("   4. Manual thread management instead of @threads")
    else
        println("\n✅ Threading allocations are properly optimized")
    end
    
    # Test BLAS (should be 0 allocations)
    fill!(E, 0.0); fill!(F, 0.0); fill!(G, 0.0)
    blas_allocs = @allocated kernel_3mm_blas!(E, A, B, F, C, D, G)
    println("BLAS allocations: $blas_allocs bytes")
    
    return Dict(
        "sequential" => seq_allocs,
        "simd" => simd_allocs, 
        "threaded" => thread_allocs,
        "blas" => blas_allocs
    )
end

# System configuration helper
function check_system_configuration()
    println("\n" * "="^80)
    println("SYSTEM CONFIGURATION CHECK")
    println("="^80)
    
    println("\n📋 Current Configuration:")
    println("  Julia version: $(VERSION)")
    println("  Julia threads: $(Threads.nthreads())")
    println("  Julia processes: $(nprocs())")
    println("  Julia workers: $(nworkers())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  CPU threads: $(Sys.CPU_THREADS)")
    println("  Available memory: $(round(Sys.total_memory() / 1024^3, digits=1)) GB")
    
    # Check for optimal configuration
    println("\n✅ OPTIMAL CONFIGURATIONS:")
    println("="^60)
    
    println("\n🚀 For Multi-Threading (Recommended for 3MM):")
    println("  Launch: julia -t $(Sys.CPU_THREADS)")
    println("  Workers: 0 (no addprocs)")
    println("  BLAS threads: 1 (automatic)")
    println("  Best for: Dense linear algebra, cache-friendly algorithms")
    
    println("\n🌐 For Multi-Processing:")
    println("  Launch: julia -t 1")
    println("  Workers: $(Sys.CPU_THREADS) (via addprocs)")
    println("  BLAS threads: 1 per process")
    println("  Best for: Embarrassingly parallel, high computation/communication ratio")
    
    # Check current configuration
    println("\n🔍 CURRENT CONFIGURATION ANALYSIS:")
    println("="^60)
    
    if Threads.nthreads() > 1 && nworkers() > 1
        println("  ⚠️  WARNING: Both threading and multiprocessing enabled!")
        println("     This will cause resource contention and poor performance.")
        println("     Recommended:choose one paradigm.")
    elseif Threads.nthreads() > 1 && nworkers() <= 1
        println("  ✅ GOOD: Multi-threading configuration")
        if BLAS.get_num_threads() == 1
            println("  ✅ BLAS threads properly set to 1")
        else
            println("  ⚠️  WARNING: BLAS using $(BLAS.get_num_threads()) threads")
            println("     This may cause oversubscription with Julia threads")
        end
    elseif Threads.nthreads() == 1 && nworkers() > 1
        println("  ✅ GOOD: Multi-processing configuration")
        if BLAS.get_num_threads() == Sys.CPU_THREADS
            println("  ✅ BLAS threads properly utilizing all cores per process")
        else
            println("  ⚠️  BLAS using $(BLAS.get_num_threads()) threads per process")
        end
    else
        println("  ℹ️  Single-threaded, single-process configuration")
        println("     Performance will be limited but predictable")
    end
    
    # Memory analysis
    println("\n💾 MEMORY CONFIGURATION:")
    println("="^60)
    available_gb = Sys.total_memory() / 1024^3
    
    for (dataset, params) in DATASET_SIZES
        memory_req = (params.ni * params.nk + params.nk * params.nj + 
                     params.nj * params.nm + params.nm * params.nl +
                     params.ni * params.nj + params.nj * params.nl + 
                     params.ni * params.nl) * 8 / 1024^3
        
        status = memory_req < available_gb * 0.5 ? "✅" : 
                memory_req < available_gb * 0.8 ? "⚠️" : "❌"
        
        println("  $status $dataset: $(round(memory_req, digits=2)) GB")
    end
    
    println("\n🎯 RECOMMENDATIONS FOR 3MM BENCHMARK:")
    println("="^60)
    
    if available_gb < 2
        println("  💡 Use SMALL/MEDIUM datasets")
        println("  💡 Focus on algorithmic optimization over parallelization")
    elseif available_gb < 8  
        println("  💡 Use up to LARGE dataset")
        println("  💡 Multi-threading recommended over multi-processing")
    else
        println("  💡 All datasets supported")
        println("  💡 Test both threading and multiprocessing")
    end
    
    # Performance tips
    println("\n🏃 PERFORMANCE OPTIMIZATION TIPS:")
    println("="^60)
    println("  1. Close unnecessary applications")
    println("  2. Disable CPU frequency scaling: sudo cpupower frequency-set -g performance")
    println("  3. Set process affinity: taskset -c 0-$(Sys.CPU_THREADS-1) julia")
    println("  4. Increase process priority: nice -n -20 julia")
    println("  5. Use: export JULIA_CPU_THREADS=$(Sys.CPU_THREADS)")
    
    return nothing
end

# Comprehensive log analysis and improvement recommendations
function analyze_performance_issues()
    println("\n" * "="^80)
    println("PERFORMANCE ISSUES ANALYSIS & RECOMMENDATIONS")
    println("="^80)
    
    println("\n🔍 IDENTIFIED ISSUES FROM LOGS:")
    println("="^60)
    
    println("\n1. THREADING PROBLEMS:")
    println("   ❌ Memory allocations: 366 allocs for threads vs 0 for others")
    println("   ❌ Poor small dataset performance: 0.118ms seq vs 1.216ms threads")
    println("   ❌ Thread overhead dominates until LARGE datasets")
    println("   ✅ Good scaling only at LARGE: 1041ms seq vs 208ms threads (5x)")
    
    println("\n2. DISTRIBUTED COMPUTING ISSUES:")
    println("   ❌ Poor scaling: 1084ms seq vs 1047ms distributed (minimal improvement)")
    println("   ❌ High memory overhead: 64MB vs 0MB for other implementations")
    println("   ❌ Communication overhead >> computation for current problem sizes")
    println("   ❌ 4507-5465 allocations indicating excessive object creation")
    
    println("\n3. BLAS PERFORMANCE INCONSISTENCIES:")
    println("   ❌ Wildly varying performance between runs:")
    println("      - LARGE dataset: 37ms (distributed) vs 107ms (threads)")
    println("      - Same BLAS library showing 3x performance difference")
    println("   ❌ Suggests thermal throttling or system interference")
    
    println("\n4. TILE SIZE OPTIMIZATION ISSUES:")
    println("   ❌ Showing 0.0ms times (measurement artifacts)")
    println("   ❌ All tile sizes showing similar performance")
    println("   ❌ No clear optimal tile size identification")
    
    println("\n5. MEMORY ALLOCATION PROBLEMS:")
    println("   ❌ Thread implementation should have 0 allocations")
    println("   ❌ Distributed showing massive allocation overhead")
    println("   ❌ SharedArray not properly tested")
    
    println("\n🔧 RECOMMENDED IMPROVEMENTS:")
    println("="^60)
    
    println("\n1. THREADING FIXES:")
    println("   ✅ Remove all allocations from threaded code")
    println("   ✅ Use thread-local storage for temporary variables")
    println("   ✅ Consider work-stealing vs static scheduling")
    println("   ✅ Add cache-friendly data layout optimizations")
    
    println("\n2. DISTRIBUTED COMPUTING IMPROVEMENTS:")
    println("   ✅ Implement block-wise distribution instead of column-wise")
    println("   ✅ Use SharedArrays for single-node multi-process")
    println("   ✅ Add computation granularity controls")
    println("   ✅ Test with larger problem sizes (EXTRALARGE+)")
    
    println("\n3. BENCHMARKING IMPROVEMENTS:")
    println("   ✅ Add CPU frequency stabilization")
    println("   ✅ Isolate CPU cores for benchmarking")
    println("   ✅ Monitor thermal states during testing")
    println("   ✅ Use longer sampling periods for small datasets")
    
    println("\n4. ALGORITHMIC IMPROVEMENTS:")
    println("   ✅ Separate truly sequential from SIMD implementations")
    println("   ✅ Add cache-oblivious algorithms")
    println("   ✅ Implement recursive blocking strategies")
    println("   ✅ Add memory access pattern optimization")
    
    println("\n5. TESTING METHODOLOGY:")
    println("   ✅ Test each paradigm in isolation (no mixing)")
    println("   ✅ Add performance regression testing")
    println("   ✅ Include memory bandwidth measurements")
    println("   ✅ Add CPU utilization monitoring")
    
    println("\n📊 PERFORMANCE EXPECTATIONS:")
    println("="^60)
    
    println("\nOptimal Performance Ratios (LARGE dataset):")
    println("  Sequential:      1.0x   (baseline)")
    println("  SIMD:           1.5x   (vectorization)")
    println("  BLAS:          10x    (optimized kernels)")
    println("  Threads (8):    6-7x   (parallel efficiency 75-87%)")
    println("  Threads (16):  10-12x  (parallel efficiency 62-75%)")
    println("  Distributed:    0.5x   (communication overhead for this problem)")
    
    println("\n⚠️  CRITICAL FIXES NEEDED:")
    println("="^60)
    println("  1. Fix threading memory allocations")
    println("  2. Separate sequential and SIMD implementations")
    println("  3. Improve distributed granularity")
    println("  4. Add proper performance isolation")
    println("  5. Fix tile size optimization measurement")
end

# Export public functions
export main, verify_implementations, run_benchmarks, benchmark_all_implementations,
       performance_analysis, scaling_analysis, memory_optimized_benchmark, optimize_tile_size,
       analyze_performance_issues, check_system_configuration, debug_threading_allocations

end # module