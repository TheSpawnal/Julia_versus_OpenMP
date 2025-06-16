module PolyBench2MM_Improved
# Improved PolyBench 2MM Benchmark suite for Julia
# 
# This module implements the 2mm kernel from PolyBench with proper optimizations
# addressing column-major order, memory allocation, and parallel efficiency.
#
# The kernel computes D := alpha*A*B*C + beta*D, which involves two
# matrix multiplications.
#
# USAGE:
#   1. Include this file: include("PolyBench2MM_Improved.jl")
#   2. Import the module: using .PolyBench2MM_Improved
#
# For distributed implementations:
#   1. First add worker processes: using Distributed; addprocs(4)
#   2. Then include this file: @everywhere include("PolyBench2MM_Improved.jl")
#   3. Import on all workers: @everywhere using .PolyBench2MM_Improved
#   4. Now run benchmarks: PolyBench2MM_Improved.main(distributed=true)

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
        # If single-threaded Julia, let BLAS use multiple threads
        BLAS.set_num_threads(Sys.CPU_THREADS)
        @info "Set BLAS threads to $(Sys.CPU_THREADS)"
    end
end

# Dataset sizes according to PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => (ni=16, nj=18, nk=22, nl=24),
    "SMALL" => (ni=40, nj=50, nk=70, nl=80),
    "MEDIUM" => (ni=180, nj=190, nk=210, nl=220),
    "LARGE" => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# Initialize arrays with the same pattern as in the C benchmark
function init_arrays!(alpha::Ref{Float32}, beta::Ref{Float32}, tmp, A, B, C, D)
    ni, nk = size(A)
    nj = size(B, 2)
    nl = size(C, 2)
    
    alpha[] = 1.5f0
    beta[] = 1.2f0
    
    # Initialize matrices in column-major order for better cache performance
    @inbounds for j = 1:nk, i = 1:ni
        A[i,j] = Float32(((i-1) * (j-1) + 1) % ni) / ni
    end
    
    @inbounds for j = 1:nj, i = 1:nk
        B[i,j] = Float32((i-1) * j % nj) / nj
    end
    
    @inbounds for j = 1:nl, i = 1:nj
        C[i,j] = Float32(((i-1) * (j+2) + 1) % nl) / nl
    end
    
    @inbounds for j = 1:nl, i = 1:ni
        D[i,j] = Float32((i-1) * (j+1) % nk) / nk
    end
    
    fill!(tmp, 0.0f0)
    
    return nothing
end

# 1. Sequential Implementation (baseline) - Column-major optimized
function kernel_2mm_seq!(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    # Column-major order: iterate over columns of B first
    @inbounds for j = 1:nj
        for k = 1:nk
            @simd for i = 1:ni
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    # Column-major order: iterate over columns of C first
    @inbounds for j = 1:nl
        for i = 1:ni
            D[i,j] *= beta
        end
        for k = 1:nj
            @simd for i = 1:ni
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 2. Multithreaded Implementation with static scheduling
function kernel_2mm_threads_static!(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    # Parallelize over columns for better cache locality
    @threads :static for j = 1:nj
        @inbounds for k = 1:nk
            @simd for i = 1:ni
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @threads :static for j = 1:nl
        @inbounds for i = 1:ni
            D[i,j] *= beta
        end
        @inbounds for k = 1:nj
            @simd for i = 1:ni
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 3. BLAS Optimized Implementation (in-place operations)
function kernel_2mm_blas!(alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0f0)  # tmp = alpha * A * B + 0.0 * tmp
    
    # Second matrix multiplication: D = beta * D + tmp * C
    mul!(D, tmp, C, 1.0f0, beta)  # D = 1.0 * tmp * C + beta * D
    
    return D
end

# 4. Tiled implementation for better cache usage
function kernel_2mm_tiled!(alpha, beta, tmp, A, B, C, D; tile_size=64)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Clear tmp matrix
    fill!(tmp, 0.0f0)
    
    # First matrix multiplication with tiling: tmp = alpha * A * B
    @inbounds for jj = 1:tile_size:nj
        for kk = 1:tile_size:nk
            for ii = 1:tile_size:ni
                for j = jj:min(jj+tile_size-1, nj)
                    for k = kk:min(kk+tile_size-1, nk)
                        @simd for i = ii:min(ii+tile_size-1, ni)
                            tmp[i,j] += alpha * A[i,k] * B[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # Second matrix multiplication with tiling: D = tmp * C + beta * D
    @inbounds for jj = 1:tile_size:nl
        for ii = 1:tile_size:ni
            # Scale D by beta
            for j = jj:min(jj+tile_size-1, nl)
                @simd for i = ii:min(ii+tile_size-1, ni)
                    D[i,j] *= beta
                end
            end
            # Multiply and accumulate
            for kk = 1:tile_size:nj
                for j = jj:min(jj+tile_size-1, nl)
                    for k = kk:min(kk+tile_size-1, nj)
                        @simd for i = ii:min(ii+tile_size-1, ni)
                            D[i,j] += tmp[i,k] * C[k,j]
                        end
                    end
                end
            end
        end
    end
    
    return D
end

# 5. SharedArray implementation for multi-process shared memory
function kernel_2mm_shared!(alpha, beta, ni, nj, nk, nl)
    # Create shared arrays that all processes can access
    A = SharedArray{Float32}(ni, nk)
    B = SharedArray{Float32}(nk, nj)
    C = SharedArray{Float32}(nj, nl)
    D = SharedArray{Float32}(ni, nl)
    tmp = SharedArray{Float32}(ni, nj)
    
    # Initialize arrays
    alpha_ref = Ref(alpha)
    beta_ref = Ref(beta)
    init_arrays!(alpha_ref, beta_ref, tmp, A, B, C, D)
    
    # First matrix multiplication: tmp = alpha * A * B
    # Distribute columns across workers
    @sync @distributed for j = 1:nj
        col_range = localindices(tmp)[2]
        if j in col_range
            @inbounds for k = 1:nk
                @simd for i = 1:ni
                    tmp[i,j] += alpha * A[i,k] * B[k,j]
                end
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @sync @distributed for j = 1:nl
        col_range = localindices(D)[2]
        if j in col_range
            @inbounds for i = 1:ni
                D[i,j] *= beta
            end
            @inbounds for k = 1:nj
                @simd for i = 1:ni
                    D[i,j] += tmp[i,k] * C[k,j]
                end
            end
        end
    end
    
    return sdata(D)
end

# 6. Distributed implementation with column partitioning
function kernel_2mm_distributed_cols!(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    p = nworkers()
    
    # First matrix multiplication: distribute columns of tmp
    # Each worker computes a subset of columns
    @sync begin
        for (idx, w) in enumerate(workers())
            cols_per_worker = ceil(Int, nj / p)
            start_col = (idx-1) * cols_per_worker + 1
            end_col = min(idx * cols_per_worker, nj)
            
            if start_col <= end_col
                @async begin
                    # Use views to avoid copying
                    tmp_cols = view(tmp, :, start_col:end_col)
                    B_cols = view(B, :, start_col:end_col)
                    
                    result = remotecall_fetch(w, tmp_cols, A, B_cols, alpha) do tmp_local, A, B_local, alpha
                        ni_local, nk_local = size(A)
                        nj_local = size(B_local, 2)
                        
                        # Clear local tmp
                        fill!(tmp_local, 0.0f0)
                        
                        @inbounds for j = 1:nj_local
                            for k = 1:nk_local
                                @simd for i = 1:ni_local
                                    tmp_local[i,j] += alpha * A[i,k] * B_local[k,j]
                                end
                            end
                        end
                        return tmp_local
                    end
                    
                    # Copy result back
                    copyto!(tmp_cols, result)
                end
            end
        end
    end
    
    # Second matrix multiplication: distribute columns of D
    @sync begin
        for (idx, w) in enumerate(workers())
            cols_per_worker = ceil(Int, nl / p)
            start_col = (idx-1) * cols_per_worker + 1
            end_col = min(idx * cols_per_worker, nl)
            
            if start_col <= end_col
                @async begin
                    D_cols = view(D, :, start_col:end_col)
                    C_cols = view(C, :, start_col:end_col)
                    
                    result = remotecall_fetch(w, D_cols, tmp, C_cols, beta) do D_local, tmp, C_local, beta
                        ni_local = size(D_local, 1)
                        nl_local = size(D_local, 2)
                        nj_local = size(tmp, 2)
                        
                        # Scale D by beta
                        @inbounds for j = 1:nl_local
                            @simd for i = 1:ni_local
                                D_local[i,j] *= beta
                            end
                        end
                        
                        # Multiply and accumulate
                        @inbounds for j = 1:nl_local
                            for k = 1:nj_local
                                @simd for i = 1:ni_local
                                    D_local[i,j] += tmp[i,k] * C_local[k,j]
                                end
                            end
                        end
                        
                        return D_local
                    end
                    
                    copyto!(D_cols, result)
                end
            end
        end
    end
    
    return D
end

# Benchmark runner with proper memory pre-allocation
function benchmark_2mm(ni, nj, nk, nl, impl_func, impl_name; 
                      samples=BenchmarkTools.DEFAULT_PARAMETERS.samples,
                      seconds=BenchmarkTools.DEFAULT_PARAMETERS.seconds)
    
    # Pre-allocate all arrays
    A = Matrix{Float32}(undef, ni, nk)
    B = Matrix{Float32}(undef, nk, nj)
    C = Matrix{Float32}(undef, nj, nl)
    D = Matrix{Float32}(undef, ni, nl)
    tmp = Matrix{Float32}(undef, ni, nj)
    D_original = Matrix{Float32}(undef, ni, nl)
    
    alpha = Ref(1.5f0)
    beta = Ref(1.2f0)
    
    # Initialize arrays once
    init_arrays!(alpha, beta, tmp, A, B, C, D_original)
    
    # Create benchmark with setup that only copies D
    b = @benchmarkable begin
        copyto!($D, $D_original)
        fill!($tmp, 0.0f0)
        $impl_func($alpha[], $beta[], $tmp, $A, $B, $C, $D)
    end samples=samples seconds=seconds evals=1
    
    # Run benchmark
    result = run(b)
    
    return result
end

# Run comprehensive benchmarks
function run_benchmarks(dataset_name; show_memory=true)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl = dataset.ni, dataset.nj, dataset.nk, dataset.nl
    
    println("\n" * "="^80)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    println("Julia version: $(VERSION)")
    println("Number of threads: $(Threads.nthreads())")
    println("Number of workers: $(nworkers())")
    println("="^80)
    
    implementations = Dict(
        "Sequential" => kernel_2mm_seq!,
        "Threads (static)" => kernel_2mm_threads_static!,
        "BLAS" => kernel_2mm_blas!,
        "Tiled" => kernel_2mm_tiled!
    )
    
    # Only add distributed implementation if workers are available
    if nworkers() > 1
        implementations["Distributed (cols)"] = kernel_2mm_distributed_cols!
    end
    
    results = Dict{String, Any}()
    
    println("\nImplementation        | Min Time (ms) | Median (ms) | Mean (ms) | Memory (MB) | Allocs")
    println("-"^80)
    
    for (name, func) in implementations
        trial = benchmark_2mm(ni, nj, nk, nl, func, name)
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
function verify_implementations(dataset="SMALL"; tolerance=1e-5)
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl
    
    # Allocate arrays
    A = Matrix{Float32}(undef, ni, nk)
    B = Matrix{Float32}(undef, nk, nj)
    C = Matrix{Float32}(undef, nj, nl)
    D_ref = Matrix{Float32}(undef, ni, nl)
    tmp_ref = Matrix{Float32}(undef, ni, nj)
    
    alpha = Ref(1.5f0)
    beta = Ref(1.2f0)
    
    # Initialize arrays
    init_arrays!(alpha, beta, tmp_ref, A, B, C, D_ref)
    
    # Reference result using BLAS
    D_ref_copy = copy(D_ref)
    tmp_ref_copy = copy(tmp_ref)
    kernel_2mm_blas!(alpha[], beta[], tmp_ref_copy, A, B, C, D_ref_copy)
    
    implementations = Dict(
        "Sequential" => kernel_2mm_seq!,
        "Threads (static)" => kernel_2mm_threads_static!,
        "Tiled" => kernel_2mm_tiled!,
    )
    
    if nworkers() > 1
        implementations["Distributed (cols)"] = kernel_2mm_distributed_cols!
    end
    
    println("Verifying implementation correctness...")
    println("Implementation        | Max Absolute Error | Status")
    println("-"^50)
    
    for (name, func) in implementations
        D_test = copy(D_ref)
        tmp_test = copy(tmp_ref)
        
        func(alpha[], beta[], tmp_test, A, B, C, D_test)
        
        max_error = maximum(abs.(D_ref_copy - D_test))
        status = max_error < tolerance ? "PASS" : "FAIL"
        
        @printf("%-20s | %17.2e | %s\n", name, max_error, status)
    end
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM", "LARGE"])
    println("PolyBench 2MM Benchmark (Improved)")
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
        ni, nj, nk, nl = dataset.ni, dataset.nj, dataset.nk, dataset.nl
        
        println("Matrix sizes: A($ni×$nk), B($nk×$nj), C($nj×$nl), D($ni×$nl)")
        
        impl_funcs = Dict(
            "seq" => kernel_2mm_seq!,
            "simd" => kernel_2mm_seq!,  # Using seq since we have integrated SIMD
            "threads" => kernel_2mm_threads_static!,
            "blas" => kernel_2mm_blas!,
            "polly" => kernel_2mm_tiled!
        )
        
        # Add distributed implementations if requested
        if distributed && nworkers() > 0
            impl_funcs["distributed"] = kernel_2mm_distributed_cols!
            if mod(ni, nworkers()) == 0
                impl_funcs["dist3"] = kernel_2mm_distributed_cols!
            end
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            trial = benchmark_2mm(ni, nj, nk, nl, func, implementation)
            
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

# Add the benchmark_all_implementations function that was referenced but missing
function benchmark_all_implementations(dataset_name, distributed=false)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl = dataset.ni, dataset.nj, dataset.nk, dataset.nl
    
    println("\n" * "="^60)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    println("="^60)
    
    implementations = Dict(
        "seq" => kernel_2mm_seq!,
        "threads" => kernel_2mm_threads_static!,
        "blas" => kernel_2mm_blas!,
        "tiled" => kernel_2mm_tiled!
    )
    
    # Run standard implementations
    println("\nImplementation | Min Time (s) | Mean Time (s) | Median Time (s) | Memory (MB)")
    println("--------------|--------------|---------------|-----------------|------------")
    
    for (name, func) in implementations
        trial = benchmark_2mm(ni, nj, nk, nl, func, name)
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
        trial = benchmark_2mm(ni, nj, nk, nl, kernel_2mm_distributed_cols!, "distributed")
        min_time = minimum(trial).time / 1e9
        @printf("%-13s | %12.6f | Memory: %.2f MB\n", 
                "distributed", min_time, trial.memory / 1024^2)
    end
end

# Export public functions
export main, verify_implementations, run_benchmarks, benchmark_all_implementations

end # module