module PolyBench2MM
# PolyBench 2MM Benchmark suite for Julia
# 
# This module implements the 2mm kernel from PolyBench in various optimized forms
# for Julia. The kernel computes D := alpha*A*B*C + beta*D, which involves two
# matrix multiplications.
#do you want use the memory allocation tracker ? In the cli : 
#$ julia track-allocation=user
# USAGE:
#   1. Include this file: include("PolyBench2MM.jl")
#   2. Import the module: using .PolyBench2MM
#
# For distributed implementations:
#   1. First add worker processes: using Distributed; addprocs(4)
#   2. Then include this file: @everywhere include("PolyBench2MM.jl")
#   3. Import on all workers: @everywhere using .PolyBench2MM
#   4. Now run benchmarks: PolyBench2MM.main(distributed=true)
#
# To run a specific implementation:
#   PolyBench2MM.main(implementation="blas")
#
# Available implementations:
#   - "seq" (Sequential baseline)
#   - "simd" (SIMD-optimized)
#   - "threads" (Multithreaded)
#   - "blas" (BLAS-optimized)
#   - "polly" (LLVM Framework optimized)
#   
# For distributed (set distributed=true):
#   - "distributed" (Distributed computing)
#   - "dist3" (Distributed Algorithm 3 with optimized communication)
#
# To verify correctness:
#   PolyBench2MM.verify_implementations()

using BenchmarkTools
using Distributed
using LinearAlgebra
using Statistics
using Base.Threads
using Printf
using SharedArrays# IMPROVEMENT 1: import SharedArrays for shared memory arrays

# Dataset sizes according to PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => (ni=16, nj=18, nk=22, nl=24),
    "SMALL" => (ni=40, nj=50, nk=70, nl=80),
    "MEDIUM" => (ni=180, nj=190, nk=210, nl=220),
    "LARGE" => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# 1. Sequential Implementation (baseline)
# IMPROVEMENT 2: Fixed column-major memory access pattern for better cache performance
function kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    @inbounds for j = 1:nj # FIXED: Changed loop order from i-j-k to j-i-k for column-major access
        for i = 1:ni
            tmp[i,j] = zero(eltype(tmp))
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta *D
    @inbounds for j = 1:nl# IMPROVEMENT: loop order j-i-k for column-major access
        for i = 1:ni
            D[i,j] *= beta
            for k = 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    return D
end

# 2. SIMD Optimization
# IMPROVEMENT 3: Fixed column-major access and improved SIMD vectorization
function kernel_2mm_simd(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    #first matrix multiplication: tmp = alpha * A * B
    @inbounds for j = 1:nj
        for i = 1:ni
            sum_val = zero(eltype(tmp))
            @simd for k = 1:nk
                sum_val += alpha * A[i,k] * B[k,j]
            end
            tmp[i,j] = sum_val
        end
    end
    
    # 2nd matrix mult: D = tmp * C + beta * D
    @inbounds for j = 1:nl
        for i = 1:ni
            D[i,j] *= beta
            sum_val = zero(eltype(D))
            @simd for k = 1:nj
                sum_val += tmp[i,k] * C[k,j]
            end
            D[i,j] += sum_val
        end
    end
    
    return D
end

# 3. Multithreaded Implementation # IMPROVEMENT 4: Using :static scheduling for better load balancing
function kernel_2mm_threads(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    # 1st matrix mult: tmp = alpha * A * B
    # IMPROVEMENT:Using :static scheduling and column-wise parallelization
    @threads :static for j = 1:nj
        @inbounds for i = 1:ni
            sum_val = zero(eltype(tmp))
            for k = 1:nk
                sum_val += alpha * A[i,k] * B[k,j]
            end
            tmp[i,j] = sum_val
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @threads :static for j = 1:nl
        @inbounds for i = 1:ni
            D[i,j] *= beta
            sum_val = zero(eltype(D))
            for k = 1:nj
                sum_val += tmp[i,k] * C[k,j]
            end
            D[i,j] += sum_val
        end
    end
    
    return D
end

# 4. BLAS Optimized Implementation
function kernel_2mm_blas(alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)  # tmp = alpha * A * B + 0.0 * tmp
    # Scale D by beta
    lmul!(beta, D)
    # 2nd matrix multiplication: D += tmp * C
    mul!(D, tmp, C, 1.0, 1.0)  # D = 1.0 * tmp * C + 1.0 * D
    return D
end

# 5. Polly-style optimization (Loop tiling and cache optimization)/# IMPROVEMENT 5: Fixed tiling for column-major layout
function kernel_2mm_polly(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    # Tile size (tuned for cache efficiency)
    tile_size = 64
    # First matrix multiplication with tiling: tmp = alpha * A * B
    fill!(tmp, zero(eltype(tmp)))# IMPROVEMENT: Changed tiling order for column-major access
    @inbounds for jj = 1:tile_size:nj
        for ii = 1:tile_size:ni
            for kk = 1:tile_size:nk
                for j = jj:min(jj+tile_size-1, nj)
                    for i = ii:min(ii+tile_size-1, ni)
                        sum_val = tmp[i,j]
                        @simd for k = kk:min(kk+tile_size-1, nk)
                            sum_val += alpha * A[i,k] * B[k,j]
                        end
                        tmp[i,j] = sum_val
                    end
                end
            end
        end
    end
    
    # 2nd matrix multiplication with tiling: D = tmp * C + beta * D
    @inbounds for jj = 1:tile_size:nl
        for ii = 1:tile_size:ni
            for j = jj:min(jj+tile_size-1, nl)
                for i = ii:min(ii+tile_size-1, ni)
                    D[i,j] *= beta
                end
            end
            for kk = 1:tile_size:nj
                for j = jj:min(jj+tile_size-1, nl)
                    for i = ii:min(ii+tile_size-1, ni)
                        sum_val = zero(eltype(D))
                        @simd for k = kk:min(kk+tile_size-1, nj)
                            sum_val += tmp[i,k] * C[k,j]
                        end
                        D[i,j] += sum_val
                    end
                end
            end
        end
    end
    
    return D
end

# 6. Distributed Implementation using @distributed macro/ # IMPROVEMENT 6: Fixed to use column-wise distribution
function kernel_2mm_distributed(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    # First matrix multiplication: tmp = alpha * A * B
    # IMPROVEMENT: Column-wise distribution for better cache usage
    @sync @distributed for j = 1:nj
        for i = 1:ni
            tmp[i,j] = zero(eltype(tmp))
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @sync @distributed for j = 1:nl
        for i = 1:ni
            D[i,j] *= beta
            for k = 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 7. Distributed Implementation (Algorithm 3 from matrix_instructions.txt)
function kernel_2mm_dist3(alpha, beta, tmp, A, B, C, D)#IMPROVEMENT 7: Using SharedArrays and views to avoid data copying
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Ensure the number of workers is suitable for the matrix size
    p = nworkers()
    @assert mod(nj, p) == 0 "Number of columns must be divisible by number of workers"
    
    cols_per_worker = div(nj, p)
    
    # First matrix multiplication: tmp = alpha * A * B
    # IMPROVEMENT: Using SharedArrays for zero-copy access
    if isa(tmp, SharedArray) && isa(B, SharedArray)
        @sync begin
            for (idx, w) in enumerate(workers())
                start_col = (idx-1) * cols_per_worker + 1
                end_col = idx * cols_per_worker
                
                @async remotecall_wait(w) do# IMPROVEMENT: Using views to avoid copying data
                    tmp_view = view(tmp, :, start_col:end_col)
                    B_view = view(B, :, start_col:end_col)
                    
                    @inbounds for j = 1:length(start_col:end_col)
                        for i = 1:ni
                            sum_val = zero(eltype(tmp))
                            for k = 1:nk
                                sum_val += alpha * A[i,k] * B_view[k,j]
                            end
                            tmp_view[i,j] = sum_val
                        end
                    end
                end
            end
        end
    else
        # Fallback for non-SharedArrays
        @sync begin
            for (idx, w) in enumerate(workers())
                start_col = (idx-1) * cols_per_worker + 1
                end_col = idx * cols_per_worker
                
                # Send part of B to worker
                B_part = B[:, start_col:end_col]
                
                @async begin
                    # Compute part of tmp on worker
                    tmp_part = remotecall_fetch(w, A, B_part, alpha) do A, B_part, alpha
                        tmp_local = zeros(eltype(A), size(A, 1), size(B_part, 2))
                        @inbounds for j = 1:size(B_part, 2)
                            for i = 1:size(A, 1)
                                for k = 1:size(A, 2)
                                    tmp_local[i,j] += alpha * A[i,k] * B_part[k,j]
                                end
                            end
                        end
                        return tmp_local
                    end
                    
                    # Update the tmp matrix
                    tmp[:, start_col:end_col] = tmp_part
                end
            end
        end
    end
    
    # Ensure we can divide work for second multiplication
    @assert mod(nl, p) == 0 "Number of columns in C must be divisible by number of workers"
    cols_per_worker_2 = div(nl, p)
    
    # Second matrix multiplication: D = tmp * C + beta * D
    if isa(D, SharedArray) && isa(C, SharedArray)
        @sync begin
            for (idx, w) in enumerate(workers())
                start_col = (idx-1) * cols_per_worker_2 + 1
                end_col = idx * cols_per_worker_2
                
                @async remotecall_wait(w) do
                    # IMPROVEMENT: Using views to avoid copying data
                    D_view = view(D, :, start_col:end_col)
                    C_view = view(C, :, start_col:end_col)
                    
                    @inbounds for j = 1:length(start_col:end_col)
                        for i = 1:ni
                            D_view[i,j] *= beta
                            for k = 1:nj
                                D_view[i,j] += tmp[i,k] * C_view[k,j]
                            end
                        end
                    end
                end
            end
        end
    else
        # Fallback for non-SharedArrays
        @sync begin
            for (idx, w) in enumerate(workers())
                start_col = (idx-1) * cols_per_worker_2 + 1
                end_col = idx * cols_per_worker_2
                
                # Send part of C to worker
                C_part = C[:, start_col:end_col]
                D_part = D[:, start_col:end_col]
                
                @async begin
                    # Compute part of D on worker
                    D_part_new = remotecall_fetch(w, tmp, C_part, D_part, beta) do tmp, C_part, D_part, beta
                        @inbounds for j = 1:size(C_part, 2)
                            for i = 1:size(tmp, 1)
                                D_part[i,j] *= beta
                                for k = 1:size(tmp, 2)
                                    D_part[i,j] += tmp[i,k] * C_part[k,j]
                                end
                            end
                        end
                        return D_part
                    end
                    
                    # Update the D matrix
                    D[:, start_col:end_col] = D_part_new
                end
            end
        end
    end
    
    return D
end

# Initialize arrays with the same pattern as in the C benchmark # IMPROVEMENT 8:using in-place operations to reduce allocations
function init_arrays(ni, nj, nk, nl)
    alpha = 1.5f0
    beta = 1.2f0
    
    tmp = zeros(Float32, ni, nj)
    A = zeros(Float32, ni, nk)
    B = zeros(Float32, nk, nj)
    C = zeros(Float32, nj, nl)
    D = zeros(Float32, ni, nl)
    
    @inbounds for i = 1:ni, j = 1:nk
        A[i,j] = ((i * j + 1) % ni) / ni
    end
    
    @inbounds for i = 1:nk, j = 1:nj
        B[i,j] = ((i * (j + 1)) % nj) / nj
    end
    
    @inbounds for i = 1:nj, j = 1:nl
        C[i,j] = ((i * (j + 3) + 1) % nl) / nl
    end
    
    @inbounds for i = 1:ni, j = 1:nl
        D[i,j] = (i * (j + 2) % nk) / nk
    end
    
    return alpha, beta, tmp, A, B, C, D
end

# IMPROVEMENT9: Initialize SharedArrays for distributed computing!!!
function init_shared_arrays(ni, nj, nk, nl)
    alpha = 1.5f0
    beta = 1.2f0
    
    tmp = SharedArray{Float32}(ni, nj)
    A = SharedArray{Float32}(ni, nk)
    B = SharedArray{Float32}(nk, nj)
    C = SharedArray{Float32}(nj, nl)
    D = SharedArray{Float32}(ni, nl)
    
    # Initialize in parallel
    @sync begin
        @distributed for idx = 1:ni*nk
            i = ((idx-1) ÷ nk) + 1
            j = ((idx-1) % nk) + 1
            A[i,j] = ((i * j + 1) % ni) / ni
        end
        
        @distributed for idx = 1:nk*nj
            i = ((idx-1) ÷ nj) + 1
            j = ((idx-1) % nj) + 1
            B[i,j] = ((i * (j + 1)) % nj) / nj
        end
        
        @distributed for idx = 1:nj*nl
            i = ((idx-1) ÷ nl) + 1
            j = ((idx-1) % nl) + 1
            C[i,j] = ((i * (j + 3) + 1) % nl) / nl
        end
        
        @distributed for idx = 1:ni*nl
            i = ((idx-1) ÷ nl) + 1
            j = ((idx-1) % nl) + 1
            D[i,j] = (i * (j + 2) % nk) / nk
        end
    end
    
    return alpha, beta, tmp, A, B, C, D
end

# Benchmark a specific implementation with proper timing
# IMPROVEMENT 10: Using BenchmarkTools properly with setup phase, instead of @time and @btime
function benchmark_2mm(ni, nj, nk, nl, impl_func, impl_name; trials=7, use_shared=false)
    # Initialize arrays based on whether we need SharedArrays
    if use_shared && nworkers() > 1
        alpha, beta, tmp_orig, A, B, C, D_orig = init_shared_arrays(ni, nj, nk, nl)
    else
        alpha, beta, tmp_orig, A, B, C, D_orig = init_arrays(ni, nj, nk, nl)
    end
    
    # Create the benchmark with proper setup
    # IMPROVEMENT: Fixed variable scope issue by interpolating all needed variables
    b = @benchmarkable begin
        copyto!(tmp, $tmp_orig)
        copyto!(D, $D_orig)
        $impl_func($alpha, $beta, tmp, $A, $B, $C, D)
    end setup=(tmp = similar($tmp_orig); D = similar($D_orig)) samples=trials seconds=30
    
    # Run the benchmark
    result = run(b)
    
    # Extract timing statistics
    times = result.times ./ 1e9  # Convert to seconds
    memory_usage = result.memory
    
    return times, memory_usage
end

# Find optimal number of processes for distributed implementations
function find_optimal_procs(ni, nj, nk, nl, impl_func, impl_name; max_procs=16)
    println("\nFinding optimal number of processes for $impl_name...")
    println("Procs | Min Time (s) | Speedup")
    println("------|-------------|--------")
    
    best_time = Inf
    best_procs = 1
    base_time = 0
    
    for p in 1:max_procs
        # Skip if matrix size is not compatible
        if impl_name == "dist3" && (mod(nj, p) != 0 || mod(nl, p) != 0)
            continue
        end
        
        try
            use_shared = (impl_name == "dist3" && p > 1)
            times, _ = benchmark_2mm(ni, nj, nk, nl, impl_func, impl_name, trials=2, use_shared=use_shared)
            min_time = minimum(times)
            
            if p == 1
                base_time = min_time
            end
            
            speedup = base_time / min_time
            @printf("%5d | %11.6f | %7.2fx\n", p, min_time, speedup)
            
            if min_time < best_time
                best_time = min_time
                best_procs = p
            elseif min_time > best_time * 1.1  # Performance degraded by 10% ?  
                println("Performance started degrading. Optimal: $best_procs processes")
                break
            end
        catch e
            println("Error with $p processes: $e")
            break
        end
    end
    
    return best_procs, best_time
end

# Run comprehensive benchmarks
function benchmark_all_implementations(dataset_name, distributed=false)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl = dataset.ni, dataset.nj, dataset.nk, dataset.nl
    
    println("\n" * "="^60)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    println("="^60)
    
    # IMPROVEMENT 11: Check thread and process configuration
    println("\nConfiguration:")
    println("  Threads: $(Threads.nthreads())")
    println("  Processes: $(nprocs()) (Workers: $(nworkers()))")
    
    implementations = Dict(
        "seq" => kernel_2mm_seq,
        "simd" => kernel_2mm_simd,
        "threads" => kernel_2mm_threads,
        "blas" => kernel_2mm_blas,
        "polly" => kernel_2mm_polly
    )
    
    # Run standard implementations
    println("\nImplementation | Min Time (s) | Mean Time (s) | Median Time (s) | Memory (MB)")
    println("--------------|--------------|---------------|-----------------|------------")
    
    for (name, func) in implementations
        times, memory = benchmark_2mm(ni, nj, nk, nl, func, name)
        min_time = minimum(times)
        mean_time = mean(times)
        median_time = median(times)
        memory_mb = memory / 1024^2
        
        @printf("%-13s | %12.6f | %13.6f | %15.6f | %10.2f\n", 
                name, min_time, mean_time, median_time, memory_mb)
    end
    
    # Run distributed implementations if requested
    if distributed && nworkers() > 0
        println("\n" * "-"^60)
        println("Distributed Implementations")
        println("-"^60)
        
        # Test distributed implementation
        if dataset_name in ["MINI", "SMALL", "MEDIUM","LARGE"]
            optimal_procs, best_time = find_optimal_procs(ni, nj, nk, nl, 
                                                         kernel_2mm_distributed, 
                                                         "distributed", 
                                                         max_procs=min(8, ni))
        else
            println("Skipping process optimization for large datasets")
            times, memory = benchmark_2mm(ni, nj, nk, nl, kernel_2mm_distributed, "distributed")
            @printf("%-13s | %12.6f | Memory: %.2f MB\n", 
                    "distributed", minimum(times), memory / 1024^2)
        end
        
        # Test dist3 implementation
        if mod(nj, nworkers()) == 0 && mod(nl, nworkers()) == 0
            if dataset_name in ["MINI", "SMALL", "MEDIUM","LARGE"]
                optimal_procs, best_time = find_optimal_procs(ni, nj, nk, nl, 
                                                             kernel_2mm_dist3, 
                                                             "dist3", 
                                                             max_procs=min(8, ni))
            else
                println("Skipping process optimization for large datasets")
                times, memory = benchmark_2mm(ni, nj, nk, nl, kernel_2mm_dist3, "dist3", use_shared=true)
                @printf("%-13s | %12.6f | Memory: %.2f MB\n", 
                        "dist3", minimum(times), memory / 1024^2)
            end
        else
            println("Skipping dist3: columns ($nj, $nl) not divisible by workers ($(nworkers()))")
        end
    end
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM","LARGE"])
    println("PolyBench 2MM Benchmark")
    println("="^60)
    
    # IMPROVEMENT 12: Verify configuration
    if distributed && nworkers() == 0
        println("WARNING: No worker processes available for distributed computing!")
        println("Add workers using: addprocs(n)")
    end
    
    if Threads.nthreads() > 1 && nworkers() > 0
        println("WARNING: Running with both threads ($(Threads.nthreads())) and workers ($(nworkers()))!")
        println("This may lead to oversubscription. Consider using either threads OR workers.")
    end
    
    if implementation == "all"
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
            "seq" => kernel_2mm_seq,
            "simd" => kernel_2mm_simd,
            "threads" => kernel_2mm_threads,
            "blas" => kernel_2mm_blas,
            "polly" => kernel_2mm_polly
        )
        
        # Add distributed implementations if requested
        if distributed && nworkers() > 0
            impl_funcs["distributed"] = kernel_2mm_distributed
            if mod(nj, nworkers()) == 0 && mod(nl, nworkers()) == 0
                impl_funcs["dist3"] = kernel_2mm_dist3
            end
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            use_shared = (implementation == "dist3" && nworkers() > 1)
            times, memory = benchmark_2mm(ni, nj, nk, nl, func, implementation, use_shared=use_shared)
            
            println("\n===== Performance Benchmark =====")
            min_time = minimum(times)
            mean_time = mean(times)
            median_time = median(times)
            @printf("%s | %.6f | %.6f | %.6f\n", 
                    implementation, min_time, mean_time, median_time)
            
            println("\n===== Memory Usage =====")
            println("$implementation: $memory bytes")
            
            return times, memory
        else
            error("Unknown implementation: $implementation")
        end
    end
end

# Function to verify correctness of all implementations
function verify_implementations(dataset="SMALL")
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl
    
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Reference result using sequential implementation
    D_ref = kernel_2mm_seq(alpha, beta, copy(tmp), A, B, C, copy(D))
    
    # Implementations to test
    implementations = Dict(
        "SIMD" => kernel_2mm_simd,
        "Multithreaded" => kernel_2mm_threads,
        "BLAS" => kernel_2mm_blas,
        "Polly" => kernel_2mm_polly
    )
    
    # Add distributed implementations if workers are available
    if nworkers() > 0
        implementations["Distributed"] = kernel_2mm_distributed
        
        if mod(nj, nworkers()) == 0 && mod(nl, nworkers()) == 0
            implementations["Distributed (Alg 3)"] = kernel_2mm_dist3
        end
    end
    
    results = Dict()
    
    println("Verifying implementation correctness...")
    println("Implementation | Max Absolute Error | Status")
    println("--------------|-------------------|--------")
    
    for (name, func) in implementations
        # Use SharedArrays for dist3 if applicable
        if name == "Distributed (Alg 3)" && nworkers() > 1
            alpha_s, beta_s, tmp_s, A_s, B_s, C_s, D_s = init_shared_arrays(ni, nj, nk, nl)
            D_test = func(alpha_s, beta_s, tmp_s, A_s, B_s, C_s, D_s)
            max_error = maximum(abs.(D_ref - D_test))
        else
            D_test = func(alpha, beta, copy(tmp), A, B, C, copy(D))
            max_error = maximum(abs.(D_ref - D_test))
        end
        
        results[name] = max_error
        status = max_error < 1e-5 ? "PASS" : "FAIL"
        @printf("%-13s | %17.2e | %s\n", name, max_error, status)
    end
    
    return results
end

end # module