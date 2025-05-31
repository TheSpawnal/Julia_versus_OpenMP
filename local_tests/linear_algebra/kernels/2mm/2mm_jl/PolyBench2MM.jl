module PolyBench2MM
# PolyBench 2MM Benchmark suite for Julia
# 
# This module implements the 2mm kernel from PolyBench in various optimized forms
# for Julia. The kernel computes D := alpha*A*B*C + beta*D, which involves two
# matrix multiplications.
#
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

# Dataset sizes according to PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => (ni=16, nj=18, nk=22, nl=24),
    "SMALL" => (ni=40, nj=50, nk=70, nl=80),
    "MEDIUM" => (ni=180, nj=190, nk=210, nl=220),
    "LARGE" => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# 1. Sequential Implementation (baseline)
function kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    @inbounds for i = 1:ni
        for j = 1:nj
            tmp[i,j] = zero(eltype(tmp))
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @inbounds for i = 1:ni
        for j = 1:nl
            D[i,j] *= beta
            for k = 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 2. SIMD Optimization
function kernel_2mm_simd(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    @inbounds for i = 1:ni
        for j = 1:nj
            sum_val = zero(eltype(tmp))
            @simd for k = 1:nk
                sum_val += alpha * A[i,k] * B[k,j]
            end
            tmp[i,j] = sum_val
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @inbounds for i = 1:ni
        for j = 1:nl
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

# 3. Multithreaded Implementation
function kernel_2mm_threads(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    @threads for i = 1:ni
        @inbounds for j = 1:nj
            sum_val = zero(eltype(tmp))
            for k = 1:nk
                sum_val += alpha * A[i,k] * B[k,j]
            end
            tmp[i,j] = sum_val
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @threads for i = 1:ni
        @inbounds for j = 1:nl
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
    
    # Second matrix multiplication: D += tmp * C
    mul!(D, tmp, C, 1.0, 1.0)  # D = 1.0 * tmp * C + 1.0 * D
    
    return D
end

# 5. Polly-style optimization (Loop tiling and cache optimization)
function kernel_2mm_polly(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Tile size (tuned for cache efficiency)
    tile_size = 64
    
    # First matrix multiplication with tiling: tmp = alpha * A * B
    fill!(tmp, zero(eltype(tmp)))
    @inbounds for ii = 1:tile_size:ni
        for jj = 1:tile_size:nj
            for kk = 1:tile_size:nk
                for i = ii:min(ii+tile_size-1, ni)
                    for j = jj:min(jj+tile_size-1, nj)
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
    
    # Second matrix multiplication with tiling: D = tmp * C + beta * D
    @inbounds for ii = 1:tile_size:ni
        for jj = 1:tile_size:nl
            for i = ii:min(ii+tile_size-1, ni)
                for j = jj:min(jj+tile_size-1, nl)
                    D[i,j] *= beta
                end
            end
            for kk = 1:tile_size:nj
                for i = ii:min(ii+tile_size-1, ni)
                    for j = jj:min(jj+tile_size-1, nl)
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

# 6. Distributed Implementation using @distributed macro
function kernel_2mm_distributed(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    @sync @distributed for i = 1:ni
        for j = 1:nj
            tmp[i,j] = zero(eltype(tmp))
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @sync @distributed for i = 1:ni
        for j = 1:nl
            D[i,j] *= beta
            for k = 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 7. Distributed Implementation (Algorithm 3 from matrix_instructions.txt)
function kernel_2mm_dist3(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Ensure the number of workers is suitable for the matrix size
    p = nworkers()
    @assert mod(ni, p) == 0 "Number of rows must be divisible by number of workers"
    
    rows_per_worker = div(ni, p)
    
    # First matrix multiplication: tmp = alpha * A * B
    @sync begin
        for (idx, w) in enumerate(workers())
            start_row = (idx-1) * rows_per_worker + 1
            end_row = idx * rows_per_worker
            
            # Send part of A to worker
            A_part = A[start_row:end_row, :]
            
            @async begin
                # Compute part of tmp on worker
                tmp_part = remotecall_fetch(w, A_part, B, alpha) do A_part, B, alpha
                    tmp_local = zeros(eltype(A_part), size(A_part, 1), size(B, 2))
                    @inbounds for i = 1:size(A_part, 1)
                        for j = 1:size(B, 2)
                            for k = 1:size(A_part, 2)
                                tmp_local[i,j] += alpha * A_part[i,k] * B[k,j]
                            end
                        end
                    end
                    return tmp_local
                end
                
                # Update the tmp matrix
                tmp[start_row:end_row, :] = tmp_part
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @sync begin
        for (idx, w) in enumerate(workers())
            start_row = (idx-1) * rows_per_worker + 1
            end_row = idx * rows_per_worker
            
            # Send part of tmp to worker
            tmp_part = tmp[start_row:end_row, :]
            D_part = D[start_row:end_row, :]
            
            @async begin
                # Compute part of D on worker
                D_part_new = remotecall_fetch(w, tmp_part, C, D_part, beta) do tmp_part, C, D_part, beta
                    @inbounds for i = 1:size(tmp_part, 1)
                        for j = 1:size(C, 2)
                            D_part[i,j] *= beta
                            for k = 1:size(tmp_part, 2)
                                D_part[i,j] += tmp_part[i,k] * C[k,j]
                            end
                        end
                    end
                    return D_part
                end
                
                # Update the D matrix
                D[start_row:end_row, :] = D_part_new
            end
        end
    end
    
    return D
end

# Initialize arrays with the same pattern as in the C benchmark
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
        B[i,j] = ((i * j + 1) % nj) / nj
    end
    
    @inbounds for i = 1:nj, j = 1:nl
        C[i,j] = ((i * (j + 3) + 1) % nl) / nl
    end
    
    @inbounds for i = 1:ni, j = 1:nl
        D[i,j] = (i * (j + 2) % nk) / nk
    end
    
    return alpha, beta, tmp, A, B, C, D
end

# Benchmark a specific implementation with proper timing
function benchmark_2mm(ni, nj, nk, nl, impl_func, impl_name; trials=7)
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Warmup run
    impl_func(alpha, beta, copy(tmp), A, B, C, copy(D))
    
    times = Float64[]
    memory_usage = 0
    
    for trial in 1:trials
        # Reset matrices
        D_copy = copy(D)
        tmp_copy = copy(tmp)
        
        # Time the execution
        start_time = time()
        result = @timed impl_func(alpha, beta, tmp_copy, A, B, C, D_copy)
        elapsed = result.time
        memory_usage = max(memory_usage, result.bytes)
        
        push!(times, elapsed)
    end
    
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
        # Check if we need to add more processes
        if nworkers() < p
            addprocs(p - nworkers())
        end
        
        # Skip if matrix size is not compatible
        if impl_name == "dist3" && mod(ni, p) != 0
            continue
        end
        
        try
            times, _ = benchmark_2mm(ni, nj, nk, nl, impl_func, impl_name, trials=2)
            min_time = minimum(times)
            
            if p == 1
                base_time = min_time
            end
            
            speedup = base_time / min_time
            @printf("%5d | %11.6f | %7.2fx\n", p, min_time, speedup)
            
            if min_time < best_time
                best_time = min_time
                best_procs = p
            elseif min_time > best_time * 1.1  # Performance degraded by 10%
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
        if dataset_name in ["MINI", "SMALL", "MEDIUM"]
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
        if mod(ni, nworkers()) == 0
            if dataset_name in ["MINI", "SMALL", "MEDIUM"]
                optimal_procs, best_time = find_optimal_procs(ni, nj, nk, nl, 
                                                             kernel_2mm_dist3, 
                                                             "dist3", 
                                                             max_procs=min(8, ni))
            else
                println("Skipping process optimization for large datasets")
                times, memory = benchmark_2mm(ni, nj, nk, nl, kernel_2mm_dist3, "dist3")
                @printf("%-13s | %12.6f | Memory: %.2f MB\n", 
                        "dist3", minimum(times), memory / 1024^2)
            end
        else
            println("Skipping dist3: rows ($ni) not divisible by workers ($(nworkers()))")
        end
    end
end

# Main function
function main(; implementation="all", distributed=false, datasets=["MINI", "SMALL", "MEDIUM","LARGE"])
    println("PolyBench 2MM Benchmark")
    println("="^60)
    
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
            if mod(ni, nworkers()) == 0
                impl_funcs["dist3"] = kernel_2mm_dist3
            end
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            times, memory = benchmark_2mm(ni, nj, nk, nl, func, implementation)
            
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
        
        if mod(ni, nworkers()) == 0
            implementations["Distributed (Alg 3)"] = kernel_2mm_dist3
        end
    end
    
    results = Dict()
    
    println("Verifying implementation correctness...")
    println("Implementation | Max Absolute Error | Status")
    println("--------------|-------------------|--------")
    
    for (name, func) in implementations
        D_test = func(alpha, beta, copy(tmp), A, B, C, copy(D))
        max_error = maximum(abs.(D_ref - D_test))
        results[name] = max_error
        status = max_error < 1e-5 ? "PASS" : "FAIL"
        @printf("%-13s | %17.2e | %s\n", name, max_error, status)
    end
    
    return results
end

end # module