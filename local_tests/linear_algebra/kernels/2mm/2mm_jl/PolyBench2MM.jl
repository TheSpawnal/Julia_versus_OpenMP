module PolyBench2MM
# PolyBench 2MM Benchmark suite for Julia - Simple Version
# Fixes the interpolation issues while maintaining proper benchmarking

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

# 1. Sequential Implementation (baseline) - Column-major optimized
function kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    # Reordered for column-major access pattern (k-i-j instead of i-j-k)
    fill!(tmp, zero(eltype(tmp)))
    @inbounds for k = 1:nk
        for i = 1:ni
            alpha_A_ik = alpha * A[i,k]
            for j = 1:nj
                tmp[i,j] += alpha_A_ik * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    # Scale D by beta first
    @inbounds for j = 1:nl
        for i = 1:ni
            D[i,j] *= beta
        end
    end
    
    # Add tmp * C (k-i-j order for cache efficiency)
    @inbounds for k = 1:nj
        for i = 1:ni
            tmp_ik = tmp[i,k]
            for j = 1:nl
                D[i,j] += tmp_ik * C[k,j]
            end
        end
    end
    
    return D
end

# 2. SIMD Optimization with column-major access
function kernel_2mm_simd(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    fill!(tmp, zero(eltype(tmp)))
    @inbounds for k = 1:nk
        for i = 1:ni
            alpha_A_ik = alpha * A[i,k]
            @simd for j = 1:nj
                tmp[i,j] += alpha_A_ik * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @inbounds for j = 1:nl
        @simd for i = 1:ni
            D[i,j] *= beta
        end
    end
    
    @inbounds for k = 1:nj
        for i = 1:ni
            tmp_ik = tmp[i,k]
            @simd for j = 1:nl
                D[i,j] += tmp_ik * C[k,j]
            end
        end
    end
    
    return D
end

# 3. Multithreaded Implementation with column-wise distribution
function kernel_2mm_threads(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    fill!(tmp, zero(eltype(tmp)))
    @threads for k = 1:nk
        @inbounds for i = 1:ni
            alpha_A_ik = alpha * A[i,k]
            for j = 1:nj
                tmp[i,j] += alpha_A_ik * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @threads for j = 1:nl
        @inbounds for i = 1:ni
            D[i,j] *= beta
        end
    end
    
    @threads for k = 1:nj
        @inbounds for i = 1:ni
            tmp_ik = tmp[i,k]
            for j = 1:nl
                D[i,j] += tmp_ik * C[k,j]
            end
        end
    end
    
    return D
end

# 4. BLAS Optimized Implementation
function kernel_2mm_blas(alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)
    
    # Scale D by beta
    lmul!(beta, D)
    
    # Second matrix multiplication: D += tmp * C
    mul!(D, tmp, C, 1.0, 1.0)
    
    return D
end

# 5. Polly-style optimization with proper tiling
function kernel_2mm_polly(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Cache-friendly tile size
    tile_size = 64
    
    # First matrix multiplication with blocking
    fill!(tmp, zero(eltype(tmp)))
    @inbounds for kk = 1:tile_size:nk
        for ii = 1:tile_size:ni
            for jj = 1:tile_size:nj
                for k = kk:min(kk+tile_size-1, nk)
                    for i = ii:min(ii+tile_size-1, ni)
                        alpha_A_ik = alpha * A[i,k]
                        @simd for j = jj:min(jj+tile_size-1, nj)
                            tmp[i,j] += alpha_A_ik * B[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # Scale D by beta
    @inbounds for j = 1:nl
        @simd for i = 1:ni
            D[i,j] *= beta
        end
    end
    
    # Second matrix multiplication with blocking
    @inbounds for kk = 1:tile_size:nj
        for ii = 1:tile_size:ni
            for jj = 1:tile_size:nl
                for k = kk:min(kk+tile_size-1, nj)
                    for i = ii:min(ii+tile_size-1, ni)
                        tmp_ik = tmp[i,k]
                        @simd for j = jj:min(jj+tile_size-1, nl)
                            D[i,j] += tmp_ik * C[k,j]
                        end
                    end
                end
            end
        end
    end
    
    return D
end

# Sequential helper function for distributed computing (must be defined everywhere)
@everywhere function matmul_seq_helper!(C, A, B)
    m, k = size(A)
    n = size(B, 2)
    @assert size(A, 2) == size(B, 1)
    @assert size(C) == (m, n)
    
    fill!(C, zero(eltype(C)))
    @inbounds for kk = 1:k
        for i = 1:m
            A_ik = A[i, kk]
            for j = 1:n
                C[i,j] += A_ik * B[kk,j]
            end
        end
    end
    return C
end

# Proper distributed implementation following Algorithm 3 pattern
function kernel_2mm_distributed(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # Check if we have workers
    if nworkers() == 0
        return kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    end
    
    # For proper load balancing, distribute rows among workers
    P = nworkers()
    
    # First matrix multiplication: tmp = alpha * A * B
    # Distribute computation by rows of tmp (and A)
    rows_per_worker = div(ni, P)
    remaining_rows = ni % P
    
    @sync begin
        row_start = 1
        for (idx, w) in enumerate(workers())
            # Calculate rows for this worker
            rows_for_worker = rows_per_worker + (idx <= remaining_rows ? 1 : 0)
            row_end = row_start + rows_for_worker - 1
            
            if rows_for_worker > 0
                # Extract the data this worker needs
                A_part = A[row_start:row_end, :]
                
                @async begin
                    # Send work to worker and get result
                    tmp_part = remotecall_fetch(w, A_part, B, alpha) do A_local, B_local, alpha_local
                        tmp_local = zeros(eltype(A_local), size(A_local, 1), size(B_local, 2))
                        matmul_seq_helper!(tmp_local, A_local, B_local)
                        # Apply alpha scaling
                        tmp_local .*= alpha_local
                        return tmp_local
                    end
                    
                    # Store result back
                    tmp[row_start:row_end, :] = tmp_part
                end
            end
            
            row_start = row_end + 1
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    # Scale D by beta first
    lmul!(beta, D)
    
    # Distribute second multiplication by rows of D (and tmp)
    @sync begin
        row_start = 1
        for (idx, w) in enumerate(workers())
            rows_for_worker = rows_per_worker + (idx <= remaining_rows ? 1 : 0)
            row_end = row_start + rows_for_worker - 1
            
            if rows_for_worker > 0
                tmp_part = tmp[row_start:row_end, :]
                D_part = D[row_start:row_end, :]
                
                @async begin
                    D_part_result = remotecall_fetch(w, tmp_part, C, D_part) do tmp_local, C_local, D_local
                        result = zeros(eltype(D_local), size(tmp_local, 1), size(C_local, 2))
                        matmul_seq_helper!(result, tmp_local, C_local)
                        result .+= D_local
                        return result
                    end
                    
                    D[row_start:row_end, :] = D_part_result
                end
            end
            
            row_start = row_end + 1
        end
    end
    
    return D
end

# Enhanced distributed implementation with optimal load calculation
function kernel_2mm_dist_optimal(alpha, beta, tmp, A, B, C, D, load_per_worker=nothing)
    ni, nk = size(A)
    nj, nl = size(C)
    
    if nworkers() == 0
        return kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    end
    
    P = nworkers()
    
    # Calculate optimal load per worker if not provided
    if load_per_worker === nothing
        load_per_worker = div(ni, P)
        if load_per_worker == 0
            load_per_worker = 1
        end
    end
    
    # Ensure total work doesn't exceed matrix size
    total_work = min(load_per_worker * P, ni)
    actual_workers_needed = div(total_work, load_per_worker)
    
    # First matrix multiplication with optimal load distribution
    @sync begin
        for iw = 1:actual_workers_needed
            w = workers()[iw]
            lb = 1 + (iw-1) * load_per_worker
            ub = min(iw * load_per_worker, ni)
            
            A_w = A[lb:ub, :]
            
            @async begin
                tmp_part = remotecall_fetch(w, A_w, B, alpha) do A_local, B_local, alpha_local
                    result = zeros(eltype(A_local), size(A_local, 1), size(B_local, 2))
                    matmul_seq_helper!(result, A_local, B_local)
                    result .*= alpha_local
                    return result
                end
                
                tmp[lb:ub, :] = tmp_part
            end
        end
    end
    
    # Scale D by beta
    lmul!(beta, D)
    
    # Second matrix multiplication with same load distribution
    @sync begin
        for iw = 1:actual_workers_needed
            w = workers()[iw]
            lb = 1 + (iw-1) * load_per_worker
            ub = min(iw * load_per_worker, ni)
            
            tmp_w = tmp[lb:ub, :]
            D_w = D[lb:ub, :]
            
            @async begin
                D_part = remotecall_fetch(w, tmp_w, C, D_w) do tmp_local, C_local, D_local
                    result = zeros(eltype(D_local), size(tmp_local, 1), size(C_local, 2))
                    matmul_seq_helper!(result, tmp_local, C_local)
                    result .+= D_local
                    return result
                end
                
                D[lb:ub, :] = D_part
            end
        end
    end
    
    return D
end

# Initialize arrays with the same pattern as in the C benchmark
function init_arrays(ni, nj, nk, nl; dtype=Float32)
    alpha = dtype(1.5)
    beta = dtype(1.2)
    
    tmp = zeros(dtype, ni, nj)
    A = zeros(dtype, ni, nk)
    B = zeros(dtype, nk, nj)
    C = zeros(dtype, nj, nl)
    D = zeros(dtype, ni, nl)
    
    @inbounds for i = 1:ni, j = 1:nk
        A[i,j] = dtype(((i * j + 1) % ni) / ni)
    end
    
    @inbounds for i = 1:nk, j = 1:nj
        B[i,j] = dtype(((i * j + 1) % nj) / nj)
    end
    
    @inbounds for i = 1:nj, j = 1:nl
        C[i,j] = dtype(((i * (j + 3) + 1) % nl) / nl)
    end
    
    @inbounds for i = 1:ni, j = 1:nl
        D[i,j] = dtype((i * (j + 2) % nk) / nk)
    end
    
    return alpha, beta, tmp, A, B, C, D
end

# Simple but accurate timing function
function time_implementation(func, alpha, beta, tmp, A, B, C, D; trials=7)
    times = Float64[]
    
    # Warmup
    func(alpha, beta, copy(tmp), A, B, C, copy(D))
    GC.gc()
    
    for i in 1:trials
        tmp_copy = copy(tmp)
        D_copy = copy(D)
        
        # Time the execution
        start_time = time_ns()
        result = func(alpha, beta, tmp_copy, A, B, C, D_copy)
        end_time = time_ns()
        
        elapsed = (end_time - start_time) / 1e9  # Convert to seconds
        push!(times, elapsed)
        
        # Force garbage collection between trials
        GC.gc()
    end
    
    return times
end

# Memory allocation tracking
function track_allocations(func, alpha, beta, tmp, A, B, C, D)
    # Warmup
    func(alpha, beta, copy(tmp), A, B, C, copy(D))
    GC.gc()
    
    # Track allocations
    tmp_copy = copy(tmp)
    D_copy = copy(D)
    
    alloc_before = Base.gc_alloc_count()
    bytes_before = Base.gc_bytes()
    
    result = func(alpha, beta, tmp_copy, A, B, C, D_copy)
    
    alloc_after = Base.gc_alloc_count()
    bytes_after = Base.gc_bytes()
    
    return (
        allocations = alloc_after - alloc_before,
        bytes = bytes_after - bytes_before
    )
end

# Find optimal load per worker for distributed implementations
function find_optimal_load(ni, nj, nk, nl; max_load=200, step=25)
    if nworkers() == 0
        println("No workers available for load optimization")
        return nothing
    end
    
    println("\nOptimizing load per worker...")
    println("Load | Min Time (s) | Mean Time (s) | Efficiency | Workers Used")
    println("-----|--------------|---------------|------------|-------------")
    
    P = nworkers()
    best_time = Inf
    best_load = step
    
    for load in step:step:max_load
        # Skip if load would require more workers than available
        required_workers = div(ni, load) + (ni % load > 0 ? 1 : 0)
        if required_workers > P
            continue
        end
        
        try
            alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
            
            # Time the implementation
            times = time_implementation(
                (a, b, t, A, B, C, D) -> kernel_2mm_dist_optimal(a, b, t, A, B, C, D, load),
                alpha, beta, tmp, A, B, C, D,
                trials=3
            )
            
            min_time = minimum(times)
            mean_time = mean(times)
            
            # Calculate efficiency
            theoretical_speedup = min(P, div(ni, load))
            efficiency = theoretical_speedup > 0 ? 100.0 / theoretical_speedup : 0.0
            
            @printf("%4d | %12.6f | %13.6f | %9.1f%% | %11d\n", 
                    load, min_time, mean_time, efficiency, required_workers)
            
            if min_time < best_time
                best_time = min_time
                best_load = load
            end
            
        catch e
            println("Error with load $load: $e")
        end
    end
    
    println("\nOptimal load per worker: $best_load")
    return best_load
end

# Comprehensive benchmark
function benchmark_all_implementations(dataset_name; track_memory=false)
    dataset = DATASET_SIZES[dataset_name]
    ni, nj, nk, nl = dataset.ni, dataset.nj, dataset.nk, dataset.nl
    
    println("\n" * "="^80)
    println("Dataset: $dataset_name (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    println("Threads: $(Threads.nthreads()), Workers: $(nworkers())")
    println("BLAS Threads: $(BLAS.get_num_threads())")
    println("="^80)
    
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    implementations = [
        ("Sequential", kernel_2mm_seq),
        ("SIMD", kernel_2mm_simd),
        ("Multithreaded", kernel_2mm_threads),
        ("BLAS", kernel_2mm_blas),
        ("Polly", kernel_2mm_polly)
    ]
    
    # Add distributed implementations if workers available
    if nworkers() > 0
        push!(implementations, ("Distributed", kernel_2mm_distributed))
        
        # Find optimal load for this dataset
        if dataset_name in ["MINI", "SMALL", "MEDIUM"]
            println("\nFinding optimal load distribution...")
            optimal_load = find_optimal_load(ni, nj, nk, nl)
            
            if optimal_load !== nothing
                optimal_func = (alpha, beta, tmp, A, B, C, D) -> 
                              kernel_2mm_dist_optimal(alpha, beta, tmp, A, B, C, D, optimal_load)
                push!(implementations, ("Dist Optimal", optimal_func))
            end
        else
            # For large datasets, use a reasonable default
            default_load = div(ni, nworkers())
            optimal_func = (alpha, beta, tmp, A, B, C, D) -> 
                          kernel_2mm_dist_optimal(alpha, beta, tmp, A, B, C, D, default_load)
            push!(implementations, ("Dist Optimal", optimal_func))
        end
    end
    
    # Print header
    if track_memory
        println("\nImplementation | Min Time (s) | Mean Time (s) | Allocations | Memory (KB)")
        println("--------------|--------------|---------------|-------------|------------")
    else
        println("\nImplementation | Min Time (s) | Mean Time (s) | Speedup")
        println("--------------|--------------|---------------|--------")
    end
    
    baseline_time = nothing
    
    for (name, func) in implementations
        try
            if track_memory
                # Track memory allocations
                alloc_info = track_allocations(func, alpha, beta, tmp, A, B, C, D)
                times = time_implementation(func, alpha, beta, tmp, A, B, C, D, trials=3)
                min_time = minimum(times)
                mean_time = mean(times)
                
                @printf("%-13s | %12.6f | %13.6f | %11d | %10.1f\n", 
                        name, min_time, mean_time, alloc_info.allocations, alloc_info.bytes/1024)
            else
                # Time measurement
                times = time_implementation(func, alpha, beta, tmp, A, B, C, D)
                min_time = minimum(times)
                mean_time = mean(times)
                
                if baseline_time === nothing
                    baseline_time = min_time
                    speedup_str = "baseline"
                else
                    speedup = baseline_time / min_time
                    speedup_str = @sprintf("%.2fx", speedup)
                end
                
                @printf("%-13s | %12.6f | %13.6f | %s\n", 
                        name, min_time, mean_time, speedup_str)
            end
            
        catch e
            println("Error benchmarking $name: $e")
        end
    end
end

# Verification function
function verify_implementations(dataset="SMALL")
    dataset_params = DATASET_SIZES[dataset]
    ni, nj, nk, nl = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl
    
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Reference result using BLAS (most reliable)
    D_ref = kernel_2mm_blas(alpha, beta, copy(tmp), A, B, C, copy(D))
    
    implementations = [
        ("Sequential", kernel_2mm_seq),
        ("SIMD", kernel_2mm_simd),
        ("Multithreaded", kernel_2mm_threads),
        ("Polly", kernel_2mm_polly)
    ]
    
    if nworkers() > 0
        implementations = vcat(implementations, [
            ("Distributed", kernel_2mm_distributed),
            ("Dist Optimal", (alpha, beta, tmp, A, B, C, D) -> 
                            kernel_2mm_dist_optimal(alpha, beta, tmp, A, B, C, D, 50))
        ])
    end
    
    println("Verifying implementation correctness against BLAS reference...")
    println("Implementation | Max Absolute Error | Status")
    println("--------------|-------------------|--------")
    
    all_correct = true
    
    for (name, func) in implementations
        try
            D_test = func(alpha, beta, copy(tmp), A, B, C, copy(D))
            max_error = maximum(abs.(D_ref - D_test))
            status = max_error < 1e-4 ? "PASS" : "FAIL"
            @printf("%-13s | %17.2e | %s\n", name, max_error, status)
            
            if status == "FAIL"
                all_correct = false
            end
        catch e
            println("$name: ERROR - $e")
            all_correct = false
        end
    end
    
    return all_correct
end

# Main function
function main(; implementation="all", dataset="MEDIUM", track_memory=false, verify_first=true)
    println("PolyBench 2MM Benchmark - Simple Version")
    println("Julia Version: $(VERSION)")
    println("BLAS Threads: $(BLAS.get_num_threads())")
    
    if verify_first
        println("\n=== Verification Phase ===")
        if !verify_implementations(dataset)
            println("⚠️  Some implementations failed verification!")
            println("Proceeding with benchmarks anyway...")
        else
            println("✅ All implementations verified correct!")
        end
    end
    
    if implementation == "all"
        return benchmark_all_implementations(dataset, track_memory=track_memory)
    else
        # Run specific implementation
        dataset_params = DATASET_SIZES[dataset]
        ni, nj, nk, nl = dataset_params.ni, dataset_params.nj, dataset_params.nk, dataset_params.nl
        
        alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
        
        impl_funcs = Dict(
            "seq" => kernel_2mm_seq,
            "simd" => kernel_2mm_simd,
            "threads" => kernel_2mm_threads,
            "blas" => kernel_2mm_blas,
            "polly" => kernel_2mm_polly
        )
        
        if nworkers() > 0
            impl_funcs["distributed"] = kernel_2mm_distributed
            impl_funcs["dist_optimal"] = (alpha, beta, tmp, A, B, C, D) -> 
                                        kernel_2mm_dist_optimal(alpha, beta, tmp, A, B, C, D, 50)
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            
            println("Benchmarking $implementation on dataset $dataset...")
            
            if track_memory
                alloc_info = track_allocations(func, alpha, beta, tmp, A, B, C, D)
                println("Allocations: $(alloc_info.allocations)")
                println("Memory: $(alloc_info.bytes / 1024) KB")
            end
            
            times = time_implementation(func, alpha, beta, tmp, A, B, C, D)
            min_time = minimum(times)
            mean_time = mean(times)
            std_time = std(times)
            
            println("Results:")
            @printf("  Min time:  %.6f seconds\n", min_time)
            @printf("  Mean time: %.6f seconds\n", mean_time)
            @printf("  Std dev:   %.6f seconds\n", std_time)
            
            return times
        else
            error("Unknown implementation: $implementation. Available: $(keys(impl_funcs))")
        end
    end
end

# Convenience functions for common use cases
function benchmark_sequential(dataset="MEDIUM")
    return main(implementation="seq", dataset=dataset, verify_first=false)
end

function benchmark_distributed(dataset="MEDIUM")
    if nworkers() == 0
        println("No workers available. Adding 4 workers...")
        addprocs(4)
        @everywhere include("PolyBench2MM.jl")
        @everywhere using .PolyBench2MM
    end
    return main(implementation="distributed", dataset=dataset, verify_first=false)
end

function compare_all(dataset="MEDIUM")
    return main(implementation="all", dataset=dataset, verify_first=true)
end

# Export main functions
export main, benchmark_all_implementations, verify_implementations
export benchmark_sequential, benchmark_distributed, compare_all
export find_optimal_load, DATASET_SIZES

end # module