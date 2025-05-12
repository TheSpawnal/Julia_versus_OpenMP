module PolyBench2MM
# PolyBench 2MM Benchmark Suite for Julia
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
#   4. Now run benchmarks: PolyBench2MM.main()
#
# To run a specific implementation:
#   PolyBench2MM.main(implementation="blas")
#
# Available implementations:
#   - "seq" (Sequential baseline)
#   - "simd" (SIMD-optimized)
#   - "threads" (Multithreaded)
#   - "blas" (BLAS-optimized)
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

# 1. Sequential Implementation (baseline)
function kernel_2mm_seq(alpha, beta, tmp, A, B, C, D)
    ni, nk = size(A)
    nj, nl = size(C)
    
    # First matrix multiplication: tmp = alpha * A * B
    fill!(tmp, zero(eltype(tmp)))
    for i = 1:ni
        for j = 1:nj
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    for i = 1:ni
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
    fill!(tmp, zero(eltype(tmp)))
    for i = 1:ni
        for j = 1:nj
            sum_val = zero(eltype(tmp))
            @simd for k = 1:nk
                sum_val += alpha * A[i,k] * B[k,j]
            end
            tmp[i,j] = sum_val
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    for i = 1:ni
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
    fill!(tmp, zero(eltype(tmp)))
    
    # Parallelize over rows
    @threads for i = 1:ni
        for j = 1:nj
            for k = 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = tmp * C + beta * D
    @threads for i = 1:ni
        for j = 1:nl
            D[i,j] *= beta
            for k = 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# 4. BLAS Optimized Implementation
function kernel_2mm_blas(alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha * A * B
    mul!(tmp, A, B, alpha, 0.0)  # tmp = alpha * A * B + 0.0 * tmp
    
    # Scale D by beta
    rmul!(D, beta)
    
    # Second matrix multiplication: D += tmp * C
    mul!(D, tmp, C, 1.0, 1.0)  # D = 1.0 * tmp * C + 1.0 * D
    
    return D
end

# 5. Distributed Implementation using @distributed macro
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

# 6. Distributed Implementation (Algorithm 3 from matrix_instructions.txt)
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
                tmp_part = remotecall_fetch(w) do
                    tmp_local = zeros(eltype(tmp), size(A_part, 1), nj)
                    for i = 1:size(A_part, 1)
                        for j = 1:nj
                            for k = 1:nk
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
                D_part = remotecall_fetch(w) do
                    for i = 1:size(tmp_part, 1)
                        for j = 1:nl
                            D_part[i,j] *= beta
                            for k = 1:nj
                                D_part[i,j] += tmp_part[i,k] * C[k,j]
                            end
                        end
                    end
                    return D_part
                end
                
                # Update the D matrix
                D[start_row:end_row, :] = D_part
            end
        end
    end
    
    return D
end

# 7. MPI-style Implementation using distributed approach
# NOTE: Commented out as it requires the MPI.jl package to be installed
#=
function kernel_2mm_mpi(alpha, beta, tmp, A, B, C, D)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        ni, nk = size(A)
        nj, nl = size(C)
        # Broadcast dimensions
        dims = [ni, nj, nk, nl]
    else
        dims = Array{Int}(undef, 4)
    end
    
    MPI.Bcast!(dims, 0, comm)
    ni, nj, nk, nl = dims
    
    # Distribute B and C to all processes
    if rank != 0
        B = Array{eltype(A)}(undef, nk, nj)
        C = Array{eltype(A)}(undef, nj, nl)
    end
    MPI.Bcast!(B, 0, comm)
    MPI.Bcast!(C, 0, comm)
    
    # Determine local work for each process
    rows_per_proc = div(ni, nprocs)
    start_row = rank * rows_per_proc + 1
    end_row = (rank == nprocs-1) ? ni : (rank+1) * rows_per_proc
    local_rows = end_row - start_row + 1
    
    # Local portions of matrices
    A_local = (rank == 0) ? A[start_row:end_row, :] : Array{eltype(A)}(undef, local_rows, nk)
    tmp_local = Array{eltype(A)}(undef, local_rows, nj)
    D_local = (rank == 0) ? D[start_row:end_row, :] : Array{eltype(A)}(undef, local_rows, nl)
    
    # Scatter A
    if rank == 0
        for p = 1:nprocs-1
            p_start = p * rows_per_proc + 1
            p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
            p_rows = p_end - p_start + 1
            MPI.Send(A[p_start:p_end, :], p, 0, comm)
        end
    else
        MPI.Recv!(A_local, 0, 0, comm)
    end
    
    # First matrix multiplication: tmp_local = alpha * A_local * B
    for i = 1:local_rows
        for j = 1:nj
            tmp_local[i,j] = zero(eltype(tmp))
            for k = 1:nk
                tmp_local[i,j] += alpha * A_local[i,k] * B[k,j]
            end
        end
    end
    
    # Scatter D if not process 0
    if rank == 0
        for p = 1:nprocs-1
            p_start = p * rows_per_proc + 1
            p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
            p_rows = p_end - p_start + 1
            MPI.Send(D[p_start:p_end, :], p, 1, comm)
        end
    else
        MPI.Recv!(D_local, 0, 1, comm)
    end
    
    # Second matrix multiplication: D_local = tmp_local * C + beta * D_local
    for i = 1:local_rows
        for j = 1:nl
            D_local[i,j] *= beta
            for k = 1:nj
                D_local[i,j] += tmp_local[i,k] * C[k,j]
            end
        end
    end
    
    # Gather results back to process 0
    if rank == 0
        for p = 1:nprocs-1
            p_start = p * rows_per_proc + 1
            p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
            p_rows = p_end - p_start + 1
            D_p = Array{eltype(D)}(undef, p_rows, nl)
            MPI.Recv!(D_p, p, 2, comm)
            D[p_start:p_end, :] = D_p
        end
    else
        MPI.Send(D_local, 0, 2, comm)
    end
    
    return (rank == 0) ? D : nothing
end
=#

# Initialize arrays with the same pattern as in the C benchmark
function init_arrays(ni, nj, nk, nl)
    alpha = 1.5f0
    beta = 1.2f0
    
    tmp = zeros(Float32, ni, nj)
    A = zeros(Float32, ni, nk)
    B = zeros(Float32, nk, nj)
    C = zeros(Float32, nj, nl)
    D = zeros(Float32, ni, nl)
    
    for i = 1:ni, j = 1:nk
        A[i,j] = ((i * j + 1) % ni) / ni
    end
    
    for i = 1:nk, j = 1:nj
        B[i,j] = ((i * j + 1) % nj) / nj
    end
    
    for i = 1:nj, j = 1:nl
        C[i,j] = ((i * (j + 3) + 1) % nl) / nl
    end
    
    for i = 1:ni, j = 1:nl
        D[i,j] = (i * (j + 2) % nk) / nk
    end
    
    return alpha, beta, tmp, A, B, C, D
end

# Benchmark a specific implementation
function benchmark_2mm(ni, nj, nk, nl, impl_func; warmup=true)
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Warmup run (to trigger compilation)
    if warmup
        impl_func(alpha, beta, copy(tmp), A, B, C, copy(D))
    end
    
    # Benchmark
    result = @benchmark $impl_func($alpha, $beta, copy($tmp), $A, $B, $C, copy($D))
    
    return result
end

# Measure memory usage of an implementation
function memory_usage_2mm(ni, nj, nk, nl, impl_func)
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Measure memory allocation
    mem = @allocated impl_func(alpha, beta, copy(tmp), A, B, C, copy(D))
    
    return mem
end

# Run comprehensive benchmarks for all implementations
function benchmark_all_2mm_implementations(ni, nj, nk, nl)
    implementations = Dict(
        "Sequential" => kernel_2mm_seq,
        "SIMD" => kernel_2mm_simd,
        "Multithreaded" => kernel_2mm_threads,
        "BLAS" => kernel_2mm_blas
    )
    
    # Add distributed implementations if workers are available
    if nworkers() > 0
        # First, check if the distributed setup is ready
        try
            @everywhere function test_worker_ready() 
                return true
            end
            
            # Test if workers can execute a function
            for w in workers()
                ready = remotecall_fetch(test_worker_ready, w)
                if !ready
                    error("Worker $w is not ready")
                end
            end
            
            println("Distributed setup is ready with $(nworkers()) workers")
            
            # Now add distributed implementations
            implementations["Distributed"] = kernel_2mm_distributed
            
            # Only add dist3 if the matrix size is compatible with the number of workers
            if mod(ni, nworkers()) == 0
                implementations["Distributed (Alg 3)"] = kernel_2mm_dist3
            else
                @warn "Skipping Distributed (Alg 3) as number of rows ($ni) is not divisible by number of workers ($(nworkers()))"
            end
        catch e
            println("Skipping distributed implementations due to: $e")
        end
    else
        println("No worker processes available. Skipping distributed implementations.")
    end
    
    results = Dict()
    memory_usage = Dict()
    
    for (name, func) in implementations
        println("Benchmarking $name implementation...")
        results[name] = benchmark_2mm(ni, nj, nk, nl, func)
        memory_usage[name] = memory_usage_2mm(ni, nj, nk, nl, func)
    end
    
    # Print results
    println("\n===== Performance Benchmarks =====")
    println("Implementation | Min Time (s) | Mean Time (s) | Median Time (s)")
    println("-------------|------------|-------------|-------------")
    for (name, result) in results
        min_time = minimum(result.times) / 1e9
        mean_time = mean(result.times) / 1e9
        median_time = median(result.times) / 1e9
        println("$name | $(round(min_time, digits=6)) | $(round(mean_time, digits=6)) | $(round(median_time, digits=6))")
    end
    
    println("\n===== Memory Usage =====")
    println("Implementation | Memory (bytes)")
    println("-------------|---------------")
    for (name, mem) in memory_usage
        println("$name | $mem")
    end
    
    # Calculate speedups relative to sequential implementation
    seq_time = minimum(results["Sequential"].times) / 1e9
    println("\n===== Speedup (relative to Sequential) =====")
    println("Implementation | Speedup")
    println("-------------|--------")
    for (name, result) in results
        if name != "Sequential"
            speedup = seq_time / (minimum(result.times) / 1e9)
            println("$name | $(round(speedup, digits=2))x")
        end
    end
    
    return results, memory_usage
end

# Main function to run benchmarks with default or custom matrix sizes
function main(; ni=800, nj=900, nk=1100, nl=1200, implementation="all")
    println("PolyBench 2MM Benchmark")
    println("Matrix sizes: A($ni×$nk), B($nk×$nj), C($nj×$nl), D($ni×$nl)")
    
    if implementation == "all"
        results, memory = benchmark_all_2mm_implementations(ni, nj, nk, nl)
        return results, memory
    else
        # Get the specified implementation function
        impl_funcs = Dict(
            "seq" => kernel_2mm_seq,
            "simd" => kernel_2mm_simd,
            "threads" => kernel_2mm_threads,
            "blas" => kernel_2mm_blas
        )
        
        # Add distributed implementation if workers are available
        if nworkers() > 0
            impl_funcs["distributed"] = kernel_2mm_distributed
            
            # Only add dist3 if the matrix size is compatible with the number of workers
            if mod(ni, nworkers()) == 0
                impl_funcs["dist3"] = kernel_2mm_dist3
            end
        end
        
        if haskey(impl_funcs, implementation)
            func = impl_funcs[implementation]
            result = benchmark_2mm(ni, nj, nk, nl, func)
            memory = memory_usage_2mm(ni, nj, nk, nl, func)
            
            println("\n===== Performance Benchmark =====")
            min_time = minimum(result.times) / 1e9
            mean_time = mean(result.times) / 1e9
            median_time = median(result.times) / 1e9
            println("$implementation | $(round(min_time, digits=6)) | $(round(mean_time, digits=6)) | $(round(median_time, digits=6))")
            
            println("\n===== Memory Usage =====")
            println("$implementation: $memory bytes")
            
            return result, memory
        else
            error("Unknown implementation: $implementation")
        end
    end
end

# Function to compare correctness of all implementations
function verify_implementations(ni=100, nj=110, nk=120, nl=130)
    alpha, beta, tmp, A, B, C, D = init_arrays(ni, nj, nk, nl)
    
    # Reference result using sequential implementation
    D_ref = kernel_2mm_seq(alpha, beta, copy(tmp), A, B, C, copy(D))
    
    # Implementations to test
    implementations = Dict(
        "SIMD" => kernel_2mm_simd,
        "Multithreaded" => kernel_2mm_threads,
        "BLAS" => kernel_2mm_blas
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
    println("Implementation | Max Absolute Error")
    println("-------------|-------------------")
    
    for (name, func) in implementations
        D_test = func(alpha, beta, copy(tmp), A, B, C, copy(D))
        max_error = maximum(abs.(D_ref - D_test))
        results[name] = max_error
        println("$name | $max_error")
    end
    
    return results
end

end # module