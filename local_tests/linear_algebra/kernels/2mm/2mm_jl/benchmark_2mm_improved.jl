using BenchmarkTools
using Statistics
using Distributed
using Printf
using LinearAlgebra

# Import kernels
include("kernel_2mm_improved.jl")
include("distributed_kernels_improved.jl")

# Setup distributed if needed
if nprocs() == 1
    addprocs(5)  # Use 8 workers by default
end

# Make distributed kernels available
@everywhere include("distributed_kernels_improved.jl")

# Add safe GFLOPS calculation to handle division by zero
function safe_gflops(flops, time)
    if time > 1e-6
        return flops / (time * 1e9)
    else
        return 0.0  # Return 0 instead of Inf
    end
end

# Add a realistic memory usage calculation function
function measure_memory_usage(func, args...)
    GC.gc()  # Force garbage collection before measuring
    mem_before = Sys.free_memory()
    result = func(args...)
    GC.gc()  # Force garbage collection after measuring
    mem_after = Sys.free_memory()
    
    # Ensure we don't get negative or unrealistically large values
    mem_diff = mem_before - mem_after
    if mem_diff < 0 || mem_diff > 10 * 1024^3  # Cap at 10 GB
        mem_diff = 0
    end
    
    return result, mem_diff
end

function init_array(ni, nj, nk, nl)
    alpha = 1.5
    beta = 1.2
    
    A = zeros(ni, nk)
    B = zeros(nk, nj)
    C = zeros(nj, nl)
    D = zeros(ni, nl)
    tmp = zeros(ni, nj)
    
    # Initialize arrays exactly like in the C benchmark (polybench_2mm.c)
    for i in 1:ni
        for j in 1:nk
            A[i,j] = ((i*j+1) % ni) / ni
        end
    end
    
    for i in 1:nk
        for j in 1:nj
            B[i,j] = (i*(j+1) % nj) / nj
        end
    end
    
    for i in 1:nj
        for j in 1:nl
            C[i,j] = ((i*(j+3)+1) % nl) / nl
        end
    end
    
    for i in 1:ni
        for j in 1:nl
            D[i,j] = (i*(j+2) % nk) / nk
        end
    end
    
    return alpha, beta, A, B, C, D, tmp
end

# Function to validate that all implementations produce correct results
function validate_implementation(impl_func, ni, nj, nk, nl)
    alpha, beta, A, B, C, D, tmp = init_array(ni, nj, nk, nl)
    
    # Create reference result using sequential implementation
    D_ref = copy(D)
    kernel_2mm_sequential_blocked!(ni, nj, nk, nl, alpha, beta, copy(tmp), A, B, C, D_ref)
    
    # Run the implementation to be validated
    if impl_func == kernel_2mm_blas!
        D_test, _ = impl_func(ni, nj, nk, nl, alpha, beta, copy(tmp), A, B, C, copy(D))
    elseif impl_func == kernel_2mm_dist_coarse_improved! || impl_func == kernel_2mm_hierarchical!
        D_test, _ = impl_func(ni, nj, nk, nl, alpha, beta, copy(tmp), A, B, C, copy(D))
    else
        D_test, _ = with_metrics(impl_func, ni, nj, nk, nl, alpha, beta, copy(tmp), A, B, C, copy(D))
    end
    
    # Check if results match within a small tolerance
    max_diff = maximum(abs.(D_ref - D_test))
    return max_diff < 1e-10
end

# Enhanced benchmark function with detailed performance metrics
function benchmark_2mm_improved(ni, nj, nk, nl; runs=3, warmup=true)
    println("\n--------------------------------------------------------------")
    println("Benchmarking 2mm with dimensions: A($(ni)×$(nk)), B($(nk)×$(nj)), C($(nj)×$(nl)), D($(ni)×$(nl))")
    println("--------------------------------------------------------------")
    
    # Define all implementations to test
    implementations = [
        ("Sequential", kernel_2mm_sequential_blocked!),
        ("Blocked Sequential", kernel_2mm_sequential_blocked!),
        ("Task-based Blocks", kernel_2mm_block_tasks!),
        ("Multithreaded", kernel_2mm_threaded!),
        ("SIMD Optimized", kernel_2mm_simd!),
        ("BLAS", kernel_2mm_blas!),
        ("Distributed Coarse", kernel_2mm_dist_coarse_improved!),
        ("Hierarchical", kernel_2mm_hierarchical!)
    ]
    
    results = Dict()
    
    # Validate all implementations
    println("Validating implementations...")
    for (name, func) in implementations
        valid = validate_implementation(func, min(ni, 100), min(nj, 100), min(nk, 100), min(nl, 100))
        println("  $name: $(valid ? "✓ Valid" : "✗ Invalid")")
        if !valid
            println("    WARNING: Implementation does not produce correct results!")
        end
    end
    println()
    
    # Benchmark system info
    println("System Information:")
    println("  Julia Version: $(VERSION)")
    println("  Number of Threads: $(Threads.nthreads())")
    println("  Number of Workers: $(nworkers())")
    println("  CPU: $(Sys.cpu_info()[1].model)")
    println("  Total Memory: $(round(Sys.total_memory() / (1024^3), digits=2)) GB")
    println()
    
    # Do warmup runs if requested
    if warmup
        println("Performing warmup runs...")
        for (name, func) in implementations
            if name ∈ ["Sequential", "Blocked Sequential", "Task-based Blocks", "Multithreaded", "SIMD Optimized"]
                # Warmup with smaller matrices
                warmup_ni, warmup_nj, warmup_nk, warmup_nl = min(ni, 100), min(nj, 100), min(nk, 100), min(nl, 100)
                alpha_w, beta_w, A_w, B_w, C_w, D_w, tmp_w = init_array(warmup_ni, warmup_nj, warmup_nk, warmup_nl)
                with_metrics(func, warmup_ni, warmup_nj, warmup_nk, warmup_nl, alpha_w, beta_w, tmp_w, A_w, B_w, C_w, D_w)
            elseif name == "BLAS"
                # Warmup BLAS
                warmup_ni, warmup_nj, warmup_nk, warmup_nl = min(ni, 100), min(nj, 100), min(nk, 100), min(nl, 100)
                alpha_w, beta_w, A_w, B_w, C_w, D_w, tmp_w = init_array(warmup_ni, warmup_nj, warmup_nk, warmup_nl)
                kernel_2mm_blas!(warmup_ni, warmup_nj, warmup_nk, warmup_nl, alpha_w, beta_w, tmp_w, A_w, B_w, C_w, D_w)
            elseif name ∈ ["Distributed Coarse", "Hierarchical"]
                # Warmup distributed with smaller matrices
                warmup_ni, warmup_nj, warmup_nk, warmup_nl = min(ni, 100), min(nj, 100), min(nk, 100), min(nl, 100)
                alpha_w, beta_w, A_w, B_w, C_w, D_w, tmp_w = init_array(warmup_ni, warmup_nj, warmup_nk, warmup_nl)
                func(warmup_ni, warmup_nj, warmup_nk, warmup_nl, alpha_w, beta_w, tmp_w, A_w, B_w, C_w, D_w)
            end
        end
        println("Warmup complete.\n")
    end
    
    # Run benchmarks
    println("Running benchmarks...")
    
    # First, run the sequential implementation as baseline
    baseline_times = []
    baseline_metrics = Dict()
    
    for r in 1:runs
        alpha, beta, A, B, C, D, tmp = init_array(ni, nj, nk, nl)
        # Use measure_memory_usage to get realistic memory metrics
        (_, metrics), mem_used = measure_memory_usage(with_metrics, kernel_2mm_sequential_blocked!, ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
        push!(baseline_times, metrics["time"])
        for (k, v) in metrics
            if !haskey(baseline_metrics, k)
                baseline_metrics[k] = []
            end
            push!(baseline_metrics[k], v)
        end
        # Add memory usage to metrics
        if !haskey(baseline_metrics, "memory_used")
            baseline_metrics["memory_used"] = []
        end
        push!(baseline_metrics["memory_used"], mem_used)
    end
    
    # Calculate average baseline metrics
    baseline_time = mean(baseline_times)
    for (k, v) in baseline_metrics
        baseline_metrics[k] = mean(v)
    end
    
    # Calculate FLOPS for the problem size
    total_flops = ni * nj * nk * 2 + ni * nl * nj * 2 + ni * nl
    
    # Now run and compare each implementation
    for (name, func) in implementations
        if name == "Sequential"
            # Already measured as baseline
            results[name] = Dict(
                "time" => baseline_time,
                "speedup" => 1.0,
                "efficiency" => 100.0,
                "metrics" => baseline_metrics
            )
            continue
        end
        
        # Run the implementation multiple times
        times = []
        all_metrics = Dict()
        memory_usages = []
        
        for r in 1:runs
            alpha, beta, A, B, C, D, tmp = init_array(ni, nj, nk, nl)
            
            if name ∈ ["Blocked Sequential", "Task-based Blocks", "Multithreaded", "SIMD Optimized"]
                # Use with_metrics wrapper with memory tracking
                (_, metrics), mem_used = measure_memory_usage(with_metrics, func, ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
                push!(times, metrics["time"])
                push!(memory_usages, mem_used)
                
                # Collect all metrics
                for (k, v) in metrics
                    if !haskey(all_metrics, k)
                        all_metrics[k] = []
                    end
                    push!(all_metrics[k], v)
                end
            elseif name == "BLAS"
                # BLAS implementation with memory tracking
                (_, metrics), mem_used = measure_memory_usage(kernel_2mm_blas!, ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
                push!(times, metrics["time"])
                push!(memory_usages, mem_used)
                
                for (k, v) in metrics
                    if !haskey(all_metrics, k)
                        all_metrics[k] = []
                    end
                    push!(all_metrics[k], v)
                end
            elseif name ∈ ["Distributed Coarse", "Hierarchical"]
                # Distributed implementations with memory tracking
                (_, metrics), mem_used = measure_memory_usage(func, ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
                push!(memory_usages, mem_used)
                
                if name == "Distributed Coarse"
                    push!(times, metrics["total_time"])
                else
                    push!(times, metrics["total_time"])
                end
                
                # Collect metrics - these differ by implementation
                for (k, v) in metrics
                    if !isa(v, Dict) && k != "detailed_stats" && k != "worker_stats"
                        if !haskey(all_metrics, k)
                            all_metrics[k] = []
                        end
                        push!(all_metrics[k], v)
                    end
                end
            end
        end
        
        # Calculate average metrics
        avg_time = mean(times)
        avg_memory = mean(memory_usages)
        
        # Ensure we have gflops for all implementations
        if !haskey(all_metrics, "gflops")
            all_metrics["gflops"] = [safe_gflops(total_flops, t) for t in times]
        end
        
        # Add memory usage to metrics
        all_metrics["memory_used"] = memory_usages
        
        avg_metrics = Dict()
        for (k, v) in all_metrics
            avg_metrics[k] = mean(v)
        end
        
        # Calculate speedup and efficiency
        speedup = baseline_time / avg_time
        
        # Efficiency depends on parallelism level
        parallelism = if name == "BLAS"
            4  # Assuming BLAS uses 4 threads by default
        elseif name == "Multithreaded" || name == "Task-based Blocks"
            Threads.nthreads()
        elseif name ∈ ["Distributed Coarse", "Hierarchical"]
            nworkers()
        else
            1
        end
        
        efficiency = (speedup / parallelism) * 100
        
        results[name] = Dict(
            "time" => avg_time,
            "speedup" => speedup,
            "efficiency" => efficiency,
            "metrics" => avg_metrics
        )
    end
    
    # Print results
    println("\nResults:")
    println("-------------------------------------------------------------")
    println("Implementation        |  Time (s)  |  Speedup  |  Efficiency  |  GFLOPS")
    println("-------------------------------------------------------------")
    
    for (name, data) in sort(collect(results), by=x->x[2]["time"])
        time_str = @sprintf("%.6f", data["time"])
        speedup_str = @sprintf("%.2f", data["speedup"])
        efficiency_str = @sprintf("%.2f%%", data["efficiency"])
        
        gflops = if haskey(data["metrics"], "gflops")
            gflops_val = data["metrics"]["gflops"]
            if gflops_val > 1000
                @sprintf("%.2f", 1000.0)  # Cap at 1000 GFLOPS for display
            elseif gflops_val < 0.01
                @sprintf("%.4f", gflops_val)
            else
                @sprintf("%.2f", gflops_val)
            end
        else
            "N/A"
        end
        
        println("$(lpad(name, 20)) | $(rpad(time_str, 10)) | $(rpad(speedup_str, 9)) | $(rpad(efficiency_str, 12)) | $(rpad(gflops, 8))")
    end
    
    println("\nDetailed Metrics:")
    println("-------------------------------------------------------------")
    
    # Print some additional metrics for each implementation
    for (name, data) in sort(collect(results), by=x->x[2]["time"])
        println("$name:")
        
        metrics = data["metrics"]
        if haskey(metrics, "memory_used")
            println("  Memory Used: $(round(metrics["memory_used"] / (1024^2), digits=2)) MB")
        end
        
        if name == "Hierarchical" && haskey(metrics, "threads_per_worker")
            println("  Thread Distribution: $(metrics["threads_per_worker"]) threads across workers")
        end
        
        println()
    end
    
    return results
end

function run_benchmarks_improved()
    # Define dataset sizes similar to polybench_2mm.h
    datasets = [
        (40, 50, 70, 80),      # SMALL_DATASET
        (180, 190, 210, 220),   # MEDIUM_DATASET
        (800, 900, 1100, 1200)  # LARGE_DATASET
    ]
    
    all_results = Dict()
    
    for (ni, nj, nk, nl) in datasets
        println("\nRunning benchmark for dataset size: $(ni)×$(nj)×$(nk)×$(nl)")
        results = benchmark_2mm_improved(ni, nj, nk, nl)
        all_results[(ni, nj, nk, nl)] = results
    end
    
    return all_results
end