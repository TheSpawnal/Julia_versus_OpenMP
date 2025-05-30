include("benchmark_2mm_improved.jl")

# Setup performance monitoring if available
try
    using PerfMonitor
    global has_perf_monitor = true
catch
    @warn "PerfMonitor package not available. Advanced hardware monitoring metrics will be disabled."
    global has_perf_monitor = false
end

try
    using CPUTime
    global has_cputime = true
catch
    @warn "CPUTime package not available. CPU time measurements will be approximated."
    global has_cputime = false
end

function run_all_benchmarks(sizes=nothing)
    # Configure Julia for best performance
    if Threads.nthreads() == 1
        @warn "Running with only one thread. For better performance, start Julia with more threads."
        @warn "Example: julia --threads=auto run_benchmarks_improved.jl"
    end
    
    if nprocs() == 1
        @warn "Running with only one process. For distributed benchmarks, more workers are needed."
        @warn "Workers will be added automatically."
    end
    
    # Print system information
    println("\n=== System Information ===")
    println("Julia Version: $(VERSION)")
    println("Number of Threads: $(Threads.nthreads())")
    println("Number of Processes: $(nprocs())")
    println("BLAS Library: $(BLAS.vendor())")
    
    # Check for hardware counters
    if has_perf_monitor
        println("Hardware Performance Monitoring: Available")
    else
        println("Hardware Performance Monitoring: Not Available")
    end
    
    # Run the standard benchmark sizes if none provided
    if sizes === nothing
        println("\n=== Running Standard PolyBench Sizes ===")
        return run_benchmarks_improved()
    else
        println("\n=== Running Custom Benchmark Sizes ===")
        
        all_results = Dict()
        for (ni, nj, nk, nl) in sizes
            println("\nRunning benchmark for dataset size: $(ni)×$(nj)×$(nk)×$(nl)")
            results = benchmark_2mm_improved(ni, nj, nk, nl)
            all_results[(ni, nj, nk, nl)] = results
        end
        return all_results
    end
end

# If run directly (not included), execute the benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    # Define custom sizes if needed
    # custom_sizes = [(100, 100, 100, 100), (500, 500, 500, 500)]
    # run_all_benchmarks(custom_sizes)
    
    # Run with standard PolyBench sizes
    run_all_benchmarks()
end