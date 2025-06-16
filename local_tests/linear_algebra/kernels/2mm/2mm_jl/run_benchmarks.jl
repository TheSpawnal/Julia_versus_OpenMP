#!/usr/bin/env julia
# Script to run PolyBench 2MM benchmarks with proper configuration
# 
# Usage examples:
#   julia run_benchmarks.jl                    # Single-threaded
#   julia -t 4 run_benchmarks.jl               # Multi-threaded (4 threads)
#   julia -p 4 run_benchmarks.jl               # Multi-process (4 workers)
#   
# DO NOT combine -t and -p flags as this leads to resource contention

using Distributed

# Check configuration
println("=== Configuration ===")
println("Julia version: $(VERSION)")
println("Number of threads: $(Threads.nthreads())")
println("Number of processes: $(nprocs())")
println("Number of workers: $(nworkers())")

# If using distributed processing, verify thread configuration
if nworkers() > 1
    println("\nChecking worker thread configuration...")
    @everywhere println("Process $(myid()): $(Threads.nthreads()) threads")
    
    # Verify all workers have 1 thread
    worker_threads = [@fetchfrom w Threads.nthreads() for w in workers()]
    if any(t -> t > 1, worker_threads)
        @warn "Some workers have multiple threads. This may cause performance issues."
        println("Recommendation: Start workers with single thread using:")
        println("  julia -p N --threads=1 run_benchmarks.jl")
    end
end

# Load the benchmark module
if nworkers() > 1
    @everywhere include("PolyBench2MM_Improved.jl")
    @everywhere using .PolyBench2MM_Improved
else
    include("PolyBench2MM_Improved.jl")
    using .PolyBench2MM_Improved
end

# Configure BenchmarkTools parameters for consistency
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5.0
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false  # Don't GC between samples

println("\n=== Running Benchmarks ===")

# Run the benchmarks
results = PolyBench2MM_Improved.main(
    datasets=["SMALL", "MEDIUM", "LARGE"],
    verify=true
)

# Additional analysis for multi-threaded runs
if Threads.nthreads() > 1
    println("\n=== Thread Scaling Analysis ===")
    println("Note: For best results, compare runs with different thread counts")
    println("Example: julia -t 1 vs julia -t 2 vs julia -t 4")
end

# Additional analysis for distributed runs
if nworkers() > 1
    println("\n=== Distributed Scaling Analysis ===")
    println("Note: For best results, compare runs with different worker counts")
    println("Example: julia -p 1 vs julia -p 2 vs julia -p 4")
    
    # Memory usage per worker
    println("\nMemory usage per worker:")
    @everywhere begin
        mem_used = Base.gc_live_bytes() / 1024^2
        println("Process $(myid()): $(round(mem_used, digits=2)) MB")
    end
end

println("\n=== Benchmark Complete ===")