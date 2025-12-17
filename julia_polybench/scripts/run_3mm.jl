

#!/usr/bin/env julia
#=
3MM Benchmark Runner
Computation: G = (A * B) * (C * D)
    E = A * B
    F = C * D
    G = E * F

Usage:
  julia -t 8 run_3mm.jl --dataset MEDIUM
  julia -t 16 run_3mm.jl --dataset LARGE --strategies threads_static,blas
  julia -t 8 run_3mm.jl --dataset MEDIUM --output csv

DAS-5 SLURM:
  srun -N 1 -c 16 --time=01:00:00 julia -t 16 run_3mm.jl --dataset LARGE
=#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Printf
using LinearAlgebra
using Base.Threads
using Statistics

include(joinpath(@__DIR__, "..", "src", "common", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "Metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "BenchCore.jl"))
include(joinpath(@__DIR__, "..", "src", "kernels", "ThreeMM.jl"))

using .Config
using .Metrics
using .BenchCore
using .ThreeMM

function parse_args(args)
    config = Dict{String, Any}(
        "dataset" => "MEDIUM",
        "strategies" => "all",
        "iterations" => 10,
        "warmup" => 5,
        "verify" => true,
        "output" => "csv",
        "verbose" => false
    )
    
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--dataset" && i < length(args)
            config["dataset"] = uppercase(args[i+1])
            i += 2
        elseif arg == "--strategies" && i < length(args)
            config["strategies"] = args[i+1]
            i += 2
        elseif arg == "--iterations" && i < length(args)
            config["iterations"] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--warmup" && i < length(args)
            config["warmup"] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--no-verify"
            config["verify"] = false
            i += 1
        elseif arg == "--output" && i < length(args)
            config["output"] = args[i+1]
            i += 2
        elseif arg == "--verbose" || arg == "-v"
            config["verbose"] = true
            i += 1
        elseif arg == "--help" || arg == "-h"
            println("3MM Benchmark Runner")
            println("Usage: julia -t N run_3mm.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      MINI/SMALL/MEDIUM/LARGE/EXTRALARGE (default: MEDIUM)")
            println("  --strategies LIST   Comma-separated or 'all' (default: all)")
            println("  --iterations N      Timed iterations (default: 10)")
            println("  --warmup N          Warmup iterations (default: 5)")
            println("  --no-verify         Skip correctness verification")
            println("  --output FORMAT     csv or json (default: csv)")
            println("  --verbose, -v       Verbose output")
            println()
            println("Strategies: ", join(STRATEGIES_3MM, ", "))
            println()
            println("Examples:")
            println("  julia -t 8 run_3mm.jl --dataset LARGE")
            println("  julia -t 16 run_3mm.jl --strategies threads_static,blas --output csv")
            exit(0)
        else
            i += 1
        end
    end
    return config
end

function verify_implementation(strategy::String, kernel_fn,
                               A, B, C, D, E, F, G, G_ref, tolerance)
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    
    kernel_fn(A, B, C, D, E, F, G)
    
    max_error = maximum(abs.(G - G_ref))
    passed = max_error < tolerance
    
    return passed, max_error
end

function run_3mm_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(ThreeMM.DATASETS_3MM, dataset_name)
        error("Unknown dataset: $dataset_name. Available: $(join(keys(ThreeMM.DATASETS_3MM), ", "))")
    end
    
    params = ThreeMM.DATASETS_3MM[dataset_name]
    ni, nj, nk, nl, nm = params.ni, params.nj, params.nk, params.nl, params.nm
    
    # Header
    println("="^70)
    println("3MM BENCHMARK")
    println("="^70)
    Config.print_system_info()
    println()
    println("Dataset: $dataset_name")
    println("Dimensions: ni=$ni, nj=$nj, nk=$nk, nl=$nl, nm=$nm")
    println("Memory: $(round(ThreeMM.memory_3mm(ni, nj, nk, nl, nm) / 2^20, digits=2)) MB")
    
    flops = ThreeMM.flops_3mm(ni, nj, nk, nl, nm)
    println("FLOPs: $flops ($(round(flops/1e9, digits=2)) GFLOPs)")
    println()
    
    # Allocate arrays
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    
    # Initialize
    init_3mm!(A, B, C, D, E, F, G)
    
    # Compute reference result
    E_ref = zeros(ni, nj)
    F_ref = zeros(nj, nl)
    G_ref = zeros(ni, nl)
    kernel_3mm_seq!(A, B, C, D, E_ref, F_ref, G_ref)
    
    # Parse strategies
    strategies = if config["strategies"] == "all"
        STRATEGIES_3MM
    else
        split(config["strategies"], ",")
    end
    
    # Verification
    if config["verify"]
        println("-"^70)
        println("VERIFICATION")
        println("-"^70)
        tolerance = 1e-10
        all_passed = true
        
        for strategy in strategies
            strategy = strip(String(strategy))
            if !(strategy in STRATEGIES_3MM)
                continue
            end
            
            kernel_fn = get_kernel(strategy)
            passed, max_error = verify_implementation(
                strategy, kernel_fn, A, B, C, D, E, F, G, G_ref, tolerance
            )
            
            status = passed ? "PASS" : "FAIL"
            @printf("  %-18s : %s (max_error: %.2e)\n", strategy, status, max_error)
            all_passed &= passed
        end
        
        if !all_passed
            println("\nVerification FAILED. Aborting benchmark.")
            return nothing
        end
        println()
    end
    
    # Benchmark
    println("-"^70)
    println("BENCHMARK RESULTS")
    println("-"^70)
    @printf("%-18s %12s %12s %12s %12s %10s\n",
            "Strategy", "Time(ms)", "Speedup", "Efficiency%", "GFLOP/s", "Allocs")
    println("-"^70)
    
    mc = MetricsCollector()
    seq_time = 0.0
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_3MM)
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = get_kernel(strategy)
        
        # Setup function
        setup_fn = () -> begin
            fill!(E, 0.0)
            fill!(F, 0.0)
            fill!(G, 0.0)
        end
        
        # Benchmark function
        bench_fn = () -> kernel_fn(A, B, C, D, E, F, G)
        
        # Run benchmark
        result = benchmark_kernel(
            bench_fn, setup_fn,
            iterations=config["iterations"],
            warmup=config["warmup"]
        )
        
        # Calculate metrics
        time_ms = result.min_time * 1000
        gflops = flops / result.min_time / 1e9
        
        if strategy == "sequential"
            seq_time = result.min_time
        end
        
        # Determine if threaded strategy
        is_threaded = strategy in ["threads_static", "threads_dynamic", "tiled", "tasks"]
        
        if seq_time > 0
            speedup = seq_time / result.min_time
            efficiency = is_threaded ? (speedup / nthreads()) * 100 : speedup * 100
        else
            speedup = 1.0
            efficiency = 100.0
        end
        
        # Record
        record!(mc, BenchmarkResult(
            strategy, dataset_name, nthreads(),
            result.min_time, result.median_time, result.mean_time, result.std_time,
            speedup, efficiency, gflops, result.allocations, result.memory
        ))
        
        @printf("%-18s %12.3f %12.2f %12.1f %12.2f %10d\n",
                strategy, time_ms, speedup, efficiency, gflops, result.allocations)
    end
    
    # Summary
    println("-"^70)
    println("Best time: $(round(minimum(r.min_time for r in mc.results) * 1000, digits=3)) ms")
    println("Best GFLOP/s: $(round(maximum(r.gflops for r in mc.results), digits=2))")
    println()
    
    # Export results
    if config["output"] == "csv"
        filename = "results_3mm_$(dataset_name)_$(nthreads())t.csv"
        export_csv(mc, filename)
        println("Results saved to: $filename")
    end
    
    return mc
end

# Main
if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    Config.configure_blas_threads()
    run_3mm_benchmark(config)
end