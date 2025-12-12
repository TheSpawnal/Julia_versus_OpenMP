#!/usr/bin/env julia
#=
Jacobi-2D Stencil Benchmark Runner
Usage: julia -t 8 run_jacobi2d.jl --dataset MEDIUM

NOTE: This is a memory-bound stencil computation.
      Arithmetic intensity: ~0.75 FLOPs/byte (very low)
      Performance is typically limited by memory bandwidth.
=#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Printf
using Base.Threads
using Statistics

include(joinpath(@__DIR__, "..", "src", "common", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "Metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "BenchCore.jl"))
include(joinpath(@__DIR__, "..", "src", "kernels", "Jacobi2D.jl"))

using .Config
using .Metrics
using .BenchCore
using .Jacobi2D: STRATEGIES_JACOBI2D, DATASETS_JACOBI2D, init_jacobi2d!,
                 get_kernel, kernel_jacobi2d_seq!

function parse_args(args)
    config = Dict{String, Any}(
        "dataset" => "MEDIUM",
        "strategies" => "all",
        "iterations" => 10,
        "warmup" => 5,
        "verify" => true,
        "output" => "csv"
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
        elseif arg == "--help" || arg == "-h"
            println("Jacobi-2D Stencil Benchmark Runner")
            println("Usage: julia -t N run_jacobi2d.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      MINI/SMALL/MEDIUM/LARGE/EXTRALARGE")
            println("  --strategies LIST   Comma-separated or 'all'")
            println("  --iterations N      Timed iterations (default: 10)")
            println("  --warmup N          Warmup iterations (default: 5)")
            println("  --no-verify         Skip verification")
            println("  --output FORMAT     csv or json")
            println()
            println("Strategies: ", join(STRATEGIES_JACOBI2D, ", "))
            println()
            println("NOTE: Memory-bound with low arithmetic intensity (~0.75 FLOPs/byte)")
            exit(0)
        else
            i += 1
        end
    end
    return config
end

function run_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_JACOBI2D, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_JACOBI2D[dataset_name]
    n, tsteps = params.n, params.tsteps
    
    println("="^70)
    println("JACOBI-2D STENCIL BENCHMARK")
    println("="^70)
    Config.print_system_info()
    println("Dataset: $dataset_name (n=$n, tsteps=$tsteps)")
    println("Memory: $(round(2*n*n*8/2^20, digits=2)) MB")
    println("NOTE: Memory-bound (arithmetic intensity ~0.75 FLOPs/byte)")
    println()
    
    # Allocate grids
    A = Matrix{Float64}(undef, n, n)
    B = Matrix{Float64}(undef, n, n)
    A_orig = Matrix{Float64}(undef, n, n)
    
    # Initialize
    init_jacobi2d!(A, B)
    copyto!(A_orig, A)
    
    # Reference result
    A_ref = copy(A_orig)
    B_ref = copy(A_orig)
    kernel_jacobi2d_seq!(A_ref, B_ref, tsteps)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_JACOBI2D : split(config["strategies"], ",")
    
    flops = Config.flops_jacobi2d(n, tsteps)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_JACOBI2D)
            println("Unknown strategy: $strategy")
            continue
        end
        
        kernel_fn = get_kernel(strategy)
        
        setup_fn = () -> begin
            copyto!(A, A_orig)
            copyto!(B, A_orig)
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(A, B, tsteps),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        # Verification (check interior points)
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(A, B, tsteps)
            
            max_diff = 0.0
            for j in 2:(n-1), i in 2:(n-1)
                max_diff = max(max_diff, abs(A[i, j] - A_ref[i, j]))
            end
            
            verified = max_diff < 1e-10
            if !verified
                @printf("  Verification FAILED: %s (err=%.2e)\n", strategy, max_diff)
            end
        end
        
        result = BenchmarkResult(
            "jacobi2d", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, flops, verified
        )
        record!(mc, result)
    end
    
    print_results(mc)
    
    # Memory bandwidth estimation
    println("\nMemory Bandwidth:")
    seq_result = filter(r -> r.strategy == "sequential", mc.results)
    if !isempty(seq_result)
        bytes_moved = 2 * (n - 2)^2 * 8 * tsteps * 2
        min_time_s = minimum(seq_result[1].times_ns) / 1e9
        bandwidth_gb = (bytes_moved / 1e9) / min_time_s
        @printf("  Estimated: %.2f GB/s\n", bandwidth_gb)
    end
    
    # Export results
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "jacobi2d_$(dataset_name)_$(mc.timestamp).csv"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_benchmark(config)
end
