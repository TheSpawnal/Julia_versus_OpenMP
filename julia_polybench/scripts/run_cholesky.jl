#!/usr/bin/env julia
#=
Cholesky Decomposition Benchmark Runner
Usage: julia -t 8 run_cholesky.jl --dataset MEDIUM
=#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Printf
using LinearAlgebra
using Base.Threads
using Statistics

include(joinpath(@__DIR__, "..", "src", "common", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "Metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "BenchCore.jl"))
include(joinpath(@__DIR__, "..", "src", "kernels", "Cholesky.jl"))

using .Config
using .Metrics
using .BenchCore
using .Cholesky

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
            println("Cholesky Benchmark Runner")
            println("Usage: julia -t N run_cholesky.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      MINI/SMALL/MEDIUM/LARGE/EXTRALARGE")
            println("  --strategies LIST   Comma-separated or 'all'")
            println("  --iterations N      Timed iterations (default: 10)")
            println("  --warmup N          Warmup iterations (default: 5)")
            println("  --no-verify         Skip verification")
            println("  --output FORMAT     csv or json")
            println()
            println("Strategies: ", join(STRATEGIES_CHOLESKY, ", "))
            exit(0)
        else
            i += 1
        end
    end
    return config
end

function run_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_CHOLESKY, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_CHOLESKY[dataset_name]
    n = params.n
    
    println("="^70)
    println("CHOLESKY DECOMPOSITION BENCHMARK")
    println("="^70)
    Config.print_system_info()
    println("Dataset: $dataset_name (n=$n)")
    println("Memory: $(round(n*n*8/2^20, digits=2)) MB")
    println()
    
    # Allocate matrix
    A = Matrix{Float64}(undef, n, n)
    A_orig = Matrix{Float64}(undef, n, n)
    
    # Initialize and save original
    init_cholesky!(A)
    copyto!(A_orig, A)
    
    # Reference result
    A_ref = copy(A_orig)
    kernel_cholesky_seq!(A_ref)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_CHOLESKY : split(config["strategies"], ",")
    
    flops = Config.flops_cholesky(n)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_CHOLESKY)
            println("Unknown strategy: $strategy")
            continue
        end
        
        kernel_fn = Cholesky.get_kernel(strategy)
        
        setup_fn = () -> copyto!(A, A_orig)
        
        timing = benchmark_kernel(
            () -> kernel_fn(A),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        # Verification
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(A)
            max_err = maximum(abs.(A - A_ref))
            verified = max_err < 1e-6
            if !verified
                @printf("  Verification FAILED: %s (err=%.2e)\n", strategy, max_err)
            end
        end
        
        result = BenchmarkResult(
            "cholesky", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, flops, verified
        )
        record!(mc, result)
    end
    
    print_results(mc)
    
    # Export results
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "cholesky_$(dataset_name)_$(mc.timestamp).csv"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_benchmark(config)
end
