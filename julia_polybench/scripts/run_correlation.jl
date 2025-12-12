#!/usr/bin/env julia
#=
Correlation Matrix Benchmark Runner
Usage: julia -t 8 run_correlation.jl --dataset MEDIUM
=#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Printf
using LinearAlgebra
using Base.Threads
using Statistics

include(joinpath(@__DIR__, "..", "src", "common", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "Metrics.jl"))
include(joinpath(@__DIR__, "..", "src", "common", "BenchCore.jl"))
include(joinpath(@__DIR__, "..", "src", "kernels", "Correlation.jl"))

using .Config
using .Metrics
using .BenchCore
using .Correlation

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
            println("Correlation Benchmark Runner")
            println("Usage: julia -t N run_correlation.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      MINI/SMALL/MEDIUM/LARGE/EXTRALARGE")
            println("  --strategies LIST   Comma-separated or 'all'")
            println("  --iterations N      Timed iterations (default: 10)")
            println("  --warmup N          Warmup iterations (default: 5)")
            println("  --no-verify         Skip verification")
            println("  --output FORMAT     csv or json")
            println()
            println("Strategies: ", join(STRATEGIES_CORRELATION, ", "))
            exit(0)
        else
            i += 1
        end
    end
    return config
end

function run_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_CORRELATION, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_CORRELATION[dataset_name]
    m, n = params.m, params.n
    
    println("="^70)
    println("CORRELATION MATRIX BENCHMARK")
    println("="^70)
    Config.print_system_info()
    println("Dataset: $dataset_name (m=$m, n=$n)")
    println("Memory: $(round((m*n + 2*n + n*n)*8/2^20, digits=2)) MB")
    println()
    
    # Allocate arrays
    data = Matrix{Float64}(undef, m, n)
    data_orig = Matrix{Float64}(undef, m, n)
    mean = Vector{Float64}(undef, n)
    stddev = Vector{Float64}(undef, n)
    corr = Matrix{Float64}(undef, n, n)
    
    # Initialize
    init_correlation!(data, mean, stddev, corr)
    copyto!(data_orig, data)
    
    # Reference result
    data_ref = copy(data_orig)
    corr_ref = zeros(Float64, n, n)
    for i in 1:n
        corr_ref[i, i] = 1.0
    end
    kernel_correlation_seq!(data_ref, copy(mean), copy(stddev), corr_ref)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_CORRELATION : split(config["strategies"], ",")
    
    flops = Config.flops_correlation(m, n)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_CORRELATION)
            println("Unknown strategy: $strategy")
            continue
        end
        
        kernel_fn = Correlation.get_kernel(strategy)
        
        setup_fn = () -> begin
            copyto!(data, data_orig)
            fill!(mean, 0.0)
            fill!(stddev, 0.0)
            fill!(corr, 0.0)
            for i in 1:n
                corr[i, i] = 1.0
            end
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(data, mean, stddev, corr),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        # Verification
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(data, mean, stddev, corr)
            max_err = maximum(abs.(corr - corr_ref))
            verified = max_err < 1e-6
            
            # Also check diagonal is 1
            for i in 1:n
                if abs(corr[i, i] - 1.0) > 1e-10
                    verified = false
                end
            end
            
            if !verified
                @printf("  Verification FAILED: %s (err=%.2e)\n", strategy, max_err)
            end
        end
        
        result = BenchmarkResult(
            "correlation", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, flops, verified
        )
        record!(mc, result)
    end
    
    print_results(mc)
    
    # Export results
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "correlation_$(dataset_name)_$(mc.timestamp).csv"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_benchmark(config)
end

