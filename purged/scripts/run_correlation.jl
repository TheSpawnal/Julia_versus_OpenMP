#!/usr/bin/env julia
#=
Correlation Benchmark Runner
Usage: julia -t 8 run_correlation.jl --dataset MEDIUM --strategies all
=#

using Printf
using LinearAlgebra
using Base.Threads
using Profile

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using PolyBenchJulia
using PolyBenchJulia.Config
using PolyBenchJulia.Metrics
using PolyBenchJulia.BenchCore
using PolyBenchJulia.Correlation

function parse_args(args)
    config = Dict{String, Any}(
        "dataset" => "MEDIUM",
        "strategies" => "all",
        "iterations" => 10,
        "warmup" => 5,
        "output" => "csv",
        "profile" => false,
        "verify" => true
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
        elseif arg == "--output" && i < length(args)
            config["output"] = args[i+1]
            i += 2
        elseif arg == "--profile"
            config["profile"] = true
            i += 1
        elseif arg == "--no-verify"
            config["verify"] = false
            i += 1
        elseif arg == "--help" || arg == "-h"
            println("Correlation Benchmark Runner")
            println("Usage: julia -t N run_correlation.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      Dataset size (MINI/SMALL/MEDIUM/LARGE/EXTRALARGE)")
            println("  --strategies LIST   Comma-separated strategies or 'all'")
            println("  --iterations N      Number of timed iterations (default: 10)")
            println("  --warmup N          Number of warmup iterations (default: 5)")
            println("  --output FORMAT     Output format: csv or json")
            println("  --profile           Enable profiling output")
            println("  --no-verify         Skip result verification")
            println()
            println("Available strategies: ", join(STRATEGIES_CORRELATION, ", "))
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_correlation_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_CORRELATION, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_CORRELATION[dataset_name]
    m, n = params.m, params.n
    
    println("="^70)
    println("CORRELATION BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Dimensions: M=$m data points, N=$n variables")
    println("Memory: $(round(Config.memory_correlation(m, n) / 2^20, digits=2)) MB")
    println("FLOPs: $(Config.flops_correlation(m, n))")
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
    
    # Compute reference
    data_ref = copy(data_orig)
    mean_ref = zeros(n)
    stddev_ref = zeros(n)
    corr_ref = zeros(n, n)
    for i in 1:n
        corr_ref[i, i] = 1.0
    end
    kernel_correlation_seq!(data_ref, mean_ref, stddev_ref, corr_ref)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_CORRELATION : split(config["strategies"], ",")
    
    flops = Config.flops_correlation(m, n)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_CORRELATION)
            println("Unknown strategy: $strategy, skipping")
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
        
        if config["profile"] && strategy == "sequential"
            setup_fn()
            Profile.clear()
            @profile kernel_fn(data, mean, stddev, corr)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(data, mean, stddev, corr),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(data, mean, stddev, corr)
            verified, max_err = verify_result(corr, corr_ref, rtol=1e-8)
            if !verified
                @printf("  Verification FAILED for %s (max_err=%.2e)\n", strategy, max_err)
            end
            
            # Check diagonal elements are 1.0
            for i in 1:n
                if abs(corr[i, i] - 1.0) > 1e-10
                    verified = false
                    @printf("  Verification FAILED: diagonal element [%d,%d] = %.6f (expected 1.0)\n", i, i, corr[i, i])
                    break
                end
            end
        end
        
        result = BenchmarkResult(
            "correlation", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, timing.memory_bytes, flops, verified
        )
        
        record!(mc, result)
    end
    
    print_results(mc)
    
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "correlation_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "correlation_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_correlation_benchmark(config)
end
