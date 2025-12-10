#!/usr/bin/env julia
#=
3MM Benchmark Runner
Usage: julia -t 8 run_3mm.jl --dataset MEDIUM --strategies all
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
using PolyBenchJulia.ThreeMM

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
            println("3MM Benchmark Runner")
            println("Usage: julia -t N run_3mm.jl [OPTIONS]")
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
            println("Available strategies: ", join(STRATEGIES_3MM, ", "))
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_3mm_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_3MM, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_3MM[dataset_name]
    ni, nj, nk, nl, nm = params.ni, params.nj, params.nk, params.nl, params.nm
    
    println("="^70)
    println("3MM BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Dimensions: ni=$ni, nj=$nj, nk=$nk, nl=$nl, nm=$nm")
    println("Memory: $(round(Config.memory_3mm(ni, nj, nk, nl, nm) / 2^20, digits=2)) MB")
    println("FLOPs: $(Config.flops_3mm(ni, nj, nk, nl, nm))")
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
    
    # Compute reference
    E_ref = zeros(ni, nj)
    F_ref = zeros(nj, nl)
    G_ref = zeros(ni, nl)
    kernel_3mm_seq!(A, B, C, D, E_ref, F_ref, G_ref)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_3MM : split(config["strategies"], ",")
    
    flops = Config.flops_3mm(ni, nj, nk, nl, nm)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_3MM)
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = ThreeMM.get_kernel(strategy)
        
        setup_fn = () -> begin
            fill!(E, 0.0)
            fill!(F, 0.0)
            fill!(G, 0.0)
        end
        
        if config["profile"] && strategy == "sequential"
            setup_fn()
            Profile.clear()
            @profile kernel_fn(A, B, C, D, E, F, G)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(A, B, C, D, E, F, G),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(A, B, C, D, E, F, G)
            v1, _ = verify_result(E, E_ref)
            v2, _ = verify_result(F, F_ref)
            v3, err = verify_result(G, G_ref)
            verified = v1 && v2 && v3
            if !verified
                @printf("  Verification FAILED for %s (G max_err=%.2e)\n", strategy, err)
            end
        end
        
        result = BenchmarkResult(
            "3mm", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, timing.memory_bytes, flops, verified
        )
        
        record!(mc, result)
    end
    
    print_results(mc)
    
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "3mm_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "3mm_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_3mm_benchmark(config)
end
