#!/usr/bin/env julia
#=
2MM Benchmark Runner
Usage: julia -t 8 run_2mm.jl --dataset MEDIUM --strategies all
=#

using Printf
using LinearAlgebra
using Base.Threads
using Profile

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using PolyBenchJulia
using PolyBenchJulia.Config
using PolyBenchJulia.Metrics
using PolyBenchJulia.BenchCore
using PolyBenchJulia.TwoMM

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
            println("2MM Benchmark Runner")
            println("Usage: julia -t N run_2mm.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --dataset NAME      Dataset size (MINI/SMALL/MEDIUM/LARGE/EXTRALARGE)")
            println("  --strategies LIST   Comma-separated strategies or 'all'")
            println("  --iterations N      Number of timed iterations (default: 10)")
            println("  --warmup N          Number of warmup iterations (default: 5)")
            println("  --output FORMAT     Output format: csv or json")
            println("  --profile           Enable profiling output")
            println("  --no-verify         Skip result verification")
            println("  --help              Show this help")
            println()
            println("Available strategies: ", join(STRATEGIES_2MM, ", "))
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_2mm_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_2MM, dataset_name)
        error("Unknown dataset: $dataset_name. Available: $(join(keys(DATASETS_2MM), ", "))")
    end
    
    params = DATASETS_2MM[dataset_name]
    ni, nj, nk, nl = params.ni, params.nj, params.nk, params.nl
    
    println("="^70)
    println("2MM BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Dimensions: ni=$ni, nj=$nj, nk=$nk, nl=$nl")
    println("Memory: $(round(Config.memory_2mm(ni, nj, nk, nl) / 2^20, digits=2)) MB")
    println("FLOPs: $(Config.flops_2mm(ni, nj, nk, nl))")
    println()
    
    # Allocate arrays
    alpha = Ref(0.0)
    beta = Ref(0.0)
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    tmp = Matrix{Float64}(undef, ni, nj)
    C = Matrix{Float64}(undef, nj, nl)
    D = Matrix{Float64}(undef, ni, nl)
    D_orig = Matrix{Float64}(undef, ni, nl)
    
    # Initialize
    init_2mm!(alpha, beta, A, B, tmp, C, D)
    copyto!(D_orig, D)
    
    # Compute reference result for verification
    D_ref = copy(D_orig)
    tmp_ref = zeros(ni, nj)
    kernel_2mm_seq!(alpha[], beta[], A, B, tmp_ref, C, D_ref)
    
    # Parse strategies
    strategies = if config["strategies"] == "all"
        STRATEGIES_2MM
    else
        split(config["strategies"], ",")
    end
    
    flops = Config.flops_2mm(ni, nj, nk, nl)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_2MM)
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = TwoMM.get_kernel(strategy)
        
        # Setup function for reset
        setup_fn = () -> begin
            fill!(tmp, 0.0)
            copyto!(D, D_orig)
        end
        
        # Profile if requested
        if config["profile"] && strategy == "sequential"
            println("Profiling sequential implementation...")
            setup_fn()
            Profile.clear()
            @profile kernel_fn(alpha[], beta[], A, B, tmp, C, D)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        # Benchmark
        timing = benchmark_kernel(
            () -> kernel_fn(alpha[], beta[], A, B, tmp, C, D),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        # Verify
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(alpha[], beta[], A, B, tmp, C, D)
            verified, max_err = verify_result(D, D_ref)
            if !verified
                @printf("  Verification FAILED for %s (max_err=%.2e)\n", strategy, max_err)
            end
        end
        
        result = BenchmarkResult(
            "2mm",
            dataset_name,
            strategy,
            nthreads(),
            1,
            timing.times_ns,
            timing.allocations,
            timing.memory_bytes,
            flops,
            verified
        )
        
        record!(mc, result)
    end
    
    # Print results
    print_results(mc)
    
    # Export results
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "2mm_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "2mm_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_2mm_benchmark(config)
end
