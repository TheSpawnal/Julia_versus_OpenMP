#!/usr/bin/env julia
#=
Nussinov RNA Folding Benchmark Runner
Usage: julia -t 8 run_nussinov.jl --dataset MEDIUM --strategies all
=#

using Printf
using Base.Threads
using Profile

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using PolyBenchJulia
using PolyBenchJulia.Config
using PolyBenchJulia.Metrics
using PolyBenchJulia.BenchCore
using PolyBenchJulia.Nussinov

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
            println("Nussinov RNA Folding Benchmark Runner")
            println("Usage: julia -t N run_nussinov.jl [OPTIONS]")
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
            println("Available strategies: ", join(STRATEGIES_NUSSINOV, ", "))
            println()
            println("NOTE: Nussinov has limited parallelism due to wavefront dependencies.")
            println("      Expected speedup: 2-4x typical (not linear)")
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_nussinov_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_NUSSINOV, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    n = DATASETS_NUSSINOV[dataset_name].n
    
    println("="^70)
    println("NUSSINOV RNA FOLDING BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Sequence length: $n")
    println("Memory: $(round(Config.memory_nussinov(n) / 2^20, digits=2)) MB")
    println("FLOPs (approx): $(Config.flops_nussinov(n))")
    println()
    println("NOTE: This benchmark has wavefront dependencies limiting parallelism.")
    println()
    
    # Allocate arrays
    seq = Vector{Int8}(undef, n)
    table = Matrix{Int}(undef, n, n)
    
    # Initialize
    init_nussinov!(seq, table)
    
    # Compute reference
    table_ref = zeros(Int, n, n)
    kernel_nussinov_seq!(seq, table_ref)
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_NUSSINOV : split(config["strategies"], ",")
    
    flops = Config.flops_nussinov(n)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_NUSSINOV)
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = Nussinov.get_kernel(strategy)
        
        setup_fn = () -> fill!(table, 0)
        
        if config["profile"] && strategy == "sequential"
            setup_fn()
            Profile.clear()
            @profile kernel_fn(seq, table)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(seq, table),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(seq, table)
            
            # Check final result (maximum base pairs)
            result_score = table[1, n]
            ref_score = table_ref[1, n]
            
            if result_score != ref_score
                verified = false
                @printf("  Verification FAILED for %s (score=%d, expected=%d)\n", 
                        strategy, result_score, ref_score)
            else
                # Check full table for consistency
                max_diff = 0
                for i in 1:n, j in (i+1):n
                    diff = abs(table[i, j] - table_ref[i, j])
                    max_diff = max(max_diff, diff)
                end
                if max_diff > 0
                    verified = false
                    @printf("  Verification FAILED for %s (max table diff=%d)\n", 
                            strategy, max_diff)
                end
            end
        end
        
        result = BenchmarkResult(
            "nussinov", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, timing.memory_bytes, flops, verified
        )
        
        record!(mc, result)
    end
    
    print_results(mc)
    
    # Print final result
    println("\nFinal Score (max base pairs): $(table_ref[1, n])")
    
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "nussinov_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "nussinov_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_nussinov_benchmark(config)
end
