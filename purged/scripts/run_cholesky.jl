#!/usr/bin/env julia
#=
Cholesky Decomposition Benchmark Runner
Usage: julia -t 8 run_cholesky.jl --dataset MEDIUM --strategies all
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
using PolyBenchJulia.Cholesky

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
            println("Cholesky Benchmark Runner")
            println("Usage: julia -t N run_cholesky.jl [OPTIONS]")
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
            println("Available strategies: ", join(STRATEGIES_CHOLESKY, ", "))
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_cholesky_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_CHOLESKY, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    n = DATASETS_CHOLESKY[dataset_name].n
    
    println("="^70)
    println("CHOLESKY DECOMPOSITION BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Matrix size: $n x $n")
    println("Memory: $(round(Config.memory_cholesky(n) / 2^20, digits=2)) MB")
    println("FLOPs: $(Config.flops_cholesky(n))")
    println()
    
    # Allocate and initialize
    A = Matrix{Float64}(undef, n, n)
    A_orig = Matrix{Float64}(undef, n, n)
    
    init_cholesky!(A)
    copyto!(A_orig, A)
    
    # Compute reference using BLAS
    A_ref = copy(A_orig)
    try
        LAPACK.potrf!('L', A_ref)
    catch e
        @error "Matrix is not positive definite" e
        return nothing
    end
    
    # Parse strategies
    strategies = config["strategies"] == "all" ? STRATEGIES_CHOLESKY : split(config["strategies"], ",")
    
    flops = Config.flops_cholesky(n)
    mc = MetricsCollector()
    
    for strategy in strategies
        strategy = strip(String(strategy))
        
        if !(strategy in STRATEGIES_CHOLESKY)
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = Cholesky.get_kernel(strategy)
        
        setup_fn = () -> copyto!(A, A_orig)
        
        if config["profile"] && strategy == "sequential"
            setup_fn()
            Profile.clear()
            @profile kernel_fn(A)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(A),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        verified = true
        if config["verify"]
            setup_fn()
            try
                kernel_fn(A)
                # Compare lower triangular part
                L = LowerTriangular(A)
                L_ref = LowerTriangular(A_ref)
                verified, max_err = verify_result(L, L_ref, rtol=1e-8)
                if !verified
                    @printf("  Verification FAILED for %s (max_err=%.2e)\n", strategy, max_err)
                end
            catch e
                verified = false
                @printf("  Verification FAILED for %s (exception: %s)\n", strategy, e)
            end
        end
        
        result = BenchmarkResult(
            "cholesky", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, timing.memory_bytes, flops, verified
        )
        
        record!(mc, result)
    end
    
    print_results(mc)
    
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "cholesky_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "cholesky_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_cholesky_benchmark(config)
end

