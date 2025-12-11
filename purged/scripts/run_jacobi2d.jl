#!/usr/bin/env julia
#=
Jacobi-2D Stencil Benchmark Runner
Usage: julia -t 8 run_jacobi2d.jl --dataset MEDIUM --strategies all
=#

using Printf
using Base.Threads
using Profile

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using PolyBenchJulia
using PolyBenchJulia.Config
using PolyBenchJulia.Metrics
using PolyBenchJulia.BenchCore
using PolyBenchJulia.Jacobi2D

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
            println("Jacobi-2D Stencil Benchmark Runner")
            println("Usage: julia -t N run_jacobi2d.jl [OPTIONS]")
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
            println("Available strategies: ", join(STRATEGIES_JACOBI2D, ", "))
            println()
            println("NOTE: Jacobi-2D is memory-bound with low arithmetic intensity.")
            println("      Performance is typically limited by memory bandwidth.")
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function run_jacobi2d_benchmark(config::Dict)
    dataset_name = config["dataset"]
    
    if !haskey(DATASETS_JACOBI2D, dataset_name)
        error("Unknown dataset: $dataset_name")
    end
    
    params = DATASETS_JACOBI2D[dataset_name]
    n = params.n
    tsteps = params.tsteps
    
    println("="^70)
    println("JACOBI-2D STENCIL BENCHMARK")
    println("="^70)
    print_system_info()
    
    println("Dataset: $dataset_name")
    println("Grid size: $n x $n")
    println("Time steps: $tsteps")
    println("Memory: $(round(Config.memory_jacobi2d(n) / 2^20, digits=2)) MB")
    println("FLOPs: $(Config.flops_jacobi2d(n, tsteps))")
    println()
    println("NOTE: This is a memory-bound stencil computation.")
    println("      Arithmetic intensity: ~0.75 FLOPs/byte (very low)")
    println()
    
    # Allocate grids
    A = Matrix{Float64}(undef, n, n)
    B = Matrix{Float64}(undef, n, n)
    A_orig = Matrix{Float64}(undef, n, n)
    
    # Initialize
    init_jacobi2d!(A, B)
    copyto!(A_orig, A)
    
    # Compute reference result
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
            println("Unknown strategy: $strategy, skipping")
            continue
        end
        
        kernel_fn = Jacobi2D.get_kernel(strategy)
        
        setup_fn = () -> begin
            copyto!(A, A_orig)
            copyto!(B, A_orig)
        end
        
        if config["profile"] && strategy == "sequential"
            setup_fn()
            Profile.clear()
            @profile kernel_fn(A, B, tsteps)
            Profile.print(IOContext(stdout, :displaysize => (40, 200)))
        end
        
        timing = benchmark_kernel(
            () -> kernel_fn(A, B, tsteps),
            setup_fn,
            iterations=config["iterations"],
            warmup_iterations=config["warmup"]
        )
        
        verified = true
        if config["verify"]
            setup_fn()
            kernel_fn(A, B, tsteps)
            
            # Compare interior points only
            max_diff = 0.0
            for j in 2:(n-1), i in 2:(n-1)
                diff = abs(A[i, j] - A_ref[i, j])
                max_diff = max(max_diff, diff)
            end
            
            verified = max_diff < 1e-10
            if !verified
                @printf("  Verification FAILED for %s (max_diff=%.2e)\n", strategy, max_diff)
            end
        end
        
        result = BenchmarkResult(
            "jacobi2d", dataset_name, strategy, nthreads(), 1,
            timing.times_ns, timing.allocations, timing.memory_bytes, flops, verified
        )
        
        record!(mc, result)
    end
    
    print_results(mc)
    
    # Memory bandwidth estimation
    println("\nMemory Bandwidth Analysis:")
    seq_result = filter(r -> r.strategy == "sequential", mc.results)
    if !isempty(seq_result)
        bytes_moved = 2 * (n - 2)^2 * 8 * tsteps * 2  # Read + Write, double buffer
        min_time_s = minimum(seq_result[1].times_ns) / 1e9
        bandwidth_gb = (bytes_moved / 1e9) / min_time_s
        @printf("  Estimated bandwidth: %.2f GB/s\n", bandwidth_gb)
    end
    
    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)
    
    timestamp = mc.timestamp
    if config["output"] == "csv"
        export_csv(mc, joinpath(results_dir, "jacobi2d_$(dataset_name)_$(timestamp).csv"))
    elseif config["output"] == "json"
        export_json(mc, joinpath(results_dir, "jacobi2d_$(dataset_name)_$(timestamp).json"))
    end
    
    return mc
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args(ARGS)
    run_jacobi2d_benchmark(config)
end
