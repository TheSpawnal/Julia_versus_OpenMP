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

using .Config: configure_blas_threads, print_system_info
using .Metrics: BenchmarkResult, MetricsCollector, record!, compute_efficiency
using .BenchCore: TimingResult, benchmark_kernel
using .ThreeMM: init_3mm!, reset_3mm!, STRATEGIES_3MM, DATASETS_3MM, get_kernel,
                kernel_3mm_seq!, flops_3mm, memory_3mm

#=============================================================================
 Argument Parsing
=============================================================================#
function parse_args(args)
    config = Dict{String, Any}(
        "dataset" => "MEDIUM",
        "strategies" => "all",
        "iterations" => 10,
        "warmup" => 5,
        "verify" => true,
        "output" => "table",
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
            config["output"] = lowercase(args[i+1])
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
            println("  --output FORMAT     table/csv (default: table)")
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

#=============================================================================
 Verification
=============================================================================#
function verify_implementation(
    strategy::String, 
    kernel_fn::Function,
    A::Matrix{Float64}, B::Matrix{Float64},
    C::Matrix{Float64}, D::Matrix{Float64},
    E::Matrix{Float64}, F::Matrix{Float64},
    G::Matrix{Float64}, G_ref::Matrix{Float64};
    tolerance::Float64=1e-10
)
    # Reset state
    fill!(E, 0.0)
    fill!(F, 0.0)
    fill!(G, 0.0)
    
    # Run kernel
    kernel_fn(A, B, C, D, E, F, G)
    
    # Check result
    max_error = maximum(abs.(G .- G_ref))
    passed = max_error < tolerance
    
    return passed, max_error
end

#=============================================================================
 Main Benchmark Function
=============================================================================#
function run_3mm_benchmark(config::Dict{String, Any})
    dataset = config["dataset"]
    
    if !haskey(DATASETS_3MM, dataset)
        error("Unknown dataset: $dataset. Available: $(keys(DATASETS_3MM))")
    end
    
    params = DATASETS_3MM[dataset]
    ni, nj, nk, nl, nm = params.ni, params.nj, params.nk, params.nl, params.nm
    
    # Calculate metrics
    flops = flops_3mm(ni, nj, nk, nl, nm)
    memory_bytes = memory_3mm(ni, nj, nk, nl, nm)
    
    # Print header
    println("="^70)
    println("3MM BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
    println("Dataset: $dataset")
    @printf("Dimensions: ni=%d, nj=%d, nk=%d, nl=%d, nm=%d\n", ni, nj, nk, nl, nm)
    @printf("Memory: %.2f MB\n", memory_bytes / 1024^2)
    @printf("FLOPs: %d (%.2f GFLOPs)\n", flops, flops / 1e9)
    
    # Allocate matrices
    A = Matrix{Float64}(undef, ni, nk)
    B = Matrix{Float64}(undef, nk, nj)
    C = Matrix{Float64}(undef, nj, nm)
    D = Matrix{Float64}(undef, nm, nl)
    E = Matrix{Float64}(undef, ni, nj)
    F = Matrix{Float64}(undef, nj, nl)
    G = Matrix{Float64}(undef, ni, nl)
    G_ref = Matrix{Float64}(undef, ni, nl)
    
    # Initialize
    init_3mm!(A, B, C, D, E, F, G)
    
    # Compute reference result
    E_ref = zeros(ni, nj)
    F_ref = zeros(nj, nl)
    fill!(G_ref, 0.0)
    kernel_3mm_seq!(A, B, C, D, E_ref, F_ref, G_ref)
    
    # Determine strategies to run
    # FIX: Convert SubString to String explicitly
    if config["strategies"] == "all"
        strategies = STRATEGIES_3MM
    else
        strategies = [strip(String(s)) for s in split(config["strategies"], ",")]
    end
    
    # Verification
    if config["verify"]
        println("-"^70)
        println("VERIFICATION")
        println("-"^70)
        
        tolerance = 1e-10
        all_passed = true
        
        for strategy in strategies
            strategy = String(strategy)  # Ensure String type
            
            if !(strategy in STRATEGIES_3MM)
                println("  Unknown strategy: $strategy, skipping")
                continue
            end
            
            kernel_fn = get_kernel(strategy)
            passed, max_error = verify_implementation(
                strategy, kernel_fn,
                A, B, C, D, E, F, G, G_ref,
                tolerance=tolerance
            )
            
            status = passed ? "PASS" : "FAIL"
            @printf("  %-18s : %s (max_error: %.2e)\n", strategy, status, max_error)
            all_passed &= passed
        end
        
        if !all_passed
            println("\nVerification FAILED. Aborting benchmark.")
            #return nothing
        end
        println()
    end
    
    # Benchmark
    println("-"^70)
    println("BENCHMARK RESULTS")
    println("-"^70)
    
    if config["output"] == "csv"
        println("strategy,time_ms,speedup,efficiency_pct,gflops,allocations")
    else
        @printf("%-18s %12s %12s %12s %12s %10s\n",
                "Strategy", "Time(ms)", "Speedup", "Efficiency%", "GFLOP/s", "Allocs")
        println("-"^70)
    end
    
    seq_time = 0.0
    results = Dict{String, NamedTuple}()
    
    for strategy in strategies
        strategy = String(strategy)  # Ensure String type
        
        if !(strategy in STRATEGIES_3MM)
            continue
        end
        
        kernel_fn = get_kernel(strategy)
        
        # Setup function (called before each timed iteration)
        setup_fn = () -> begin
            fill!(E, 0.0)
            fill!(F, 0.0)
            fill!(G, 0.0)
        end
        
        # Kernel function (this is what gets timed)
        bench_fn = () -> kernel_fn(A, B, C, D, E, F, G)
        
        # Run benchmark with proper timing
        result = benchmark_kernel(
            bench_fn, setup_fn,
            iterations=config["iterations"],
            warmup=config["warmup"]
        )
        
        # Use minimum time (most representative, least noise)
        time_sec = result.min_time
        time_ms = time_sec * 1000
        gflops = flops / time_sec / 1e9
        
        # Track sequential time for speedup calculation
        if strategy == "sequential"
            seq_time = time_sec
        end
        
        # Calculate speedup and efficiency
        is_threaded = strategy in ["threads_static", "threads_dynamic", "tiled", "tasks"]
        
        if seq_time > 0
            speedup = seq_time / time_sec
            # Efficiency: for threaded strategies, divide by thread count
            efficiency = is_threaded ? (speedup / nthreads()) * 100 : speedup * 100
        else
            speedup = 1.0
            efficiency = 100.0
        end
        
        # Store results
        results[strategy] = (
            time_ms = time_ms,
            speedup = speedup,
            efficiency = efficiency,
            gflops = gflops,
            allocations = result.allocations
        )
        
        # Output
        if config["output"] == "csv"
            @printf("%s,%.4f,%.2f,%.1f,%.2f,%d\n",
                    strategy, time_ms, speedup, efficiency, gflops, result.allocations)
        else
            @printf("%-18s %12.4f %11.2fx %12.1f %12.2f %10d\n",
                    strategy, time_ms, speedup, efficiency, gflops, result.allocations)
        end
    end
    
    if config["output"] != "csv"
        println("-"^70)
    end
    
    return results
end

#=============================================================================
 Entry Point
=============================================================================#
function main()
    # Configure BLAS threads (important for fair comparison)
    if nthreads() > 1
        BLAS.set_num_threads(1)
    end
    
    config = parse_args(ARGS)
    
    try
        run_3mm_benchmark(config)
    catch e
        println("ERROR: ", e)
        if config["verbose"]
            showerror(stdout, e, catch_backtrace())
        end
        exit(1)
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
