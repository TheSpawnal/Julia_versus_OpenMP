#(warning, very experimental, out of control yet)
#!/usr/bin/env julia
#===============================================================================
# Julia PolyBench - Unified Benchmark Runner
#
# Executes all benchmarks sequentially with consistent configuration.
# Generates combined CSV output for visualization.
#
# USAGE:
#   julia -t 16 scripts/run_all.jl --datasets MEDIUM,LARGE
#   julia -t 16 scripts/run_all.jl --benchmarks 2mm,3mm --datasets LARGE
#
# OUTPUT:
#   results/all_benchmarks_{timestamp}.csv
#===============================================================================

using Printf
using Dates
using Statistics

#=============================================================================
 Configuration
=============================================================================#
const AVAILABLE_BENCHMARKS = ["2mm", "3mm", "cholesky", "correlation", "jacobi2d", "nussinov"]
const AVAILABLE_DATASETS = ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"]

#=============================================================================
 Argument Parsing
=============================================================================#
function parse_args()
    config = Dict{String,Any}(
        "benchmarks" => AVAILABLE_BENCHMARKS,
        "datasets" => ["MEDIUM"],
        "iterations" => 10,
        "warmup" => 5,
        "output_csv" => true
    )
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--benchmarks" && i < length(ARGS)
            config["benchmarks"] = split(ARGS[i+1], ",")
            i += 2
        elseif arg == "--datasets" && i < length(ARGS)
            config["datasets"] = [uppercase(d) for d in split(ARGS[i+1], ",")]
            i += 2
        elseif arg == "--iterations" && i < length(ARGS)
            config["iterations"] = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--warmup" && i < length(ARGS)
            config["warmup"] = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--no-output"
            config["output_csv"] = false
            i += 1
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            i += 1
        end
    end
    
    return config
end

function print_help()
    println("""
Julia PolyBench - Unified Benchmark Runner

USAGE:
    julia -t <threads> scripts/run_all.jl [options]

OPTIONS:
    --benchmarks LIST   Comma-separated: 2mm,3mm,cholesky,correlation,jacobi2d,nussinov
    --datasets LIST     Comma-separated: MINI,SMALL,MEDIUM,LARGE,EXTRALARGE
    --iterations N      Number of timed iterations (default: 10)
    --warmup N          Number of warmup iterations (default: 5)
    --no-output         Skip CSV file generation
    --help              Show this help

EXAMPLES:
    julia -t 16 scripts/run_all.jl --datasets MEDIUM,LARGE
    julia -t 8 scripts/run_all.jl --benchmarks 2mm,3mm --datasets LARGE
    julia -t 32 scripts/run_all.jl --benchmarks cholesky --iterations 20
    """)
end

#=============================================================================
 Benchmark Execution
=============================================================================#
function run_benchmark_script(benchmark::String, dataset::String, iterations::Int, warmup::Int)
    script_path = joinpath(@__DIR__, "run_$(benchmark).jl")
    
    if !isfile(script_path)
        @warn "Script not found: $script_path"
        return nothing
    end
    
    # Build command
    cmd = `julia -t $(Threads.nthreads()) $script_path --dataset $dataset --iterations $iterations --warmup $warmup --output csv`
    
    println("\n>>> Running: $benchmark ($dataset)")
    
    try
        run(cmd)
        
        # Find the most recent results file
        results_dir = joinpath(dirname(@__DIR__), "results")
        pattern = Regex("$(benchmark)_$(dataset)_.*\\.csv")
        
        files = filter(f -> occursin(pattern, f), readdir(results_dir))
        if !isempty(files)
            sort!(files, rev=true)  # Most recent first (by timestamp in filename)
            return joinpath(results_dir, files[1])
        end
    catch e
        @error "Failed to run $benchmark: $e"
    end
    
    return nothing
end

#=============================================================================
 CSV Aggregation
=============================================================================#
function aggregate_results(result_files::Vector{String}, output_path::String)
    all_lines = String[]
    header_seen = false
    
    for filepath in result_files
        if !isfile(filepath)
            continue
        end
        
        lines = readlines(filepath)
        if isempty(lines)
            continue
        end
        
        # Only include header once
        if !header_seen
            push!(all_lines, lines[1])
            header_seen = true
        end
        
        # Add data lines
        for line in lines[2:end]
            if !isempty(strip(line))
                push!(all_lines, line)
            end
        end
    end
    
    if isempty(all_lines)
        return false
    end
    
    open(output_path, "w") do io
        for line in all_lines
            println(io, line)
        end
    end
    
    return true
end

#=============================================================================
 Main
=============================================================================#
function main()
    config = parse_args()
    
    println("="^70)
    println("Julia PolyBench - Unified Benchmark Runner")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("Benchmarks: $(join(config["benchmarks"], ", "))")
    println("Datasets: $(join(config["datasets"], ", "))")
    println("Iterations: $(config["iterations"])")
    println("Warmup: $(config["warmup"])")
    println("="^70)
    
    # Ensure results directory exists
    results_dir = joinpath(dirname(@__DIR__), "results")
    mkpath(results_dir)
    
    # Track result files
    result_files = String[]
    
    # Run each benchmark/dataset combination
    total_runs = length(config["benchmarks"]) * length(config["datasets"])
    current_run = 0
    
    for benchmark in config["benchmarks"]
        for dataset in config["datasets"]
            current_run += 1
            println("\n[$current_run/$total_runs] Running $benchmark - $dataset")
            
            result_file = run_benchmark_script(
                benchmark, 
                dataset, 
                config["iterations"], 
                config["warmup"]
            )
            
            if result_file !== nothing
                push!(result_files, result_file)
            end
        end
    end
    
    # Aggregate results
    if config["output_csv"] && !isempty(result_files)
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        combined_path = joinpath(results_dir, "all_benchmarks_$(timestamp).csv")
        
        if aggregate_results(result_files, combined_path)
            println("\n" * "="^70)
            println("All benchmarks complete!")
            println("Combined results: $combined_path")
            println("Individual results: $(length(result_files)) files")
            println("="^70)
        end
    end
    
    println("\nGenerate visualizations with:")
    println("  python3 visualize_benchmarks.py results/all_benchmarks_*.csv")
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
