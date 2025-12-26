module Metrics
#=
Metrics Collection Module for Julia PolyBench Benchmarks

Provides:
- BenchmarkResult struct for storing results
- MetricsCollector for aggregating multiple runs
- Strategy-aware efficiency calculation
- CSV export for analysis

Key design points:
- Efficiency calculation differs by strategy type:
  - Non-threaded (seq, blas): efficiency = speedup * 100
  - Threaded (threads, tiled, tasks): efficiency = (speedup / nthreads) * 100
=#

using Printf
using Statistics
using Dates

export BenchmarkResult, MetricsCollector
export record!, print_results, export_csv
export compute_efficiency

#=============================================================================
 Strategy Classification
 Non-threaded strategies should have efficiency = speedup * 100
 Threaded strategies should have efficiency = (speedup / threads) * 100
=============================================================================#
const NON_THREADED_STRATEGIES = Set([
    "sequential", "seq", "simd", "blas", "colmajor"
])

const THREADED_STRATEGIES = Set([
    "threads", "threads_static", "threads_dynamic", 
    "tiled", "blocked", "tasks", "wavefront"
])

#=============================================================================
 Result Structure
=============================================================================#
struct BenchmarkResult
    benchmark::String       # e.g., "2MM", "3MM", "Cholesky"
    dataset::String         # e.g., "MEDIUM", "LARGE"
    strategy::String        # e.g., "sequential", "threads_static"
    threads::Int            # Number of Julia threads
    workers::Int            # Number of distributed workers (0 if none)
    times_ns::Vector{Float64}  # All timing samples in nanoseconds
    allocations::Int        # Memory allocations
    flops::Float64          # Total FLOPs for the computation
    verified::Bool          # Whether result was verified correct
end

# Convenience constructor for common case
function BenchmarkResult(
    benchmark::String, 
    dataset::String, 
    strategy::String,
    threads::Int,
    times_ns::Vector{Float64},
    allocations::Int,
    flops::Float64;
    workers::Int=0,
    verified::Bool=true
)
    return BenchmarkResult(
        benchmark, dataset, strategy, threads, workers,
        times_ns, allocations, flops, verified
    )
end

#=============================================================================
 Metrics Collector
=============================================================================#
mutable struct MetricsCollector
    results::Vector{BenchmarkResult}
    timestamp::String
    
    MetricsCollector() = new(BenchmarkResult[], Dates.format(now(), "yyyymmdd_HHMMSS"))
end

function record!(mc::MetricsCollector, result::BenchmarkResult)
    push!(mc.results, result)
end

#=============================================================================
 Efficiency Calculation
 
 This is a critical function for correct performance analysis:
 - Sequential/SIMD/BLAS: efficiency = speedup * 100 (should be ~100% for seq)
 - Threaded strategies: efficiency = (speedup / nthreads) * 100
 
 Example with 8 threads:
 - Sequential: speedup=1.0, efficiency=100%
 - Threads achieving 4x speedup: efficiency = (4/8)*100 = 50%
 - Perfect scaling (8x speedup): efficiency = (8/8)*100 = 100%
=============================================================================#
function compute_efficiency(strategy::AbstractString, speedup::Float64, nthreads::Int)
    strategy_lower = lowercase(String(strategy))
    
    if strategy_lower in NON_THREADED_STRATEGIES
        # Non-threaded: efficiency = speedup * 100
        # Sequential baseline should be 100%
        return speedup * 100.0
    else
        # Threaded strategies: efficiency = (speedup / threads) * 100
        # Perfect scaling would be 100%
        return (speedup / max(nthreads, 1)) * 100.0
    end
end

# Check if strategy is threaded
function is_threaded_strategy(strategy::AbstractString)
    return lowercase(String(strategy)) in THREADED_STRATEGIES
end

#=============================================================================
 Print Results
=============================================================================#
function print_results(mc::MetricsCollector)
    isempty(mc.results) && return
    
    # Find sequential baseline for speedup calculation
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    println()
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for r in mc.results
        min_t = minimum(r.times_ns) / 1e6     # ms
        med_t = median(r.times_ns) / 1e6      # ms
        mean_t = mean(r.times_ns) / 1e6       # ms
        gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
        
        speedup = seq_time === nothing ? 1.0 : seq_time / minimum(r.times_ns)
        efficiency = compute_efficiency(r.strategy, speedup, r.threads)
        
        @printf("%-16s | %10.3f | %10.3f | %10.3f | %8.2f | %7.2fx | %6.1f\n",
                r.strategy, min_t, med_t, mean_t, gflops, speedup, efficiency)
    end
    println("-"^90)
end

#=============================================================================
 CSV Export
=============================================================================#
function export_csv(mc::MetricsCollector, filepath::String)
    # Find sequential baseline
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    open(filepath, "w") do io
        # Header
        println(io, "benchmark,dataset,strategy,threads,workers,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,allocations,verified")
        
        for r in mc.results
            min_t = minimum(r.times_ns) / 1e6
            med_t = median(r.times_ns) / 1e6
            mean_t = mean(r.times_ns) / 1e6
            std_t = length(r.times_ns) > 1 ? std(r.times_ns) / 1e6 : 0.0
            gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
            
            speedup = seq_time === nothing ? 1.0 : seq_time / minimum(r.times_ns)
            efficiency = compute_efficiency(r.strategy, speedup, r.threads)
            
            @printf(io, "%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.1f,%d,%s\n",
                    r.benchmark, r.dataset, r.strategy, r.threads, r.workers,
                    min_t, med_t, mean_t, std_t, gflops, speedup, efficiency,
                    r.allocations, r.verified ? "PASS" : "FAIL")
        end
    end
    println("Results exported to: $filepath")
end

#=============================================================================
 JSON Export (for programmatic analysis)
=============================================================================#
function export_json(mc::MetricsCollector, filepath::String)
    seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], mc.results)
    seq_time = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"timestamp\": \"$(mc.timestamp)\",")
        println(io, "  \"results\": [")
        
        for (idx, r) in enumerate(mc.results)
            min_t = minimum(r.times_ns) / 1e6
            gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
            speedup = seq_time === nothing ? 1.0 : seq_time / minimum(r.times_ns)
            efficiency = compute_efficiency(r.strategy, speedup, r.threads)
            
            println(io, "    {")
            println(io, "      \"benchmark\": \"$(r.benchmark)\",")
            println(io, "      \"dataset\": \"$(r.dataset)\",")
            println(io, "      \"strategy\": \"$(r.strategy)\",")
            println(io, "      \"threads\": $(r.threads),")
            println(io, "      \"min_time_ms\": $(round(min_t, digits=4)),")
            println(io, "      \"gflops\": $(round(gflops, digits=2)),")
            println(io, "      \"speedup\": $(round(speedup, digits=2)),")
            println(io, "      \"efficiency\": $(round(efficiency, digits=1)),")
            println(io, "      \"allocations\": $(r.allocations),")
            println(io, "      \"verified\": $(r.verified)")
            println(io, idx < length(mc.results) ? "    }," : "    }")
        end
        
        println(io, "  ]")
        println(io, "}")
    end
    println("Results exported to: $filepath")
end

end # module Metrics
