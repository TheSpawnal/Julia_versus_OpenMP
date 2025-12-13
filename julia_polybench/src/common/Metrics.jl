# Corrected Metrics.jl for PolyBench Julia
# Key Fix: Efficiency calculation now properly handles sequential and non-threaded strategies

module Metrics

using Printf
using Statistics
using Dates

export BenchmarkResult, MetricsCollector
export compute_statistics, compute_speedup, compute_efficiency, compute_gflops
export print_summary, export_csv, export_json

struct BenchmarkResult
    benchmark::String
    dataset::String
    strategy::String
    threads::Int
    workers::Int
    times_ns::Vector{Float64}
    flops::Float64
    allocations::Int
    memory_bytes::Int
    verified::Bool
end

mutable struct MetricsCollector
    results::Vector{BenchmarkResult}
    timestamp::String
    
    function MetricsCollector()
        new(BenchmarkResult[], Dates.format(now(), "yyyymmdd_HHMMSS"))
    end
end

function add_result!(mc::MetricsCollector, result::BenchmarkResult)
    push!(mc.results, result)
end

# Statistics computation
function compute_statistics(times_ns::Vector{Float64})
    return (
        min = minimum(times_ns) / 1e6,      # ms
        max = maximum(times_ns) / 1e6,      # ms
        median = median(times_ns) / 1e6,    # ms
        mean = mean(times_ns) / 1e6,        # ms
        std = length(times_ns) > 1 ? std(times_ns) / 1e6 : 0.0  # ms
    )
end

# Speedup relative to baseline
function compute_speedup(result::BenchmarkResult, baseline_ns::Float64)
    if baseline_ns <= 0.0
        return 1.0
    end
    return baseline_ns / minimum(result.times_ns)
end

# CORRECTED EFFICIENCY CALCULATION
# Key insight: Efficiency is only meaningful for parallel strategies that use Julia threads
function compute_efficiency(speedup::Float64, strategy::String, threads::Int)
    # Strategies that don't use Julia threading
    non_threaded_strategies = [
        "sequential", "seq",
        "simd",
        "blas",
        "colmajor"  # This is typically a memory optimization, not parallelization
    ]
    
    # Check if this is a non-threaded strategy
    strategy_lower = lowercase(strategy)
    
    if any(s -> occursin(s, strategy_lower), non_threaded_strategies)
        # For non-threaded strategies, efficiency is 100% (they use what they have optimally)
        # Or return NaN to indicate N/A
        return 100.0
    end
    
    # For threaded strategies
    if threads <= 1
        return 100.0
    end
    
    # Parallel efficiency = (speedup / num_threads) * 100
    return (speedup / threads) * 100.0
end

# Alternative: More explicit version with thread counting
function compute_efficiency_v2(speedup::Float64, strategy::String, 
                               julia_threads::Int, strategy_threads::Int)
    # strategy_threads = number of threads the strategy actually uses
    # For sequential: strategy_threads = 1
    # For @threads: strategy_threads = julia_threads
    # For BLAS: strategy_threads = 1 (BLAS uses its own pool)
    
    if strategy_threads <= 1
        return 100.0
    end
    
    return (speedup / strategy_threads) * 100.0
end

# Determine how many threads a strategy actually uses
function get_strategy_thread_count(strategy::String, julia_threads::Int)
    strategy_lower = lowercase(strategy)
    
    if occursin("sequential", strategy_lower) || strategy_lower == "seq"
        return 1
    elseif occursin("simd", strategy_lower)
        return 1  # SIMD is vectorization, not threading
    elseif occursin("blas", strategy_lower)
        return 1  # We set BLAS threads to 1 in multithreaded Julia
    elseif occursin("thread", strategy_lower)
        return julia_threads
    elseif occursin("tiled", strategy_lower)
        # Tiled might or might not use threads - check implementation
        # If the tiled impl uses @threads, return julia_threads
        # If pure tiled without threading, return 1
        return 1  # Conservative: assume non-threaded unless suffixed
    elseif occursin("task", strategy_lower)
        return julia_threads  # Task-based uses all threads
    elseif occursin("distributed", strategy_lower) || occursin("dist", strategy_lower)
        return 1  # Distributed uses workers, not threads for efficiency calc
    else
        return julia_threads  # Default assumption
    end
end

# GFLOP/s computation
function compute_gflops(result::BenchmarkResult)
    if result.flops <= 0.0
        return 0.0
    end
    min_time_s = minimum(result.times_ns) / 1e9
    return result.flops / (min_time_s * 1e9)
end

# Print formatted summary
function print_summary(mc::MetricsCollector; io::IO=stdout)
    println(io, "\n", "="^90)
    println(io, "BENCHMARK RESULTS SUMMARY")
    println(io, "="^90)
    println(io, "Timestamp: $(mc.timestamp)")
    println(io, "Total benchmarks: $(length(mc.results))")
    
    for bench in unique(r.benchmark for r in mc.results)
        bench_results = filter(r -> r.benchmark == bench, mc.results)
        seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], bench_results)
        baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
        
        dataset = bench_results[1].dataset
        println(io, "\n", "-"^90)
        @printf(io, "%-15s | Dataset: %-10s | Baseline: %.3f ms\n", 
                uppercase(bench), dataset, baseline_ns / 1e6)
        println(io, "-"^90)
        
        # Header
        @printf(io, "%-16s | %5s | %10s | %10s | %10s | %8s | %8s | %7s\n",
                "Strategy", "Thr", "Min(ms)", "Med(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
        println(io, "-"^90)
        
        for r in bench_results
            stats = compute_statistics(r.times_ns)
            speedup = compute_speedup(r, baseline_ns)
            
            # CORRECTED: Use strategy-aware efficiency calculation
            strategy_threads = get_strategy_thread_count(r.strategy, r.threads)
            eff = compute_efficiency(speedup, r.strategy, strategy_threads)
            
            gflops = compute_gflops(r)
            
            status = r.verified ? "" : " [!]"
            
            @printf(io, "%-16s | %5d | %10.3f | %10.3f | %10.3f | %8.2f | %7.2fx | %6.1f%s\n",
                    r.strategy, r.threads, stats.min, stats.median, stats.mean,
                    gflops, speedup, eff, status)
        end
    end
    println(io)
end

# CSV export
function export_csv(mc::MetricsCollector, filepath::String)
    open(filepath, "w") do io
        # Header
        println(io, "benchmark,dataset,strategy,julia_threads,strategy_threads,workers,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,allocations,memory_mb,verified")
        
        for bench in unique(r.benchmark for r in mc.results)
            bench_results = filter(r -> r.benchmark == bench, mc.results)
            seq_results = filter(r -> lowercase(r.strategy) in ["sequential", "seq"], bench_results)
            baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
            
            for r in bench_results
                stats = compute_statistics(r.times_ns)
                speedup = compute_speedup(r, baseline_ns)
                strategy_threads = get_strategy_thread_count(r.strategy, r.threads)
                eff = compute_efficiency(speedup, r.strategy, strategy_threads)
                gflops = compute_gflops(r)
                memory_mb = r.memory_bytes / (1024^2)
                
                @printf(io, "%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.2f,%d,%.3f,%s\n",
                        r.benchmark, r.dataset, r.strategy, 
                        r.threads, strategy_threads, r.workers,
                        stats.min, stats.median, stats.mean, stats.std,
                        gflops, speedup, eff, r.allocations, memory_mb,
                        r.verified ? "true" : "false")
            end
        end
    end
    println("Results exported to: $filepath")
end

# JSON export
function export_json(mc::MetricsCollector, filepath::String)
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"timestamp\": \"$(mc.timestamp)\",")
        println(io, "  \"results\": [")
        
        for (idx, r) in enumerate(mc.results)
            seq_results = filter(x -> x.benchmark == r.benchmark && 
                                      lowercase(x.strategy) in ["sequential", "seq"], mc.results)
            baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
            
            stats = compute_statistics(r.times_ns)
            speedup = compute_speedup(r, baseline_ns)
            strategy_threads = get_strategy_thread_count(r.strategy, r.threads)
            eff = compute_efficiency(speedup, r.strategy, strategy_threads)
            gflops = compute_gflops(r)
            
            comma = idx < length(mc.results) ? "," : ""
            
            println(io, "    {")
            println(io, "      \"benchmark\": \"$(r.benchmark)\",")
            println(io, "      \"dataset\": \"$(r.dataset)\",")
            println(io, "      \"strategy\": \"$(r.strategy)\",")
            println(io, "      \"julia_threads\": $(r.threads),")
            println(io, "      \"strategy_threads\": $(strategy_threads),")
            println(io, "      \"workers\": $(r.workers),")
            @printf(io, "      \"min_ms\": %.4f,\n", stats.min)
            @printf(io, "      \"median_ms\": %.4f,\n", stats.median)
            @printf(io, "      \"mean_ms\": %.4f,\n", stats.mean)
            @printf(io, "      \"std_ms\": %.4f,\n", stats.std)
            @printf(io, "      \"gflops\": %.3f,\n", gflops)
            @printf(io, "      \"speedup\": %.3f,\n", speedup)
            @printf(io, "      \"efficiency\": %.2f,\n", eff)
            println(io, "      \"allocations\": $(r.allocations),")
            @printf(io, "      \"memory_mb\": %.3f,\n", r.memory_bytes / (1024^2))
            println(io, "      \"verified\": $(r.verified)")
            println(io, "    }$comma")
        end
        
        println(io, "  ]")
        println(io, "}")
    end
    println("Results exported to: $filepath")
end

end # module
