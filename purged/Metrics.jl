module Metrics

using Statistics
using Printf
using Dates

export BenchmarkResult, MetricsCollector
export record!, compute_statistics, print_results, export_csv, export_json

struct BenchmarkResult
    benchmark::String
    dataset::String
    strategy::String
    threads::Int
    workers::Int
    times_ns::Vector{Float64}
    allocations::Int
    memory_bytes::Int
    flops::Int64
    verified::Bool
end

mutable struct MetricsCollector
    results::Vector{BenchmarkResult}
    baseline_time::Float64
    timestamp::String
    
    MetricsCollector() = new(BenchmarkResult[], 0.0, Dates.format(now(), "yyyymmdd_HHMMSS"))
end

function record!(mc::MetricsCollector, result::BenchmarkResult)
    push!(mc.results, result)
    if result.strategy == "sequential" && mc.baseline_time == 0.0
        mc.baseline_time = minimum(result.times_ns)
    end
end

function compute_statistics(times_ns::Vector{Float64})
    times_ms = times_ns ./ 1e6
    (
        min = minimum(times_ms),
        max = maximum(times_ms),
        median = median(times_ms),
        mean = mean(times_ms),
        std = length(times_ms) > 1 ? std(times_ms) : 0.0,
        samples = length(times_ms)
    )
end

function compute_speedup(result::BenchmarkResult, baseline_ns::Float64)
    if baseline_ns > 0
        baseline_ns / minimum(result.times_ns)
    else
        1.0
    end
end

function compute_efficiency(speedup::Float64, threads::Int)
    threads > 0 ? (speedup / threads) * 100.0 : 0.0
end

function compute_gflops(result::BenchmarkResult)
    min_time_s = minimum(result.times_ns) / 1e9
    min_time_s > 0 ? (result.flops / 1e9) / min_time_s : 0.0
end

function print_results(mc::MetricsCollector; io::IO=stdout)
    println(io, "="^100)
    println(io, "BENCHMARK RESULTS - $(mc.timestamp)")
    println(io, "="^100)
    println(io)
    
    # Group by benchmark
    benchmarks = unique(r.benchmark for r in mc.results)
    
    for bench in benchmarks
        bench_results = filter(r -> r.benchmark == bench, mc.results)
        if isempty(bench_results)
            continue
        end
        
        # Get baseline
        seq_results = filter(r -> r.strategy == "sequential", bench_results)
        baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
        
        dataset = bench_results[1].dataset
        println(io, "-"^100)
        @printf(io, "%-15s | Dataset: %-10s | Baseline: %.3f ms\n", 
                uppercase(bench), dataset, baseline_ns / 1e6)
        println(io, "-"^100)
        
        # Header
        @printf(io, "%-20s | %6s | %10s | %10s | %10s | %8s | %8s | %8s | %8s\n",
                "Strategy", "Thr", "Min(ms)", "Med(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)", "Allocs")
        println(io, "-"^100)
        
        for r in bench_results
            stats = compute_statistics(r.times_ns)
            speedup = compute_speedup(r, baseline_ns)
            eff = compute_efficiency(speedup, r.threads)
            gflops = compute_gflops(r)
            
            status = r.verified ? "" : " [!]"
            
            @printf(io, "%-20s | %6d | %10.3f | %10.3f | %10.3f | %8.2f | %8.2fx | %7.1f | %8d%s\n",
                    r.strategy, r.threads, stats.min, stats.median, stats.mean,
                    gflops, speedup, eff, r.allocations, status)
        end
        println(io)
    end
end

function export_csv(mc::MetricsCollector, filepath::String)
    open(filepath, "w") do io
        # Header
        println(io, "benchmark,dataset,strategy,threads,workers,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,allocations,memory_mb,verified")
        
        for bench in unique(r.benchmark for r in mc.results)
            bench_results = filter(r -> r.benchmark == bench, mc.results)
            seq_results = filter(r -> r.strategy == "sequential", bench_results)
            baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
            
            for r in bench_results
                stats = compute_statistics(r.times_ns)
                speedup = compute_speedup(r, baseline_ns)
                eff = compute_efficiency(speedup, r.threads)
                gflops = compute_gflops(r)
                memory_mb = r.memory_bytes / (1024^2)
                
                @printf(io, "%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.2f,%d,%.3f,%s\n",
                        r.benchmark, r.dataset, r.strategy, r.threads, r.workers,
                        stats.min, stats.median, stats.mean, stats.std,
                        gflops, speedup, eff, r.allocations, memory_mb,
                        r.verified ? "true" : "false")
            end
        end
    end
    println("Results exported to: $filepath")
end

function export_json(mc::MetricsCollector, filepath::String)
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"timestamp\": \"$(mc.timestamp)\",")
        println(io, "  \"results\": [")
        
        for (idx, r) in enumerate(mc.results)
            seq_results = filter(x -> x.benchmark == r.benchmark && x.strategy == "sequential", mc.results)
            baseline_ns = isempty(seq_results) ? 0.0 : minimum(seq_results[1].times_ns)
            
            stats = compute_statistics(r.times_ns)
            speedup = compute_speedup(r, baseline_ns)
            eff = compute_efficiency(speedup, r.threads)
            gflops = compute_gflops(r)
            
            comma = idx < length(mc.results) ? "," : ""
            
            println(io, "    {")
            println(io, "      \"benchmark\": \"$(r.benchmark)\",")
            println(io, "      \"dataset\": \"$(r.dataset)\",")
            println(io, "      \"strategy\": \"$(r.strategy)\",")
            println(io, "      \"threads\": $(r.threads),")
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
