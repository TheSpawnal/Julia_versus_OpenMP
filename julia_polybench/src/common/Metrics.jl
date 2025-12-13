module Metrics

using Printf
using Statistics
using Dates

export BenchmarkResult, MetricsCollector
export record!, print_results, export_csv

struct BenchmarkResult
    benchmark::String
    dataset::String
    strategy::String
    threads::Int
    workers::Int
    times_ns::Vector{Float64}
    allocations::Int
    flops::Float64
    verified::Bool
end

mutable struct MetricsCollector
    results::Vector{BenchmarkResult}
    timestamp::String
    
    MetricsCollector() = new(BenchmarkResult[], Dates.format(now(), "yyyymmdd_HHMMSS"))
end

function record!(mc::MetricsCollector, result::BenchmarkResult)
    push!(mc.results, result)
end

function print_results(mc::MetricsCollector)
    isempty(mc.results) && return
    
    # Find sequential baseline for speedup calculation
    seq_results = filter(r -> r.strategy == "sequential", mc.results)
    seq_time = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    println()
    println("-"^90)
    @printf("%-16s | %10s | %10s | %10s | %8s | %8s | %6s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("-"^90)
    
    for r in mc.results
        min_t = minimum(r.times_ns) / 1e6
        med_t = median(r.times_ns) / 1e6
        mean_t = mean(r.times_ns) / 1e6
        gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
        
        speedup = seq_time === nothing ? 1.0 : seq_time / minimum(r.times_ns)
        efficiency = speedup / max(r.threads, 1) * 100
        
        @printf("%-16s | %10.3f | %10.3f | %10.3f | %8.2f | %8.2fx | %6.1f\n",
                r.strategy, min_t, med_t, mean_t, gflops, speedup, efficiency)
    end
    println("-"^90)
end

function export_csv(mc::MetricsCollector, filepath::String)
    # Find sequential baseline
    seq_results = filter(r -> r.strategy == "sequential", mc.results)
    seq_time = isempty(seq_results) ? nothing : minimum(seq_results[1].times_ns)
    
    open(filepath, "w") do io
        println(io, "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified")
        
        for r in mc.results
            min_t = minimum(r.times_ns) / 1e6
            med_t = median(r.times_ns) / 1e6
            mean_t = mean(r.times_ns) / 1e6
            std_t = std(r.times_ns) / 1e6
            gflops = r.flops / (minimum(r.times_ns) / 1e9) / 1e9
            
            speedup = seq_time === nothing ? 1.0 : seq_time / minimum(r.times_ns)
            efficiency = speedup / max(r.threads, 1) * 100
            
            @printf(io, "%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%s\n",
                    r.benchmark, r.dataset, r.strategy, r.threads,
                    min_t, med_t, mean_t, std_t, gflops, speedup, efficiency,
                    r.verified ? "PASS" : "FAIL")
        end
    end
    println("Results exported to: $filepath")
end

end # module
