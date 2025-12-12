module BenchCore

using Statistics

export TimingResult, benchmark_kernel

struct TimingResult
    times_ns::Vector{Float64}
    allocations::Int
    memory_bytes::Int
end

function benchmark_kernel(
    kernel_fn::Function,
    setup_fn::Function;
    iterations::Int=10,
    warmup_iterations::Int=5
)
    # Warmup phase
    for _ in 1:warmup_iterations
        setup_fn()
        kernel_fn()
    end
    
    # Force GC before timing
    GC.gc()
    
    times = Float64[]
    allocs = 0
    bytes = 0
    
    for i in 1:iterations
        setup_fn()
        
        # Measure allocations on first iteration only
        if i == 1
            stats = @timed kernel_fn()
            push!(times, stats.time * 1e9)
            allocs = Int(stats.gcstats.allocd > 0 ? stats.gcstats.allocd : 0)
            bytes = Int(stats.bytes)
        else
            t0 = time_ns()
            kernel_fn()
            t1 = time_ns()
            push!(times, Float64(t1 - t0))
        end
    end
    
    return TimingResult(times, allocs, bytes)
end

# Quick timing for simple benchmarks
function time_kernel(kernel_fn::Function; samples::Int=5)
    times = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        kernel_fn()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
    end
    return minimum(times) / 1e6  # Return min time in ms
end

end # module
