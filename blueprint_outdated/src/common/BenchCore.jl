

module BenchCore
#=
Benchmark Core - Timing Infrastructure for Julia HPC Benchmarks

Design Principles:
1. Separate JIT warmup from measurement
2. Setup/teardown outside timed region
3. Minimal overhead timing with time_ns()
4. Statistics: min, median, mean, std
5. Optional allocation tracking (first iteration only)

Usage:
    result = benchmark_kernel(
        kernel_fn,       # Function to benchmark
        setup_fn;        # Called before each iteration (not timed)
        iterations=10,
        warmup=5
    )
    
    # result.min_time, result.median_time, result.times_ns, etc.
=#

using Statistics

export TimingResult, benchmark_kernel, time_kernel_simple

#=============================================================================
 Result Structure
=============================================================================#
struct TimingResult
    times_ns::Vector{Float64}      # All sample times in nanoseconds
    min_time::Float64              # Minimum time in seconds
    median_time::Float64           # Median time in seconds
    mean_time::Float64             # Mean time in seconds
    std_time::Float64              # Standard deviation in seconds
    allocations::Int               # Allocations (from first iteration)
    memory_bytes::Int              # Memory allocated (from first iteration)
end

#=============================================================================
 Primary Benchmark Function
 
 Key features:
 - JIT warmup phase (not included in measurements)
 - Setup function called before each iteration (not timed)
 - GC forced between warmup and measurement
 - Uses time_ns() for minimal overhead
 - Allocation tracking on first iteration only
=============================================================================#
function benchmark_kernel(
    kernel_fn::Function,
    setup_fn::Function;
    iterations::Int=10,
    warmup::Int=5
)
    # Phase 1: JIT Warmup (not measured)
    # This compiles the kernel and setup functions
    for _ in 1:warmup
        setup_fn()
        kernel_fn()
    end
    
    # Force GC before measurement phase
    GC.gc()
    GC.gc()  # Double GC to clear finalizers
    
    # Pre-allocate timing array
    times = Vector{Float64}(undef, iterations)
    allocs = 0
    bytes = 0
    
    # Phase 2: Measurement
    for i in 1:iterations
        # Setup (not timed)
        setup_fn()
        
        # Track allocations on first iteration only
        # (subsequent iterations should have zero allocations)
        if i == 1
            GC.gc()
            stats = @timed kernel_fn()
            times[i] = stats.time * 1e9  # Convert to nanoseconds
            allocs = Int(stats.gcstats.allocd > 0 ? stats.gcstats.allocd : 0)
            bytes = Int(stats.bytes)
        else
            # Pure timing with minimal overhead
            t0 = time_ns()
            kernel_fn()
            t1 = time_ns()
            times[i] = Float64(t1 - t0)
        end
    end
    
    # Convert to seconds for statistics
    times_sec = times ./ 1e9
    
    return TimingResult(
        times,
        minimum(times_sec),
        median(times_sec),
        mean(times_sec),
        std(times_sec),
        allocs,
        bytes
    )
end

#=============================================================================
 Simple Timing Function
 
 For quick benchmarks without setup/teardown.
 Returns minimum time in milliseconds.
=============================================================================#
function time_kernel_simple(kernel_fn::Function; samples::Int=5, warmup::Int=3)
    # Warmup
    for _ in 1:warmup
        kernel_fn()
    end
    
    GC.gc()
    
    # Measure
    times = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        kernel_fn()
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
    end
    
    return minimum(times) / 1e6  # Return min time in milliseconds
end

#=============================================================================
 Helper: Check for Zero Allocations
 
 Usage:
    allocs = check_allocations() do
        kernel_fn(args...)
    end
    @assert allocs == 0 "Kernel should not allocate!"
=============================================================================#
function check_allocations(f::Function)
    # First call for JIT
    f()
    GC.gc()
    
    # Measure allocations
    return @allocated f()
end

end # module BenchCore
