module BenchCore

using Base.Threads
using LinearAlgebra
using Printf

export configure_environment, check_configuration, warmup!, benchmark_kernel
export verify_result, print_system_info

function configure_environment()
    # Configure BLAS threads based on Julia threading
    if nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
    end
end

function check_configuration()
    issues = String[]
    
    # Check for thread/process conflict
    if nthreads() > 1 && nworkers() > 1
        push!(issues, "Both threading ($(nthreads()) threads) and multiprocessing ($(nworkers()) workers) active - may cause contention")
    end
    
    # Check BLAS configuration
    blas_threads = BLAS.get_num_threads()
    if nthreads() > 1 && blas_threads > 1
        push!(issues, "BLAS using $blas_threads threads while Julia has $(nthreads()) threads - potential oversubscription")
    end
    
    return issues
end

function print_system_info(io::IO=stdout)
    println(io, "System Configuration")
    println(io, "-"^40)
    println(io, "Julia version: $(VERSION)")
    println(io, "Threads: $(nthreads())")
    println(io, "Workers: $(nworkers())")
    println(io, "BLAS: $(BLAS.get_config())")
    println(io, "BLAS threads: $(BLAS.get_num_threads())")
    println(io, "CPU cores: $(Sys.CPU_THREADS)")
    println(io, "Total memory: $(round(Sys.total_memory() / 2^30, digits=1)) GB")
    println(io, "Free memory: $(round(Sys.free_memory() / 2^30, digits=1)) GB")
    println(io)
    
    issues = check_configuration()
    if !isempty(issues)
        println(io, "Configuration issues:")
        for issue in issues
            println(io, "  - $issue")
        end
        println(io)
    end
end

# Get number of workers (handle non-distributed case)
function nworkers()
    try
        return length(workers())
    catch
        return 1
    end
end

function workers()
    try
        Main.Distributed.workers()
    catch
        Int[]
    end
end

function warmup!(kernel_fn::Function, args...; iterations::Int=5)
    for _ in 1:iterations
        kernel_fn(args...)
    end
end

struct BenchmarkTiming
    times_ns::Vector{Float64}
    allocations::Int
    memory_bytes::Int
end

function benchmark_kernel(
    kernel_fn::Function,
    setup_fn::Function,
    args...;
    iterations::Int=10,
    warmup_iterations::Int=5,
    gc_between::Bool=true
)
    # Initial setup
    setup_fn(args...)
    
    # Warmup phase (eliminate JIT effects)
    for _ in 1:warmup_iterations
        setup_fn(args...)
        kernel_fn(args...)
    end
    
    # Timed runs
    times_ns = Vector{Float64}(undef, iterations)
    total_allocs = 0
    total_memory = 0
    
    for i in 1:iterations
        setup_fn(args...)
        
        gc_between && GC.gc(false)
        
        # Measure allocations
        stats_before = Base.gc_num()
        
        # High-precision timing
        t_start = time_ns()
        kernel_fn(args...)
        t_end = time_ns()
        
        stats_after = Base.gc_num()
        
        times_ns[i] = Float64(t_end - t_start)
        
        if i == 1
            total_allocs = stats_after.malloc - stats_before.malloc
            total_memory = stats_after.total_allocd - stats_before.total_allocd
        end
    end
    
    return BenchmarkTiming(times_ns, total_allocs, total_memory)
end

function verify_result(
    computed::AbstractArray,
    reference::AbstractArray;
    rtol::Float64=1e-10,
    atol::Float64=1e-14
)
    if size(computed) != size(reference)
        return false, Inf
    end
    
    max_error = 0.0
    for i in eachindex(computed, reference)
        err = abs(computed[i] - reference[i])
        ref_val = abs(reference[i])
        rel_err = ref_val > atol ? err / ref_val : err
        max_error = max(max_error, rel_err)
    end
    
    passed = max_error < rtol
    return passed, max_error
end

function verify_result(computed::Real, reference::Real; rtol::Float64=1e-10, atol::Float64=1e-14)
    err = abs(computed - reference)
    ref_val = abs(reference)
    rel_err = ref_val > atol ? err / ref_val : err
    passed = rel_err < rtol
    return passed, rel_err
end

# Memory estimation helper
function estimate_memory_requirement(arrays::Vector{Tuple{Tuple, DataType}})
    total = 0
    for (dims, T) in arrays
        total += prod(dims) * sizeof(T)
    end
    return total
end

function check_memory_available(required_bytes::Int; safety_margin::Float64=0.8)
    available = Sys.free_memory() * safety_margin
    if required_bytes > available
        required_gb = required_bytes / 2^30
        available_gb = available / 2^30
        error("Insufficient memory: required $(round(required_gb, digits=2)) GB, " *
              "available $(round(available_gb, digits=2)) GB")
    end
end

end # module
