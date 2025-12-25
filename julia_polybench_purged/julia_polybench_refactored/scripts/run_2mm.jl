#!/usr/bin/env julia
#=
2MM Benchmark Runner - REFACTORED VERSION
Computation: D = alpha * A * B * C + beta * D

USAGE:
  julia -t 8 run_2mm.jl --dataset MEDIUM
  julia -t 16 run_2mm.jl --dataset LARGE --output csv
  julia -t 8 run_2mm.jl --strategies sequential,threads_static,blas

DAS-5 SLURM:
  srun -N 1 -c 16 --time=01:00:00 julia -t 16 run_2mm.jl --dataset LARGE

For EXTRALARGE datasets (longer runtime):
  sbatch --time=02:00:00 das5_extralarge.slurm 2mm

Flame Graph Profiling (add --profile flag):
  julia -t 8 run_2mm.jl --dataset LARGE --profile
=#

using LinearAlgebra
using Statistics
using Printf
using Dates

#=============================================================================
 BLAS Configuration - MUST happen before any BLAS operations
=============================================================================#
function configure_blas_threads(;for_blas_benchmark::Bool=false)
    if for_blas_benchmark
        BLAS.set_num_threads(Sys.CPU_THREADS)
    elseif Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
    end
end

configure_blas_threads()

#=============================================================================
 Dataset Sizes - PolyBench Standard
=============================================================================#
const DATASET_SIZES = Dict(
    "MINI"       => (ni=16,   nj=18,   nk=22,   nl=24),
    "SMALL"      => (ni=40,   nj=50,   nk=70,   nl=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=210,  nl=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

#=============================================================================
 FLOPs and Memory Calculations
=============================================================================#
function flops_2mm(ni, nj, nk, nl)
    return 2.0 * ni * nj * nk + 2.0 * ni * nl * nj + Float64(ni * nl)
end

function memory_bytes_2mm(ni, nj, nk, nl)
    return (ni*nk + nk*nj + ni*nj + nj*nl + 2*ni*nl) * sizeof(Float64)
end

#=============================================================================
 Strategy Classification
=============================================================================#
const PARALLEL_STRATEGIES = Set(["threads_static", "threads_dynamic", "tiled", "tasks"])
const NON_PARALLEL_STRATEGIES = Set(["sequential", "blas"])
const STRATEGY_ORDER = ["sequential", "threads_static", "threads_dynamic", "tiled", "blas", "tasks"]

function is_parallel_strategy(strategy::String)::Bool
    return lowercase(strategy) in PARALLEL_STRATEGIES
end

function compute_parallel_efficiency(strategy::String, speedup::Float64, threads::Int)::Union{Float64, Nothing}
    if !is_parallel_strategy(strategy)
        return nothing  # Not applicable
    end
    if threads <= 0
        return nothing
    end
    return (speedup / threads) * 100.0
end

#=============================================================================
 Local BenchmarkResult Struct
=============================================================================#
struct BenchmarkResult
    strategy::String
    times_ms::Vector{Float64}
    verified::Bool
    max_error::Float64
    allocations::Int
end

#=============================================================================
 Data Initialization (PolyBench standard)
=============================================================================#
function init_2mm!(alpha::Ref{Float64}, beta::Ref{Float64},
                   A::Matrix{Float64}, B::Matrix{Float64},
                   tmp::Matrix{Float64}, C::Matrix{Float64},
                   D::Matrix{Float64})
    ni, nk = size(A)
    nj = size(B, 2)
    nl = size(C, 2)
    
    alpha[] = 1.5
    beta[] = 1.2
    
    @inbounds for j in 1:nk, i in 1:ni
        A[i, j] = ((i - 1) * (j - 1) + 1) % ni / Float64(ni)
    end
    
    @inbounds for j in 1:nj, i in 1:nk
        B[i, j] = (i - 1) * j % nj / Float64(nj)
    end
    
    @inbounds for j in 1:nl, i in 1:nj
        C[i, j] = ((i - 1) * (j + 2) + 1) % nl / Float64(nl)
    end
    
    @inbounds for j in 1:nl, i in 1:ni
        D[i, j] = (i - 1) * (j + 1) % nk / Float64(nk)
    end
    
    fill!(tmp, 0.0)
    return nothing
end

#=============================================================================
 Kernel Implementations
=============================================================================#

# Sequential baseline
function kernel_2mm_seq!(alpha, beta, A, B, tmp, C, D)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    @inbounds for j in 1:nj
        for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    @inbounds for j in 1:nl
        @simd for i in 1:ni
            D[i, j] *= beta
        end
        for k in 1:nj
            c_kj = C[k, j]
            @simd for i in 1:ni
                D[i, j] += tmp[i, k] * c_kj
            end
        end
    end
    return nothing
end

# Threads static
function kernel_2mm_threads_static!(alpha, beta, A, B, tmp, C, D)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    Threads.@threads :static for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    Threads.@threads :static for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    return nothing
end

# Threads dynamic
function kernel_2mm_threads_dynamic!(alpha, beta, A, B, tmp, C, D)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    Threads.@threads :dynamic for j in 1:nj
        @inbounds for k in 1:nk
            b_kj = alpha * B[k, j]
            @simd for i in 1:ni
                tmp[i, j] += A[i, k] * b_kj
            end
        end
    end
    
    Threads.@threads :dynamic for j in 1:nl
        @inbounds begin
            @simd for i in 1:ni
                D[i, j] *= beta
            end
            for k in 1:nj
                c_kj = C[k, j]
                @simd for i in 1:ni
                    D[i, j] += tmp[i, k] * c_kj
                end
            end
        end
    end
    return nothing
end

# Tiled
function kernel_2mm_tiled!(alpha, beta, A, B, tmp, C, D; tile_size::Int=64)
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    ts = tile_size
    
    Threads.@threads :static for jj in 1:ts:nj
        j_end = min(jj + ts - 1, nj)
        @inbounds for kk in 1:ts:nk
            k_end = min(kk + ts - 1, nk)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        b_kj = alpha * B[k, j]
                        @simd for i in ii:i_end
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    Threads.@threads :static for jj in 1:ts:nl
        j_end = min(jj + ts - 1, nl)
        @inbounds for j in jj:j_end
            @simd for i in 1:ni
                D[i, j] *= beta
            end
        end
        @inbounds for kk in 1:ts:nj
            k_end = min(kk + ts - 1, nj)
            for ii in 1:ts:ni
                i_end = min(ii + ts - 1, ni)
                for j in jj:j_end
                    for k in kk:k_end
                        c_kj = C[k, j]
                        @simd for i in ii:i_end
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# BLAS
function kernel_2mm_blas!(alpha, beta, A, B, tmp, C, D)
    mul!(tmp, A, B, alpha, 0.0)
    mul!(D, tmp, C, 1.0, beta)
    return nothing
end

# Tasks
function kernel_2mm_tasks!(alpha, beta, A, B, tmp, C, D; num_tasks::Int=Threads.nthreads())
    ni, nk = size(A)
    _, nj = size(B)
    _, nl = size(C)
    
    chunk_j = cld(nj, num_tasks)
    @sync begin
        for t in 1:num_tasks
            j_start = (t - 1) * chunk_j + 1
            j_end = min(t * chunk_j, nj)
            j_start > nj && continue
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    for k in 1:nk
                        b_kj = alpha * B[k, j]
                        @simd for i in 1:ni
                            tmp[i, j] += A[i, k] * b_kj
                        end
                    end
                end
            end
        end
    end
    
    chunk_l = cld(nl, num_tasks)
    @sync begin
        for t in 1:num_tasks
            j_start = (t - 1) * chunk_l + 1
            j_end = min(t * chunk_l, nl)
            j_start > nl && continue
            Threads.@spawn begin
                @inbounds for j in j_start:j_end
                    @simd for i in 1:ni
                        D[i, j] *= beta
                    end
                    for k in 1:nj
                        c_kj = C[k, j]
                        @simd for i in 1:ni
                            D[i, j] += tmp[i, k] * c_kj
                        end
                    end
                end
            end
        end
    end
    return nothing
end

#=============================================================================
 Get Kernel by Name
=============================================================================#
function get_kernel(name::String)
    name_lower = lowercase(name)
    kernels = Dict(
        "sequential" => kernel_2mm_seq!,
        "threads_static" => kernel_2mm_threads_static!,
        "threads_dynamic" => kernel_2mm_threads_dynamic!,
        "tiled" => kernel_2mm_tiled!,
        "blas" => kernel_2mm_blas!,
        "tasks" => kernel_2mm_tasks!,
    )
    return get(kernels, name_lower, nothing)
end

#=============================================================================
 Benchmark Runner
=============================================================================#
function run_benchmark(kernel!, alpha::Float64, beta::Float64,
                       A::Matrix{Float64}, B::Matrix{Float64},
                       tmp::Matrix{Float64}, C::Matrix{Float64},
                       D::Matrix{Float64}, D_orig::Matrix{Float64};
                       warmup::Int=5, iterations::Int=10)
    times = Float64[]
    
    # Warmup (JIT compilation)
    for _ in 1:warmup
        fill!(tmp, 0.0)
        copyto!(D, D_orig)
        kernel!(alpha, beta, A, B, tmp, C, D)
    end
    GC.gc()
    
    # Check allocations
    fill!(tmp, 0.0)
    copyto!(D, D_orig)
    allocs = @allocated kernel!(alpha, beta, A, B, tmp, C, D)
    
    # Timed runs
    for _ in 1:iterations
        fill!(tmp, 0.0)
        copyto!(D, D_orig)
        t = @elapsed kernel!(alpha, beta, A, B, tmp, C, D)
        push!(times, t * 1000)  # ms
    end
    
    return times, allocs
end

#=============================================================================
 Verification
=============================================================================#
function verify_result(D_ref::Matrix{Float64}, D_test::Matrix{Float64}, 
                       ni::Int, nj::Int, nk::Int, nl::Int)
    max_error = maximum(abs.(D_ref .- D_test))
    scale_factor = sqrt(Float64(ni) * Float64(nj) * Float64(nk) * Float64(nl))
    tolerance = max(1e-10, 1e-14 * scale_factor)
    return max_error, max_error < tolerance
end

#=============================================================================
 Main Function
=============================================================================#
function main()
    # Argument parsing
    dataset = "MEDIUM"
    strategies_arg = "all"
    iterations = 10
    warmup = 5
    do_verify = true
    output_csv = false
    do_profile = false
    
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--dataset" && i < length(ARGS)
            dataset = uppercase(ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--strategies" && i < length(ARGS)
            strategies_arg = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--iterations" && i < length(ARGS)
            iterations = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--warmup" && i < length(ARGS)
            warmup = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--no-verify"
            do_verify = false
            i += 1
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_csv = "csv" in lowercase(ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--profile"
            do_profile = true
            i += 1
        else
            i += 1
        end
    end
    
    # Get dataset parameters
    if !haskey(DATASET_SIZES, dataset)
        println("Unknown dataset: $dataset")
        println("Available: ", join(keys(DATASET_SIZES), ", "))
        return
    end
    
    params = DATASET_SIZES[dataset]
    ni, nj, nk, nl = params.ni, params.nj, params.nk, params.nl
    flops = flops_2mm(ni, nj, nk, nl)
    mem_bytes = memory_bytes_2mm(ni, nj, nk, nl)
    nthreads = Threads.nthreads()
    
    # Select strategies
    if strategies_arg == "all"
        strategies = STRATEGY_ORDER
    else
        strategies = [strip(s) for s in split(strategies_arg, ",")]
    end
    
    # Print header
    println("="^100)
    println("2MM BENCHMARK")
    println("="^100)
    println("Julia version: $(VERSION)")
    println("Threads: $nthreads")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("Dataset: $dataset (ni=$ni, nj=$nj, nk=$nk, nl=$nl)")
    println("Memory: $(round(mem_bytes / 1024^2, digits=2)) MB")
    println("FLOPs: $(round(flops / 1e9, digits=2)) GFLOP")
    println("Warmup: $warmup, Iterations: $iterations")
    println("="^100)
    
    # Allocate arrays
    A = zeros(Float64, ni, nk)
    B = zeros(Float64, nk, nj)
    tmp = zeros(Float64, ni, nj)
    C = zeros(Float64, nj, nl)
    D = zeros(Float64, ni, nl)
    D_orig = zeros(Float64, ni, nl)
    
    alpha = Ref(0.0)
    beta = Ref(0.0)
    
    # Initialize
    init_2mm!(alpha, beta, A, B, tmp, C, D)
    copyto!(D_orig, D)
    
    # Reference result (sequential)
    fill!(tmp, 0.0)
    copyto!(D, D_orig)
    kernel_2mm_seq!(alpha[], beta[], A, B, tmp, C, D)
    D_ref = copy(D)
    
    # Results storage
    results = BenchmarkResult[]
    
    # Run benchmarks
    for strategy in strategies
        kernel! = get_kernel(strategy)
        if kernel! === nothing
            println("Unknown strategy: $strategy")
            continue
        end
        
        # Special BLAS handling
        if strategy == "blas"
            configure_blas_threads(for_blas_benchmark=true)
        end
        
        times, allocs = run_benchmark(kernel!, alpha[], beta[], A, B, tmp, C, D, D_orig;
                                      warmup=warmup, iterations=iterations)
        
        # Verify
        verified = true
        max_error = 0.0
        if do_verify
            fill!(tmp, 0.0)
            copyto!(D, D_orig)
            kernel!(alpha[], beta[], A, B, tmp, C, D)
            max_error, verified = verify_result(D_ref, D, ni, nj, nk, nl)
        end
        
        push!(results, BenchmarkResult(strategy, times, verified, max_error, allocs))
        
        # Restore BLAS config
        if strategy == "blas"
            configure_blas_threads()
        end
    end
    
    # Find sequential baseline for speedup
    seq_result = findfirst(r -> r.strategy == "sequential", results)
    seq_min_time = seq_result !== nothing ? minimum(results[seq_result].times_ms) : nothing
    
    # Print results table
    println()
    println("="^100)
    @printf("%-18s | %10s | %10s | %10s | %10s | %8s | %8s | %8s\n",
            "Strategy", "Min(ms)", "Median(ms)", "Mean(ms)", "Std(ms)", "GFLOP/s", "Speedup", "Eff(%)")
    println("="^100)
    
    for r in results
        min_t = minimum(r.times_ms)
        med_t = Statistics.median(r.times_ms)
        mean_t = Statistics.mean(r.times_ms)
        std_t = length(r.times_ms) > 1 ? Statistics.std(r.times_ms) : 0.0
        gflops = flops / (min_t / 1000) / 1e9
        
        speedup = seq_min_time !== nothing ? seq_min_time / min_t : 1.0
        efficiency = compute_parallel_efficiency(r.strategy, speedup, nthreads)
        
        eff_str = efficiency === nothing ? "    N/A" : @sprintf("%7.1f", efficiency)
        status = r.verified ? "" : " [FAIL]"
        
        @printf("%-18s | %10.3f | %10.3f | %10.3f | %10.3f | %8.2f | %7.2fx | %s%s\n",
                r.strategy, min_t, med_t, mean_t, std_t, gflops, speedup, eff_str, status)
    end
    println("="^100)
    
    # CSV output
    if output_csv
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        csv_path = "results/2mm_$(dataset)_$(timestamp).csv"
        mkpath("results")
        
        open(csv_path, "w") do io
            println(io, "benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations")
            
            for r in results
                min_t = minimum(r.times_ms)
                med_t = Statistics.median(r.times_ms)
                mean_t = Statistics.mean(r.times_ms)
                std_t = length(r.times_ms) > 1 ? Statistics.std(r.times_ms) : 0.0
                gflops = flops / (min_t / 1000) / 1e9
                speedup = seq_min_time !== nothing ? seq_min_time / min_t : 1.0
                efficiency = compute_parallel_efficiency(r.strategy, speedup, nthreads)
                is_par = is_parallel_strategy(r.strategy)
                eff_str = efficiency === nothing ? "" : @sprintf("%.2f", efficiency)
                verified_str = r.verified ? "PASS" : "FAIL"
                
                @printf(io, "2mm,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%.2e,%d\n",
                        dataset, r.strategy, nthreads, is_par,
                        min_t, med_t, mean_t, std_t, gflops, speedup,
                        eff_str, verified_str, r.max_error, r.allocations)
            end
        end
        println("\nCSV exported: $csv_path")
    end
    
    # Profiling (if requested)
    if do_profile
        println("\nGenerating flame graph profile...")
        try
            using Profile
            using ProfileSVG
            
            # Profile threads_static (most common parallel strategy)
            kernel! = get_kernel("threads_static")
            
            # Warmup
            for _ in 1:3
                fill!(tmp, 0.0)
                copyto!(D, D_orig)
                kernel!(alpha[], beta[], A, B, tmp, C, D)
            end
            
            Profile.clear()
            @profile for _ in 1:50
                fill!(tmp, 0.0)
                copyto!(D, D_orig)
                kernel!(alpha[], beta[], A, B, tmp, C, D)
            end
            
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            svg_path = "results/flamegraph_2mm_$(dataset)_$(timestamp).svg"
            ProfileSVG.save(svg_path)
            println("Flame graph: $svg_path")
        catch e
            println("Profiling requires ProfileSVG.jl: Pkg.add(\"ProfileSVG\")")
            println("Error: $e")
        end
    end
    
    println("\nLegend:")
    println("  Speedup = T_sequential / T_strategy")
    println("  Eff(%) = (Speedup / Threads) * 100  [parallel strategies only]")
    println("  N/A = Efficiency not applicable (non-parallel strategy)")
end

main()
