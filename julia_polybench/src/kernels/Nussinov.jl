# Nussinov RNA Folding Benchmark - Corrected Implementation
#
# The original had two bugs:
# 1. get_kernel(::SubString{String}) method not found
# 2. Naive parallelization breaking wavefront dependencies
#
# This implementation uses anti-diagonal (wavefront) parallelism which is
# the ONLY correct way to parallelize Nussinov.

module PolyBenchNussinov_Corrected

using Base.Threads
using BenchmarkTools
using Printf
using Statistics

export kernel_nussinov_seq!, kernel_nussinov_simd!, kernel_nussinov_wavefront!
export kernel_nussinov_tiled_wavefront!, kernel_nussinov_tasks!
export main, verify_implementations, run_benchmarks, get_kernel

# Dataset sizes (sequence lengths)
const DATASET_SIZES = Dict(
    "MINI" => 60,
    "SMALL" => 180,
    "MEDIUM" => 500,
    "LARGE" => 2500,
    "EXTRALARGE" => 5500
)

# Base pairing function (simplified Watson-Crick)
@inline function can_pair(b1::Char, b2::Char)
    return (b1 == 'A' && b2 == 'U') || (b1 == 'U' && b2 == 'A') ||
           (b1 == 'G' && b2 == 'C') || (b1 == 'C' && b2 == 'G') ||
           (b1 == 'G' && b2 == 'U') || (b1 == 'U' && b2 == 'G')  # Wobble pair
end

# Initialize sequence with random bases
function init_sequence(n::Int)
    bases = ['A', 'C', 'G', 'U']
    return String([bases[rand(1:4)] for _ in 1:n])
end

# Initialize score matrix
function init_table!(S::Matrix{Int}, n::Int)
    fill!(S, 0)
end

#=============================================================================
 Strategy 1: Sequential baseline
 
 Nussinov recurrence (i < j):
 S[i,j] = max(
     S[i+1,j],           # i unpaired
     S[i,j-1],           # j unpaired  
     S[i+1,j-1] + match, # i,j paired
     max(S[i,k] + S[k+1,j] for k in i:j-1)  # bifurcation
 )
 
 Dependencies: S[i,j] depends on S[i+1,*], S[i,j-1], S[i+1,j-1]
 Must compute in order of increasing diagonal offset (j - i)
=============================================================================#
function kernel_nussinov_seq!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    # Process diagonals (anti-diagonals in standard matrix view)
    # d = j - i (diagonal offset)
    @inbounds for d in 2:n-1  # Start from d=2 (d=0,1 are base cases)
        for i in 1:n-d
            j = i + d
            
            # Option 1: i unpaired
            best = S[i+1, j]
            
            # Option 2: j unpaired
            best = max(best, S[i, j-1])
            
            # Option 3: i,j paired (if compatible bases)
            if can_pair(seq[i], seq[j])
                best = max(best, S[i+1, j-1] + 1)
            end
            
            # Option 4: Bifurcation (try all split points)
            for k in i+1:j-1
                best = max(best, S[i, k] + S[k+1, j])
            end
            
            S[i, j] = best
        end
    end
    
    return S[1, n]  # Optimal score
end

#=============================================================================
 Strategy 2: SIMD-optimized (vectorize bifurcation loop)
=============================================================================#
function kernel_nussinov_simd!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    @inbounds for d in 2:n-1
        for i in 1:n-d
            j = i + d
            
            best = max(S[i+1, j], S[i, j-1])
            
            if can_pair(seq[i], seq[j])
                best = max(best, S[i+1, j-1] + 1)
            end
            
            # SIMD for bifurcation (limited benefit due to data dependencies)
            bifurc_max = 0
            @simd for k in i+1:j-1
                bifurc_max = max(bifurc_max, S[i, k] + S[k+1, j])
            end
            best = max(best, bifurc_max)
            
            S[i, j] = best
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Strategy 3: Wavefront parallelism (CORRECT parallelization)
 
 Key insight: All cells on the same anti-diagonal (same d = j - i)
 can be computed in parallel because they only depend on cells
 from previous diagonals.
=============================================================================#
function kernel_nussinov_wavefront!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    @inbounds for d in 2:n-1
        # Parallelize over cells in this diagonal
        @threads :static for i in 1:n-d
            j = i + d
            
            best = max(S[i+1, j], S[i, j-1])
            
            if can_pair(seq[i], seq[j])
                best = max(best, S[i+1, j-1] + 1)
            end
            
            # Bifurcation
            for k in i+1:j-1
                best = max(best, S[i, k] + S[k+1, j])
            end
            
            S[i, j] = best
        end
        # Implicit synchronization at end of @threads block
    end
    
    return S[1, n]
end

#=============================================================================
 Strategy 4: Tiled wavefront (better cache utilization)
 
 Process tiles along the anti-diagonal, with explicit synchronization
 between tile rows.
=============================================================================#
function kernel_nussinov_tiled_wavefront!(S::Matrix{Int}, seq::String; tile_size::Int=64)
    n = length(seq)
    
    @inbounds for d in 2:n-1
        # For small diagonals, just use sequential
        if n - d < tile_size * 2
            for i in 1:n-d
                j = i + d
                compute_cell!(S, seq, i, j)
            end
        else
            # Parallelize in tiles
            num_tiles = cld(n - d, tile_size)
            @threads :static for t in 1:num_tiles
                i_start = (t - 1) * tile_size + 1
                i_end = min(t * tile_size, n - d)
                
                for i in i_start:i_end
                    j = i + d
                    compute_cell!(S, seq, i, j)
                end
            end
        end
    end
    
    return S[1, n]
end

@inline function compute_cell!(S::Matrix{Int}, seq::String, i::Int, j::Int)
    @inbounds begin
        best = max(S[i+1, j], S[i, j-1])
        
        if can_pair(seq[i], seq[j])
            best = max(best, S[i+1, j-1] + 1)
        end
        
        for k in i+1:j-1
            best = max(best, S[i, k] + S[k+1, j])
        end
        
        S[i, j] = best
    end
end

#=============================================================================
 Strategy 5: Task-based with coarse granularity
=============================================================================#
function kernel_nussinov_tasks!(S::Matrix{Int}, seq::String; chunk_size::Int=32)
    n = length(seq)
    
    @inbounds for d in 2:n-1
        diag_length = n - d
        
        if diag_length < chunk_size * 2
            # Small diagonal: sequential
            for i in 1:diag_length
                compute_cell!(S, seq, i, i + d)
            end
        else
            # Large diagonal: spawn tasks
            num_chunks = cld(diag_length, chunk_size)
            @sync for chunk in 1:num_chunks
                Threads.@spawn begin
                    i_start = (chunk - 1) * chunk_size + 1
                    i_end = min(chunk * chunk_size, diag_length)
                    @inbounds for i in i_start:i_end
                        compute_cell!(S, seq, i, i + d)
                    end
                end
            end
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Kernel selector - FIXED to accept AbstractString
=============================================================================#
function get_kernel(name::AbstractString)  # Fixed: accepts both String and SubString
    name_lower = lowercase(String(name))  # Convert to String for safety
    
    kernels = Dict(
        "sequential" => kernel_nussinov_seq!,
        "seq" => kernel_nussinov_seq!,
        "simd" => kernel_nussinov_simd!,
        "wavefront" => kernel_nussinov_wavefront!,
        "threads" => kernel_nussinov_wavefront!,  # Alias
        "tiled" => kernel_nussinov_tiled_wavefront!,
        "tiled_wavefront" => kernel_nussinov_tiled_wavefront!,
        "tasks" => kernel_nussinov_tasks!,
    )
    
    if haskey(kernels, name_lower)
        return kernels[name_lower]
    else
        available = join(keys(kernels), ", ")
        error("Unknown kernel: $name. Available: $available")
    end
end

#=============================================================================
 Verification and Benchmarking
=============================================================================#
function verify_implementations(dataset="SMALL"; tolerance=0)
    n = DATASET_SIZES[dataset]
    seq = init_sequence(n)
    
    # Reference result (sequential)
    S_ref = zeros(Int, n, n)
    init_table!(S_ref, n)
    ref_score = kernel_nussinov_seq!(S_ref, seq)
    
    implementations = [
        ("Sequential", kernel_nussinov_seq!),
        ("SIMD", kernel_nussinov_simd!),
        ("Wavefront", kernel_nussinov_wavefront!),
        ("Tiled Wavefront", kernel_nussinov_tiled_wavefront!),
        ("Tasks", kernel_nussinov_tasks!),
    ]
    
    println("Verifying Nussinov implementations on $dataset (n=$n)...")
    println("Reference optimal score: $ref_score")
    println("-"^60)
    @printf("%-20s | %10s | %10s\n", "Implementation", "Score", "Status")
    println("-"^60)
    
    all_pass = true
    for (name, func) in implementations
        S_test = zeros(Int, n, n)
        init_table!(S_test, n)
        test_score = func(S_test, seq)
        
        # Compare scores (should be identical)
        status = test_score == ref_score ? "PASS" : "FAIL"
        if test_score != ref_score
            all_pass = false
        end
        
        @printf("%-20s | %10d | %10s\n", name, test_score, status)
    end
    
    return all_pass
end

function run_benchmarks(dataset_name="SMALL"; samples=10, seconds=5)
    n = DATASET_SIZES[dataset_name]
    
    # FLOPs approximation: O(n^3) for the bifurcation loops
    flops = Float64(n)^3
    
    println("\n", "="^70)
    println("NUSSINOV RNA FOLDING BENCHMARK")
    println("="^70)
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("Dataset: $dataset_name (n=$n)")
    println("Memory: $(round(n * n * 4 / 1024^2, digits=2)) MB (Int matrix)")
    println()
    println("NOTE: Wavefront dependencies limit parallelism to O(n) per diagonal.")
    println()
    
    # Generate test sequence
    seq = init_sequence(n)
    
    implementations = [
        ("sequential", kernel_nussinov_seq!),
        ("simd", kernel_nussinov_simd!),
        ("wavefront", kernel_nussinov_wavefront!),
        ("tiled", kernel_nussinov_tiled_wavefront!),
        ("tasks", kernel_nussinov_tasks!),
    ]
    
    # Baseline
    S = zeros(Int, n, n)
    baseline_trial = @benchmark kernel_nussinov_seq!($S, $seq) setup=(fill!($S, 0)) samples=samples seconds=seconds
    baseline_ns = minimum(baseline_trial).time
    
    println("-"^70)
    @printf("%-18s | %10s | %10s | %10s | %8s\n",
            "Strategy", "Min(ms)", "Med(ms)", "Mean(ms)", "Speedup")
    println("-"^70)
    
    for (name, func) in implementations
        S = zeros(Int, n, n)
        trial = @benchmark $func($S, $seq) setup=(fill!($S, 0)) samples=samples seconds=seconds
        
        min_time = minimum(trial).time / 1e6
        med_time = median(trial).time / 1e6
        mean_time = mean(trial).time / 1e6
        speedup = baseline_ns / minimum(trial).time
        
        @printf("%-18s | %10.3f | %10.3f | %10.3f | %7.2fx\n",
                name, min_time, med_time, mean_time, speedup)
    end
    
    println("-"^70)
end

function main(; datasets=["MINI", "SMALL"])
    # Configure threading
    if Threads.nthreads() > 1
        println("Running with $(Threads.nthreads()) threads")
    else
        println("Warning: Single-threaded mode. Use julia -t N for parallelism.")
    end
    
    # Verify first
    verify_implementations("MINI")
    
    # Run benchmarks
    for dataset in datasets
        if haskey(DATASET_SIZES, dataset)
            run_benchmarks(dataset)
        else
            println("Unknown dataset: $dataset")
        end
    end
end

end # module
