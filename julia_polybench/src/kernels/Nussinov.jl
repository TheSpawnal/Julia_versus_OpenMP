module Nussinov

using Base.Threads
using Printf

export DATASET_SIZES, flops_nussinov
export init_sequence, init_table!
export kernel_nussinov_seq!, kernel_nussinov_simd!
export kernel_nussinov_wavefront!, kernel_nussinov_tiled_wavefront!
export get_kernel, verify_implementations

# Dataset sizes following PolyBench specification
const DATASET_SIZES = Dict(
    "MINI" => 60,
    "SMALL" => 180,
    "MEDIUM" => 500,
    "LARGE" => 2500,
    "EXTRALARGE" => 5500
)

# FLOPs estimate: O(n^3) for the DP table
function flops_nussinov(n::Int)
    return Float64(n)^3 / 6
end

# Base pair matching (Watson-Crick pairs)
@inline function match_pair(b1::Char, b2::Char)::Int
    # A-U and G-C pairs
    if (b1 == 'A' && b2 == 'U') || (b1 == 'U' && b2 == 'A')
        return 1
    elseif (b1 == 'G' && b2 == 'C') || (b1 == 'C' && b2 == 'G')
        return 1
    else
        return 0
    end
end

# Initialize random RNA sequence
function init_sequence(n::Int)
    bases = ['A', 'C', 'G', 'U']
    # Deterministic initialization for reproducibility
    seq = Vector{Char}(undef, n)
    for i in 1:n
        seq[i] = bases[((i-1) % 4) + 1]
    end
    return String(seq)
end

# Initialize DP table
function init_table!(S::Matrix{Int}, n::Int)
    fill!(S, 0)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Standard DP with i decreasing, j increasing
 - Respects all data dependencies
=============================================================================#
function kernel_nussinov_seq!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    # Process anti-diagonals (diagonal d = j - i)
    @inbounds for d in 1:(n-1)
        for i in 1:(n-d)
            j = i + d
            
            # Option 1: Don't pair position j
            S[i, j] = S[i, j-1]
            
            # Option 2: Don't pair position i
            S[i, j] = max(S[i, j], S[i+1, j])
            
            # Option 3: Pair positions i and j
            if match_pair(seq[i], seq[j]) == 1
                S[i, j] = max(S[i, j], S[i+1, j-1] + 1)
            end
            
            # Option 4: Split at some k
            for k in (i+1):(j-1)
                S[i, j] = max(S[i, j], S[i, k] + S[k+1, j])
            end
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Strategy 2: SIMD-optimized k-loop
 - Vectorizes the split point search
=============================================================================#
function kernel_nussinov_simd!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    @inbounds for d in 1:(n-1)
        for i in 1:(n-d)
            j = i + d
            
            # Options 1-3
            best = S[i, j-1]
            best = max(best, S[i+1, j])
            
            if match_pair(seq[i], seq[j]) == 1
                best = max(best, S[i+1, j-1] + 1)
            end
            
            # Option 4: Split search with SIMD hint
            max_split = 0
            @simd for k in (i+1):(j-1)
                split_val = S[i, k] + S[k+1, j]
                max_split = max(max_split, split_val)
            end
            best = max(best, max_split)
            
            S[i, j] = best
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Strategy 3: Wavefront parallel
 - Parallelizes along anti-diagonals
 - Each anti-diagonal has independent cells
=============================================================================#
function kernel_nussinov_wavefront!(S::Matrix{Int}, seq::String)
    n = length(seq)
    
    @inbounds for d in 1:(n-1)
        diag_len = n - d
        
        # Parallelize anti-diagonal
        Threads.@threads :static for idx in 1:diag_len
            i = idx
            j = i + d
            
            # Options 1-3
            best = S[i, j-1]
            best = max(best, S[i+1, j])
            
            if match_pair(seq[i], seq[j]) == 1
                best = max(best, S[i+1, j-1] + 1)
            end
            
            # Option 4: Split search
            for k in (i+1):(j-1)
                best = max(best, S[i, k] + S[k+1, j])
            end
            
            S[i, j] = best
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Strategy 4: Tiled wavefront
 - Combines tiling with wavefront parallelism
 - Better cache utilization
=============================================================================#
function kernel_nussinov_tiled_wavefront!(S::Matrix{Int}, seq::String; tile_size::Int=32)
    n = length(seq)
    ts = tile_size
    
    @inbounds for d in 1:(n-1)
        diag_len = n - d
        num_tiles = cld(diag_len, ts)
        
        # Process tiles along anti-diagonal
        Threads.@threads :static for tile in 1:num_tiles
            i_start = (tile - 1) * ts + 1
            i_end = min(tile * ts, diag_len)
            
            for i in i_start:i_end
                j = i + d
                
                # Options 1-3
                best = S[i, j-1]
                best = max(best, S[i+1, j])
                
                if match_pair(seq[i], seq[j]) == 1
                    best = max(best, S[i+1, j-1] + 1)
                end
                
                # Option 4: Split search
                for k in (i+1):(j-1)
                    best = max(best, S[i, k] + S[k+1, j])
                end
                
                S[i, j] = best
            end
        end
    end
    
    return S[1, n]
end

#=============================================================================
 Kernel selector
=============================================================================#
function get_kernel(name::AbstractString)
    name_lower = lowercase(String(name))
    
    kernels = Dict(
        "sequential" => kernel_nussinov_seq!,
        "seq" => kernel_nussinov_seq!,
        "simd" => kernel_nussinov_simd!,
        "wavefront" => kernel_nussinov_wavefront!,
        "threads" => kernel_nussinov_wavefront!,
        "tiled" => kernel_nussinov_tiled_wavefront!,
        "tiled_wavefront" => kernel_nussinov_tiled_wavefront!,
    )
    
    if haskey(kernels, name_lower)
        return kernels[name_lower]
    else
        available = join(keys(kernels), ", ")
        error("Unknown kernel: $name. Available: $available")
    end
end

#=============================================================================
 Verification
=============================================================================#
function verify_implementations(dataset::String="SMALL")
    n = DATASET_SIZES[dataset]
    seq = init_sequence(n)
    
    # Reference (sequential)
    S_ref = zeros(Int, n, n)
    init_table!(S_ref, n)
    ref_score = kernel_nussinov_seq!(S_ref, seq)
    
    implementations = [
        ("Sequential", kernel_nussinov_seq!),
        ("SIMD", kernel_nussinov_simd!),
        ("Wavefront", kernel_nussinov_wavefront!),
        ("Tiled", kernel_nussinov_tiled_wavefront!),
    ]
    
    println("Verifying Nussinov on $dataset (n=$n)")
    println("Reference score: $ref_score")
    println("-"^50)
    @printf("%-20s | %10s | %10s\n", "Implementation", "Score", "Status")
    println("-"^50)
    
    all_pass = true
    for (name, func) in implementations
        S_test = zeros(Int, n, n)
        init_table!(S_test, n)
        score = func(S_test, seq)
        
        status = score == ref_score ? "PASS" : "FAIL"
        all_pass &= (score == ref_score)
        
        @printf("%-20s | %10d | %10s\n", name, score, status)
    end
    println("-"^50)
    
    return all_pass
end

end # module