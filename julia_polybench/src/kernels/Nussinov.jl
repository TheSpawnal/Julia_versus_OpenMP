
module Nussinov

using Base.Threads

export init_nussinov!, kernel_nussinov_seq!, kernel_nussinov_wavefront!
export kernel_nussinov_tiled!, kernel_nussinov_tasks!, kernel_nussinov_pipeline!
export kernel_nussinov_hybrid!
export STRATEGIES_NUSSINOV

const STRATEGIES_NUSSINOV = [
    "sequential",
    "wavefront",
    "tiled",
    "tasks",
    "pipeline",
    "hybrid"
]

# Base types for RNA sequence (A=0, C=1, G=2, U=3)
@enum Base::Int8 A=0 C=1 G=2 U=3

# Watson-Crick base pairing check
@inline function can_pair(b1::Int8, b2::Int8)::Int
    # A-U (0+3=3) and C-G (1+2=3) pairs
    return (b1 + b2 == 3) ? 1 : 0
end

# Initialize RNA sequence and DP table
function init_nussinov!(seq::Vector{Int8}, table::Matrix{Int})
    n = length(seq)
    
    # Initialize sequence (cyclic pattern)
    @inbounds for i in 1:n
        seq[i] = Int8((i - 1) % 4)
    end
    
    # Initialize DP table to zeros
    fill!(table, 0)
    
    return nothing
end

# Reset table for next iteration
function reset_nussinov!(table::Matrix{Int})
    fill!(table, 0)
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Standard DP with correct dependency order
 - Iteration order: i from n-1 down to 1, j from i+1 to n
=============================================================================#
function kernel_nussinov_seq!(seq::Vector{Int8}, table::Matrix{Int})
    n = length(seq)
    
    @inbounds for i in (n-1):-1:1
        for j in (i+1):n
            # Case 1: i and j pair (if span > 1)
            if j - i > 1
                table[i, j] = max(table[i, j], 
                                  table[i+1, j-1] + can_pair(seq[i], seq[j]))
            elseif j - i == 1
                table[i, j] = max(table[i, j], can_pair(seq[i], seq[j]))
            end
            
            # Case 2: i unpaired
            if i + 1 <= n
                table[i, j] = max(table[i, j], table[i+1, j])
            end
            
            # Case 3: j unpaired  
            if j - 1 >= 1
                table[i, j] = max(table[i, j], table[i, j-1])
            end
            
            # Case 4: bifurcation at k
            for k in (i+1):(j-1)
                table[i, j] = max(table[i, j], table[i, k] + table[k+1, j])
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Wavefront/Anti-diagonal parallel
 - Elements on same anti-diagonal are independent
 - Parallelize over anti-diagonals
=============================================================================#
function kernel_nussinov_wavefront!(seq::Vector{Int8}, table::Matrix{Int})
    n = length(seq)
    
    # Process anti-diagonals (diag = j - i, from 1 to n-1)
    for diag in 1:(n-1)
        num_elements = n - diag
        
        # Parallelize elements on this anti-diagonal
        @threads :dynamic for idx in 1:num_elements
            i = idx
            j = idx + diag
            
            @inbounds begin
                # Case 1: i and j pair
                if diag > 1
                    table[i, j] = max(table[i, j],
                                      table[i+1, j-1] + can_pair(seq[i], seq[j]))
                elseif diag == 1
                    table[i, j] = can_pair(seq[i], seq[j])
                end
                
                # Case 2: i unpaired
                table[i, j] = max(table[i, j], table[i+1, j])
                
                # Case 3: j unpaired
                table[i, j] = max(table[i, j], table[i, j-1])
                
                # Case 4: bifurcation
                local_max = table[i, j]
                for k in (i+1):(j-1)
                    local_max = max(local_max, table[i, k] + table[k+1, j])
                end
                table[i, j] = local_max
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Tiled wavefront
 - Process in tile diagonals for better cache utilization
=============================================================================#
function kernel_nussinov_tiled!(seq::Vector{Int8}, table::Matrix{Int}; tile_size::Int=32)
    n = length(seq)
    ts = tile_size
    
    # Number of tiles along each dimension
    num_tiles = cld(n, ts)
    
    # Process tile diagonals
    for tile_diag in 0:(2 * num_tiles - 1)
        # Tiles on this diagonal can be processed in parallel
        @threads :static for tile_i in 0:min(tile_diag, num_tiles-1)
            tile_j = tile_diag - tile_i
            
            tile_j >= num_tiles && continue
            tile_i > tile_j && continue  # Upper triangle only
            
            i_start = tile_i * ts + 1
            j_start = tile_j * ts + 1
            i_end = min(i_start + ts - 1, n)
            j_end = min(j_start + ts - 1, n)
            
            # Process elements within tile in correct order
            @inbounds for diag in 1:(ts * 2)
                for local_i in max(1, diag - ts + 1):min(diag, ts)
                    local_j = diag - local_i + 1
                    
                    i = i_start + local_i - 1
                    j = j_start + local_j - 1
                    
                    i > i_end && continue
                    j > j_end && continue
                    j <= i && continue
                    j > n && continue
                    
                    span = j - i
                    
                    # Case 1: i and j pair
                    if span > 1
                        table[i, j] = max(table[i, j],
                                          table[i+1, j-1] + can_pair(seq[i], seq[j]))
                    elseif span == 1
                        table[i, j] = can_pair(seq[i], seq[j])
                    end
                    
                    # Case 2: i unpaired
                    if i + 1 <= n
                        table[i, j] = max(table[i, j], table[i+1, j])
                    end
                    
                    # Case 3: j unpaired
                    if j - 1 >= i
                        table[i, j] = max(table[i, j], table[i, j-1])
                    end
                    
                    # Case 4: bifurcation
                    local_max = table[i, j]
                    for k in (i+1):(j-1)
                        local_max = max(local_max, table[i, k] + table[k+1, j])
                    end
                    table[i, j] = local_max
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Task-based with dependencies
 - Create tasks for chunks of anti-diagonals
=============================================================================#
function kernel_nussinov_tasks!(seq::Vector{Int8}, table::Matrix{Int}; min_chunk::Int=50)
    n = length(seq)
    
    for diag in 1:(n-1)
        num_elements = n - diag
        
        if num_elements <= min_chunk
            # Sequential for small diagonals
            @inbounds for idx in 1:num_elements
                compute_cell!(seq, table, idx, idx + diag)
            end
        else
            # Task-based for larger diagonals
            num_chunks = cld(num_elements, min_chunk)
            
            @sync begin
                for chunk_id in 1:num_chunks
                    idx_start = (chunk_id - 1) * min_chunk + 1
                    idx_end = min(chunk_id * min_chunk, num_elements)
                    
                    Threads.@spawn begin
                        @inbounds for idx in idx_start:idx_end
                            compute_cell!(seq, table, idx, idx + diag)
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

# Helper function for computing a single cell
@inline function compute_cell!(seq::Vector{Int8}, table::Matrix{Int}, i::Int, j::Int)
    span = j - i
    
    @inbounds begin
        # Case 1
        if span > 1
            table[i, j] = max(table[i, j], table[i+1, j-1] + can_pair(seq[i], seq[j]))
        elseif span == 1
            table[i, j] = can_pair(seq[i], seq[j])
        end
        
        # Case 2
        n = length(seq)
        if i + 1 <= n
            table[i, j] = max(table[i, j], table[i+1, j])
        end
        
        # Case 3
        if j - 1 >= 1
            table[i, j] = max(table[i, j], table[i, j-1])
        end
        
        # Case 4
        local_max = table[i, j]
        for k in (i+1):(j-1)
            local_max = max(local_max, table[i, k] + table[k+1, j])
        end
        table[i, j] = local_max
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: Pipeline parallel
 - Overlapping computation of adjacent anti-diagonals
=============================================================================#
function kernel_nussinov_pipeline!(seq::Vector{Int8}, table::Matrix{Int}; stripe_width::Int=16)
    n = length(seq)
    num_stripes = cld(n, stripe_width)
    
    # Process in stages, with pipeline fill/drain
    for stage in 1:(n - 1 + num_stripes - 1)
        @threads :static for stripe in 0:(num_stripes-1)
            diag = stage - stripe
            
            # Skip if diagonal not yet ready or already done
            (diag < 1 || diag >= n) && continue
            
            i_start = stripe * stripe_width + 1
            i_end = min(i_start + stripe_width - 1, n - diag)
            
            i_start > n - diag && continue
            
            @inbounds for i in i_start:i_end
                j = i + diag
                j > n && continue
                
                span = j - i
                
                # Case 1
                if span > 1
                    table[i, j] = max(table[i, j],
                                      table[i+1, j-1] + can_pair(seq[i], seq[j]))
                elseif span == 1
                    table[i, j] = can_pair(seq[i], seq[j])
                end
                
                # Case 2
                if i + 1 <= n
                    table[i, j] = max(table[i, j], table[i+1, j])
                end
                
                # Case 3
                if j - 1 >= i
                    table[i, j] = max(table[i, j], table[i, j-1])
                end
                
                # Case 4
                local_max = table[i, j]
                for k in (i+1):(j-1)
                    local_max = max(local_max, table[i, k] + table[k+1, j])
                end
                table[i, j] = local_max
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 6: Hybrid coarse+fine grained
 - Combines tiling with thread-level parallelism within tiles
=============================================================================#
function kernel_nussinov_hybrid!(seq::Vector{Int8}, table::Matrix{Int}; 
                                  coarse_tile::Int=128, fine_tile::Int=32)
    n = length(seq)
    ct = coarse_tile
    ft = fine_tile
    
    num_coarse_tiles = cld(n, ct)
    
    # Coarse tile diagonals
    for ct_diag in 0:(2 * num_coarse_tiles - 1)
        
        # Coarse tiles on this diagonal (parallel)
        @threads :static for ct_i in 0:min(ct_diag, num_coarse_tiles-1)
            ct_j = ct_diag - ct_i
            
            ct_j >= num_coarse_tiles && continue
            ct_i > ct_j && continue  # Upper triangle only
            
            ci_start = ct_i * ct + 1
            cj_start = ct_j * ct + 1
            ci_end = min(ci_start + ct - 1, n)
            cj_end = min(cj_start + ct - 1, n)
            
            # Within coarse tile: process with fine tiles
            num_fine_tiles = cld(ct, ft)
            
            for ft_diag in 0:(2 * num_fine_tiles - 1)
                for ft_i in 0:min(ft_diag, num_fine_tiles-1)
                    ft_j = ft_diag - ft_i
                    
                    ft_j >= num_fine_tiles && continue
                    
                    fi_start = ci_start + ft_i * ft
                    fj_start = cj_start + ft_j * ft
                    fi_end = min(fi_start + ft - 1, ci_end)
                    fj_end = min(fj_start + ft - 1, cj_end)
                    
                    # Sequential within fine tile (good cache locality)
                    @inbounds for i in (fi_end):-1:fi_start
                        for j in fj_start:fj_end
                            j <= i && continue
                            j > n && continue
                            
                            span = j - i
                            
                            # Case 1
                            if span > 1 && i + 1 <= n && j - 1 >= 1
                                table[i, j] = max(table[i, j],
                                                  table[i+1, j-1] + can_pair(seq[i], seq[j]))
                            elseif span == 1
                                table[i, j] = max(table[i, j], can_pair(seq[i], seq[j]))
                            end
                            
                            # Case 2
                            if i + 1 <= n
                                table[i, j] = max(table[i, j], table[i+1, j])
                            end
                            
                            # Case 3
                            if j - 1 >= i
                                table[i, j] = max(table[i, j], table[i, j-1])
                            end
                            
                            # Case 4
                            local_max = table[i, j]
                            for k in (i+1):(j-1)
                                local_max = max(local_max, table[i, k] + table[k+1, j])
                            end
                            table[i, j] = local_max
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

# Get kernel by strategy name
function get_kernel(strategy::String)
    kernels = Dict(
        "sequential" => kernel_nussinov_seq!,
        "wavefront" => kernel_nussinov_wavefront!,
        "tiled" => kernel_nussinov_tiled!,
        "tasks" => kernel_nussinov_tasks!,
        "pipeline" => kernel_nussinov_pipeline!,
        "hybrid" => kernel_nussinov_hybrid!
    )
    return get(kernels, strategy, kernel_nussinov_seq!)
end

end # module
