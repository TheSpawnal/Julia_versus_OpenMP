module Correlation

using LinearAlgebra
using Base.Threads

export init_correlation!, kernel_correlation_seq!, kernel_correlation_threads!
export kernel_correlation_colmajor!, kernel_correlation_tiled!, kernel_correlation_reduction!
export STRATEGIES_CORRELATION

const STRATEGIES_CORRELATION = [
    "sequential",
    "threads",
    "colmajor",
    "tiled",
    "reduction"
]

const EPS = 0.1

# Initialize data matrix following PolyBench pattern
# data: M data points x N variables (column-major: each column is a variable)
function init_correlation!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    
    # Initialize data matrix (column-major order)
    @inbounds for j in 1:n, i in 1:m
        data[i, j] = ((i - 1) * j) / m + i
    end
    
    fill!(mean, 0.0)
    fill!(stddev, 0.0)
    fill!(corr, 0.0)
    
    # Set diagonal of correlation matrix to 1
    @inbounds for i in 1:n
        corr[i, i] = 1.0
    end
    
    return nothing
end

# Reset arrays for next iteration
function reset_correlation!(
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    fill!(mean, 0.0)
    fill!(stddev, 0.0)
    fill!(corr, 0.0)
    n = size(corr, 1)
    @inbounds for i in 1:n
        corr[i, i] = 1.0
    end
    return nothing
end

#=============================================================================
 Strategy 1: Sequential baseline
 - Direct implementation following PolyBench/C
=============================================================================#
function kernel_correlation_seq!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    float_m = Float64(m)
    
    # Step 1: Compute mean for each variable (column)
    @inbounds for j in 1:n
        sum_val = 0.0
        for i in 1:m
            sum_val += data[i, j]
        end
        mean[j] = sum_val / float_m
    end
    
    # Step 2: Compute standard deviation for each variable
    @inbounds for j in 1:n
        sum_sq = 0.0
        for i in 1:m
            diff = data[i, j] - mean[j]
            sum_sq += diff * diff
        end
        stddev[j] = sqrt(sum_sq / float_m)
        if stddev[j] <= EPS
            stddev[j] = 1.0
        end
    end
    
    # Step 3: Center and scale the data (normalize)
    @inbounds for j in 1:n
        sqrt_m = sqrt(float_m)
        inv_std = 1.0 / (sqrt_m * stddev[j])
        for i in 1:m
            data[i, j] = (data[i, j] - mean[j]) * inv_std
        end
    end
    
    # Step 4: Compute correlation matrix (upper triangle)
    @inbounds for i in 1:(n-1)
        for j in (i+1):n
            sum_val = 0.0
            for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val  # Symmetric
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 2: Threaded parallel
 - Parallelizes over columns (variables)
=============================================================================#
function kernel_correlation_threads!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    float_m = Float64(m)
    
    # Step 1: Mean (parallel over columns)
    @threads :static for j in 1:n
        @inbounds begin
            sum_val = 0.0
            for i in 1:m
                sum_val += data[i, j]
            end
            mean[j] = sum_val / float_m
        end
    end
    
    # Step 2: Standard deviation (parallel over columns)
    @threads :static for j in 1:n
        @inbounds begin
            sum_sq = 0.0
            m_j = mean[j]
            for i in 1:m
                diff = data[i, j] - m_j
                sum_sq += diff * diff
            end
            stddev[j] = sqrt(sum_sq / float_m)
            if stddev[j] <= EPS
                stddev[j] = 1.0
            end
        end
    end
    
    # Step 3: Normalize (parallel over columns)
    sqrt_m = sqrt(float_m)
    @threads :static for j in 1:n
        @inbounds begin
            inv_std = 1.0 / (sqrt_m * stddev[j])
            m_j = mean[j]
            for i in 1:m
                data[i, j] = (data[i, j] - m_j) * inv_std
            end
        end
    end
    
    # Step 4: Correlation (parallel over rows of upper triangle)
    @threads :static for i in 1:(n-1)
        @inbounds for j in (i+1):n
            sum_val = 0.0
            for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 3: Column-major optimized
 - Reorders operations for better cache utilization
=============================================================================#
function kernel_correlation_colmajor!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    float_m = Float64(m)
    sqrt_m = sqrt(float_m)
    
    # Combined mean and stddev computation (better cache reuse)
    @threads :static for j in 1:n
        @inbounds begin
            sum_val = 0.0
            sum_sq = 0.0
            for i in 1:m
                val = data[i, j]
                sum_val += val
                sum_sq += val * val
            end
            m_j = sum_val / float_m
            mean[j] = m_j
            # variance = E[X^2] - E[X]^2
            variance = (sum_sq / float_m) - m_j * m_j
            stddev[j] = sqrt(max(0.0, variance))
            if stddev[j] <= EPS
                stddev[j] = 1.0
            end
        end
    end
    
    # Normalize in place
    @threads :static for j in 1:n
        @inbounds begin
            inv_std = 1.0 / (sqrt_m * stddev[j])
            m_j = mean[j]
            for i in 1:m
                data[i, j] = (data[i, j] - m_j) * inv_std
            end
        end
    end
    
    # Correlation: column-major order for j (outer)
    @threads :static for j in 2:n
        @inbounds for i in 1:(j-1)
            sum_val = 0.0
            for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 4: Tiled for cache optimization
=============================================================================#
function kernel_correlation_tiled!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64};
    tile_size::Int=64
)
    m, n = size(data)
    float_m = Float64(m)
    sqrt_m = sqrt(float_m)
    ts = tile_size
    
    # Mean and stddev (parallel over column tiles)
    @threads :static for jj in 1:ts:n
        j_end = min(jj + ts - 1, n)
        @inbounds for j in jj:j_end
            sum_val = 0.0
            sum_sq = 0.0
            for i in 1:m
                val = data[i, j]
                sum_val += val
                sum_sq += val * val
            end
            m_j = sum_val / float_m
            mean[j] = m_j
            variance = (sum_sq / float_m) - m_j * m_j
            stddev[j] = sqrt(max(0.0, variance))
            if stddev[j] <= EPS
                stddev[j] = 1.0
            end
        end
    end
    
    # Normalize (tiled)
    @threads :static for jj in 1:ts:n
        j_end = min(jj + ts - 1, n)
        @inbounds for j in jj:j_end
            inv_std = 1.0 / (sqrt_m * stddev[j])
            m_j = mean[j]
            for i in 1:m
                data[i, j] = (data[i, j] - m_j) * inv_std
            end
        end
    end
    
    # Correlation (tiled over output matrix)
    @threads :static for jj in 1:ts:n
        j_end = min(jj + ts - 1, n)
        for ii in 1:ts:n
            i_end = min(ii + ts - 1, n)
            
            @inbounds for j in max(jj, 2):j_end
                for i in ii:min(i_end, j-1)
                    sum_val = 0.0
                    for k in 1:m
                        sum_val += data[k, i] * data[k, j]
                    end
                    corr[i, j] = sum_val
                    corr[j, i] = sum_val
                end
            end
        end
    end
    
    return nothing
end

#=============================================================================
 Strategy 5: Reduction-based parallel
 - Uses thread-local accumulators to avoid false sharing
=============================================================================#
function kernel_correlation_reduction!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    float_m = Float64(m)
    sqrt_m = sqrt(float_m)
    nt = nthreads()
    
    # Thread-local accumulators for mean
    local_sums = zeros(Float64, n, nt)
    
    # Parallel sum with thread-local storage
    @threads :static for tid in 1:nt
        chunk = cld(m, nt)
        i_start = (tid - 1) * chunk + 1
        i_end = min(tid * chunk, m)
        
        @inbounds for j in 1:n
            local_sum = 0.0
            for i in i_start:i_end
                local_sum += data[i, j]
            end
            local_sums[j, tid] = local_sum
        end
    end
    
    # Reduce and compute mean/stddev
    @threads :static for j in 1:n
        @inbounds begin
            total = 0.0
            for t in 1:nt
                total += local_sums[j, t]
            end
            mean[j] = total / float_m
        end
    end
    
    # Stddev with reduction
    local_sq = zeros(Float64, n, nt)
    
    @threads :static for tid in 1:nt
        chunk = cld(m, nt)
        i_start = (tid - 1) * chunk + 1
        i_end = min(tid * chunk, m)
        
        @inbounds for j in 1:n
            local_sum = 0.0
            m_j = mean[j]
            for i in i_start:i_end
                diff = data[i, j] - m_j
                local_sum += diff * diff
            end
            local_sq[j, tid] = local_sum
        end
    end
    
    @threads :static for j in 1:n
        @inbounds begin
            total = 0.0
            for t in 1:nt
                total += local_sq[j, t]
            end
            stddev[j] = sqrt(total / float_m)
            if stddev[j] <= EPS
                stddev[j] = 1.0
            end
        end
    end
    
    # Normalize
    @threads :static for j in 1:n
        @inbounds begin
            inv_std = 1.0 / (sqrt_m * stddev[j])
            m_j = mean[j]
            for i in 1:m
                data[i, j] = (data[i, j] - m_j) * inv_std
            end
        end
    end
    
    # Correlation with thread-local accumulation
    @threads :static for i in 1:(n-1)
        @inbounds for j in (i+1):n
            sum_val = 0.0
            for k in 1:m
                sum_val += data[k, i] * data[k, j]
            end
            corr[i, j] = sum_val
            corr[j, i] = sum_val
        end
    end
    
    return nothing
end

# Get kernel by strategy name
function get_kernel(strategy::String)
    kernels = Dict(
        "sequential" => kernel_correlation_seq!,
        "threads" => kernel_correlation_threads!,
        "colmajor" => kernel_correlation_colmajor!,
        "tiled" => kernel_correlation_tiled!,
        "reduction" => kernel_correlation_reduction!
    )
    return get(kernels, strategy, kernel_correlation_seq!)
end

end # module
