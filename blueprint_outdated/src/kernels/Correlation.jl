module Correlation

using LinearAlgebra
using Base.Threads
using Printf

export STRATEGIES_CORRELATION, DATASETS_CORRELATION
export init_correlation!, get_kernel
export kernel_correlation_seq!, kernel_correlation_threads!
export kernel_correlation_colmajor!, kernel_correlation_tiled!

const STRATEGIES_CORRELATION = [
    "sequential",
    "threads",
    "colmajor",
    "tiled"
]

const DATASETS_CORRELATION = Dict(
    "MINI" => (m=28, n=32),
    "SMALL" => (m=80, n=100),
    "MEDIUM" => (m=240, n=260),
    "LARGE" => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

const EPS = 0.1

# Initialize data matrix following PolyBench pattern
function init_correlation!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    
    # Initialize data (column-major)
    @inbounds for j in 1:n, i in 1:m
        data[i, j] = Float64((i - 1) * j) / m + i
    end
    
    fill!(mean, 0.0)
    fill!(stddev, 0.0)
    fill!(corr, 0.0)
    
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
    
    # Step 1: Mean for each column
    @inbounds for j in 1:n
        sum_val = 0.0
        for i in 1:m
            sum_val += data[i, j]
        end
        mean[j] = sum_val / float_m
    end
    
    # Step 2: Standard deviation
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
    
    # Step 3: Normalize data
    sqrt_m = sqrt(float_m)
    @inbounds for j in 1:n
        inv_std = 1.0 / (sqrt_m * stddev[j])
        for i in 1:m
            data[i, j] = (data[i, j] - mean[j]) * inv_std
        end
    end
    
    # Step 4: Correlation matrix (upper triangle)
    @inbounds for i in 1:(n-1)
        for j in (i+1):n
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
 Strategy 2: Threaded parallel
 - Parallelizes over columns
=============================================================================#
function kernel_correlation_threads!(
    data::Matrix{Float64},
    mean::Vector{Float64},
    stddev::Vector{Float64},
    corr::Matrix{Float64}
)
    m, n = size(data)
    float_m = Float64(m)
    
    # Mean (parallel over columns)
    @threads :static for j in 1:n
        @inbounds begin
            sum_val = 0.0
            for i in 1:m
                sum_val += data[i, j]
            end
            mean[j] = sum_val / float_m
        end
    end
    
    # Standard deviation (parallel)
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
    
    # Normalize (parallel)
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
    
    # Correlation (parallel over rows of upper triangle)
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
 - Fuses mean and stddev computation for better cache reuse
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
    
    # Fused mean and stddev (one pass per column)
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
    
    # Correlation: parallel over j (column-major friendly)
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
    
    # Mean and stddev (tiled over columns)
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

# Get kernel by strategy name
function get_kernel(strategy::AbstractString)
    kernels = Dict(
        "sequential" => kernel_correlation_seq!,
        "threads" => kernel_correlation_threads!,
        "colmajor" => kernel_correlation_colmajor!,
        "tiled" => kernel_correlation_tiled!
    )
    return get(kernels, String(strategy), kernel_correlation_seq!)
end

end # module
