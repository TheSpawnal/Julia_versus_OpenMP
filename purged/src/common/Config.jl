module Config

export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY, DATASETS_CORRELATION, DATASETS_NUSSINOV
export DatasetConfig, get_dataset

struct DatasetConfig{T}
    name::String
    params::T
end

# 2MM: D = alpha*A*B*C + beta*D
# A: ni x nk, B: nk x nj, tmp: ni x nj, C: nj x nl, D: ni x nl
const DATASETS_2MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl), NTuple{4, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=22,   nl=24),
    "SMALL"      => (ni=40,   nj=50,   nk=70,   nl=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=210,  nl=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# 3MM: E = A*B; F = C*D; G = E*F
# A: ni x nk, B: nk x nj, E: ni x nj
# C: nj x nm, D: nm x nl, F: nj x nl
# G: ni x nl
const DATASETS_3MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl, :nm), NTuple{5, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=20,   nl=22,   nm=24),
    "SMALL"      => (ni=40,   nj=50,   nk=60,   nl=70,   nm=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=200,  nl=210,  nm=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

# Cholesky: A = L * L^T, square matrix
const DATASETS_CHOLESKY = Dict{String, NamedTuple{(:n,), Tuple{Int}}}(
    "MINI"       => (n=40,),
    "SMALL"      => (n=120,),
    "MEDIUM"     => (n=400,),
    "LARGE"      => (n=2000,),
    "EXTRALARGE" => (n=4000,)
)

# Correlation: M data points, N variables
const DATASETS_CORRELATION = Dict{String, NamedTuple{(:m, :n), Tuple{Int, Int}}}(
    "MINI"       => (m=28,   n=32),
    "SMALL"      => (m=80,   n=100),
    "MEDIUM"     => (m=240,  n=260),
    "LARGE"      => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

# Nussinov: RNA sequence length
const DATASETS_NUSSINOV = Dict{String, NamedTuple{(:n,), Tuple{Int}}}(
    "MINI"       => (n=60,),
    "SMALL"      => (n=180,),
    "MEDIUM"     => (n=500,),
    "LARGE"      => (n=2500,),
    "EXTRALARGE" => (n=5500,)
)

# FLOP calculations
function flops_2mm(ni, nj, nk, nl)
    # tmp = alpha * A * B: 2*ni*nj*nk
    # D = D + tmp * C: 2*ni*nl*nj + ni*nl
    return 2 * ni * nj * nk + 2 * ni * nl * nj + ni * nl
end

function flops_3mm(ni, nj, nk, nl, nm)
    # E = A*B: 2*ni*nj*nk
    # F = C*D: 2*nj*nl*nm
    # G = E*F: 2*ni*nl*nj
    return 2 * ni * nj * nk + 2 * nj * nl * nm + 2 * ni * nl * nj
end

function flops_cholesky(n)
    # Approximately n^3/3 for Cholesky
    return div(n^3, 3)
end

function flops_correlation(m, n)
    # Mean: n*m, stddev: 2*n*m, normalize: n*m, correlation: n^2*m
    return n * m + 2 * n * m + n * m + n * n * m
end

function flops_nussinov(n)
    # Approximately n^3 for the DP table
    return n^3
end

# Memory estimation (bytes)
function memory_2mm(ni, nj, nk, nl; T=Float64)
    sizeof(T) * (ni * nk + nk * nj + ni * nj + nj * nl + ni * nl)
end

function memory_3mm(ni, nj, nk, nl, nm; T=Float64)
    sizeof(T) * (ni * nk + nk * nj + nj * nm + nm * nl + ni * nj + nj * nl + ni * nl)
end

function memory_cholesky(n; T=Float64)
    sizeof(T) * n * n
end

function memory_correlation(m, n; T=Float64)
    sizeof(T) * (m * n + n + n + n * n)  # data + mean + stddev + corr
end

function memory_nussinov(n)
    sizeof(Int) * n * n + n  # table + sequence
end

end # module
