module Config
#=
Configuration Module for Julia PolyBench Benchmarks

Provides:
- BLAS thread configuration
- Dataset size constants for all benchmarks
- FLOPs calculation functions
- System info printing

Usage:
    using .Config
    configure_blas_threads()
    params = DATASETS_2MM["MEDIUM"]
    flops = flops_2mm(params.ni, params.nj, params.nk, params.nl)
=#

using LinearAlgebra
using Base.Threads

export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky, flops_correlation, flops_jacobi2d, flops_nussinov

#=============================================================================
 BLAS Configuration
 
 Critical for fair benchmarking:
 - When Julia uses multiple threads, disable BLAS threading to avoid
   over-subscription
 - When Julia is single-threaded, let BLAS use all cores
=============================================================================#
function configure_blas_threads(;verbose::Bool=false)
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        verbose && println("BLAS threads: 1 (Julia using $(Threads.nthreads()) threads)")
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
        verbose && println("BLAS threads: $(Sys.CPU_THREADS) (Julia single-threaded)")
    end
end

function print_system_info()
    println("System Configuration:")
    println("  Julia version: $(VERSION)")
    println("  Threads: $(Threads.nthreads())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  BLAS vendor: $(BLAS.vendor())")
    println("  CPU threads: $(Sys.CPU_THREADS)")
end

#=============================================================================
 Dataset Sizes - PolyBench Standard
=============================================================================#

# 2MM: D = alpha*A*B*C + beta*D
const DATASETS_2MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl), NTuple{4, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=22,   nl=24),
    "SMALL"      => (ni=40,   nj=50,   nk=70,   nl=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=210,  nl=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

# 3MM: G = (A*B)*(C*D)
const DATASETS_3MM = Dict{String, NamedTuple{(:ni, :nj, :nk, :nl, :nm), NTuple{5, Int}}}(
    "MINI"       => (ni=16,   nj=18,   nk=20,   nl=22,   nm=24),
    "SMALL"      => (ni=40,   nj=50,   nk=60,   nl=70,   nm=80),
    "MEDIUM"     => (ni=180,  nj=190,  nk=200,  nl=210,  nm=220),
    "LARGE"      => (ni=800,  nj=900,  nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

# Cholesky decomposition
const DATASETS_CHOLESKY = Dict{String, NamedTuple{(:n,), Tuple{Int}}}(
    "MINI"       => (n=40,),
    "SMALL"      => (n=120,),
    "MEDIUM"     => (n=400,),
    "LARGE"      => (n=2000,),
    "EXTRALARGE" => (n=4000,)
)

# Correlation matrix
const DATASETS_CORRELATION = Dict{String, NamedTuple{(:m, :n), Tuple{Int, Int}}}(
    "MINI"       => (m=28,   n=32),
    "SMALL"      => (m=80,   n=100),
    "MEDIUM"     => (m=240,  n=260),
    "LARGE"      => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

# Jacobi-2D stencil
const DATASETS_JACOBI2D = Dict{String, NamedTuple{(:n, :tsteps), Tuple{Int, Int}}}(
    "MINI"       => (n=30,   tsteps=20),
    "SMALL"      => (n=90,   tsteps=40),
    "MEDIUM"     => (n=250,  tsteps=100),
    "LARGE"      => (n=1300, tsteps=500),
    "EXTRALARGE" => (n=2800, tsteps=1000)
)

# Nussinov RNA folding
const DATASETS_NUSSINOV = Dict{String, NamedTuple{(:n,), Tuple{Int}}}(
    "MINI"       => (n=60,),
    "SMALL"      => (n=180,),
    "MEDIUM"     => (n=500,),
    "LARGE"      => (n=2500,),
    "EXTRALARGE" => (n=5500,)
)

#=============================================================================
 FLOPs Calculations
 
 Note: These calculate floating-point operations for the kernel computation.
 For multiply-add operations, we count 2 FLOPs (1 multiply + 1 add).
=============================================================================#

# 2MM: tmp = A*B (2*ni*nj*nk), D = tmp*C + beta*D (2*ni*nl*nj + 2*ni*nl)
function flops_2mm(ni, nj, nk, nl)
    return 2*ni*nj*nk + 2*ni*nl*nj + 2*ni*nl
end

# 3MM: E=A*B (2*ni*nj*nk), F=C*D (2*nj*nl*nm), G=E*F (2*ni*nl*nj)
function flops_3mm(ni, nj, nk, nl, nm)
    return 2*ni*nj*nk + 2*nj*nl*nm + 2*ni*nl*nj
end

# Cholesky: approximately n^3/3 FLOPs
function flops_cholesky(n)
    return Int(round(n^3 / 3))
end

# Correlation: mean (n*m), stddev (2*n*m), normalize (2*n*m), correlation (n^2*m)
function flops_correlation(m, n)
    return n*m + 2*n*m + 2*n*m + n^2*m
end

# Jacobi-2D: 5-point stencil, (n-2)^2 interior points, tsteps iterations
function flops_jacobi2d(n, tsteps)
    return tsteps * (n-2)^2 * 5
end

# Nussinov: O(n^3) dynamic programming
function flops_nussinov(n)
    return Int(round(n^3 / 6))
end

end # module Config
