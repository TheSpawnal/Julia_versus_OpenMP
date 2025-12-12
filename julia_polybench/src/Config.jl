module Config

using LinearAlgebra
using Base.Threads

export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky, flops_correlation, flops_jacobi2d, flops_nussinov

# Configure BLAS threads based on Julia configuration
function configure_blas_threads()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    else
        BLAS.set_num_threads(Sys.CPU_THREADS)
    end
end

function print_system_info()
    println("Julia version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CPU threads: $(Sys.CPU_THREADS)")
end

# Dataset sizes following PolyBench specification
const DATASETS_2MM = Dict(
    "MINI" => (ni=16, nj=18, nk=22, nl=24),
    "SMALL" => (ni=40, nj=50, nk=70, nl=80),
    "MEDIUM" => (ni=180, nj=190, nk=210, nl=220),
    "LARGE" => (ni=800, nj=900, nk=1100, nl=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2200, nl=2400)
)

const DATASETS_3MM = Dict(
    "MINI" => (ni=16, nj=18, nk=20, nl=22, nm=24),
    "SMALL" => (ni=40, nj=50, nk=60, nl=70, nm=80),
    "MEDIUM" => (ni=180, nj=190, nk=200, nl=210, nm=220),
    "LARGE" => (ni=800, nj=900, nk=1000, nl=1100, nm=1200),
    "EXTRALARGE" => (ni=1600, nj=1800, nk=2000, nl=2200, nm=2400)
)

const DATASETS_CHOLESKY = Dict(
    "MINI" => (n=40,),
    "SMALL" => (n=120,),
    "MEDIUM" => (n=400,),
    "LARGE" => (n=2000,),
    "EXTRALARGE" => (n=4000,)
)

const DATASETS_CORRELATION = Dict(
    "MINI" => (m=28, n=32),
    "SMALL" => (m=80, n=100),
    "MEDIUM" => (m=240, n=260),
    "LARGE" => (m=1200, n=1400),
    "EXTRALARGE" => (m=2600, n=3000)
)

const DATASETS_JACOBI2D = Dict(
    "MINI" => (n=30, tsteps=20),
    "SMALL" => (n=90, tsteps=40),
    "MEDIUM" => (n=250, tsteps=100),
    "LARGE" => (n=1300, tsteps=500),
    "EXTRALARGE" => (n=2800, tsteps=1000)
)

const DATASETS_NUSSINOV = Dict(
    "MINI" => (n=60,),
    "SMALL" => (n=180,),
    "MEDIUM" => (n=500,),
    "LARGE" => (n=2500,),
    "EXTRALARGE" => (n=5500,)
)

# FLOPs calculation for each benchmark
function flops_2mm(ni, nj, nk, nl)
    # tmp = A*B: 2*ni*nj*nk
    # D = tmp*C + D: 2*ni*nl*nj + ni*nl (add) + ni*nl (scale)
    return 2*ni*nj*nk + 2*ni*nl*nj + 2*ni*nl
end

function flops_3mm(ni, nj, nk, nl, nm)
    # E = A*B: 2*ni*nj*nk
    # F = C*D: 2*nj*nl*nm
    # G = E*F: 2*ni*nl*nj
    return 2*ni*nj*nk + 2*nj*nl*nm + 2*ni*nl*nj
end

function flops_cholesky(n)
    return n^3 / 3
end

function flops_correlation(m, n)
    # Mean: n*m, Stddev: 2*n*m, Normalize: 2*n*m, Corr: n^2 * m
    return n*m + 2*n*m + 2*n*m + n^2*m
end

function flops_jacobi2d(n, tsteps)
    # 5-point stencil: 5 loads, 4 adds, 1 mul per point, (n-2)^2 points
    return tsteps * (n-2)^2 * 5
end

function flops_nussinov(n)
    # O(n^3) for the DP table
    return (n^3) / 6
end

end # module
