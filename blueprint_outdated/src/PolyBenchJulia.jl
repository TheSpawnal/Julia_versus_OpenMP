module PolyBenchJulia
#=
PolyBench Julia - High Performance Computing Benchmark Suite

A Julia implementation of selected PolyBench kernels for benchmarking
parallel computing strategies against OpenMP implementations.

Kernels implemented:
- 2MM: D = alpha*A*B*C + beta*D (chained matrix multiplication)
- 3MM: G = (A*B)*(C*D) (triple matrix multiplication)
- Cholesky: Cholesky decomposition
- Correlation: Correlation matrix computation
- Jacobi2D: 2D Jacobi stencil iteration
- Nussinov: RNA secondary structure prediction (dynamic programming)

Each kernel implements multiple parallelization strategies:
- sequential: Baseline with SIMD optimization
- threads_static: Static thread scheduling
- threads_dynamic: Dynamic thread scheduling
- tiled: Cache-blocked with parallel tiles
- blas: BLAS/LAPACK reference implementation
- tasks: Task-based parallelism

Usage:
    using PolyBenchJulia
    
    # Configure BLAS
    configure_blas_threads()
    
    # Get dataset parameters
    params = DATASETS_2MM["MEDIUM"]
    
    # Calculate expected FLOPs
    flops = flops_2mm(params.ni, params.nj, params.nk, params.nl)

Author:Aldric de Jacquelin;  SpawnAl / Falkor collaboration
Project: Scientific Parallel Computing & Multithreading with Julia
=#

# Common utilities
include("common/Config.jl")
include("common/Metrics.jl")
include("common/BenchCore.jl")

#submodules
include("kernels/Cholesky.jl")
include("kernels/Correlation.jl")
include("kernels/Jacobi2D.jl")
include("kernels/Nussinov.jl")  # Add Nussinov
include("kernels/TwoMM.jl")
include("kernels/ThreeMM.jl")

# Import and export submodules
using .Config
using .Metrics
using .Cholesky
using .Correlation
using .Jacobi2D
using .Nussinov  # Import Nussinov
using .TwoMM
using .ThreeMM

# Export all submodules
export Config, Metrics
export Cholesky, Correlation, Jacobi2D, Nussinov, ThreeMM, TwoMM 

# Re-export commonly used functions from Config
export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky, flops_correlation, flops_jacobi2d, flops_nussinov

# Re-export from Metrics
export BenchmarkResult, MetricsCollector, record!, print_results, export_csv, compute_efficiency

# Re-export from BenchCore
export TimingResult, benchmark_kernel, time_kernel_simple

# Re-export 2MM kernel functions
export init_2mm!, reset_2mm!
export kernel_2mm_seq!, kernel_2mm_threads_static!, kernel_2mm_threads_dynamic!
export kernel_2mm_tiled!, kernel_2mm_blas!, kernel_2mm_tasks!
export STRATEGIES_2MM

# Re-export 3MM kernel functions
export init_3mm!, reset_3mm!
export kernel_3mm_seq!, kernel_3mm_threads_static!, kernel_3mm_threads_dynamic!
export kernel_3mm_tiled!, kernel_3mm_blas!, kernel_3mm_tasks!
export STRATEGIES_3MM

end # module PolyBenchJulia
