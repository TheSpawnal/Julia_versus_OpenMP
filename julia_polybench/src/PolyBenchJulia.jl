module PolyBenchJulia

# Common utilities
include("common/Config.jl")
include("common/Metrics.jl")

# Kernel modules
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
export Cholesky, Correlation, Jacobi2D, Nussinov, ThreeMM, TwoMM  # Export Nussinov

# Re-export commonly used functions from Config
export configure_blas_threads, print_system_info
export DATASETS_2MM, DATASETS_3MM, DATASETS_CHOLESKY
export DATASETS_CORRELATION, DATASETS_JACOBI2D, DATASETS_NUSSINOV
export flops_2mm, flops_3mm, flops_cholesky, flops_correlation, flops_jacobi2d, flops_nussinov

# Re-export from Metrics
export BenchmarkResult, MetricsCollector, record!, print_results, export_csv, compute_efficiency

end # module