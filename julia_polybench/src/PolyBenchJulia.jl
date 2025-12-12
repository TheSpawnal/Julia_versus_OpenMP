module PolyBenchJulia

# Common utilities
include("common/Config.jl")
include("common/Metrics.jl")
include("common/BenchCore.jl")

# Kernel implementations
include("kernels/Cholesky.jl")
include("kernels/Correlation.jl")
include("kernels/Jacobi2D.jl")
include("kernels/Nussinov.jl")

# Re-export modules
using .Config
using .Metrics
using .BenchCore
using .Cholesky
using .Correlation
using .Jacobi2D
using .Nussinov

export Config, Metrics, BenchCore
export Cholesky, Correlation, Jacobi2D, Nussinov

# Export common functions
export configure_blas_threads, print_system_info
export MetricsCollector, BenchmarkResult, record!, print_results, export_csv
export benchmark_kernel, TimingResult

# Version
const VERSION = v"1.0.0"

function __init__()
    configure_blas_threads()
end

end # module
