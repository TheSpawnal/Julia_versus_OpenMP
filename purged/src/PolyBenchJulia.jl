module PolyBenchJulia

# Common utilities
include("common/Config.jl")
include("common/Metrics.jl")
include("common/BenchCore.jl")

# Kernel implementations
include("kernels/TwoMM.jl")
include("kernels/ThreeMM.jl")
include("kernels/Cholesky.jl")
include("kernels/Correlation.jl")
include("kernels/Nussinov.jl")
include("kernels/Jacobi2D.jl")

# Re-export modules
using .Config
using .Metrics
using .BenchCore
using .TwoMM
using .ThreeMM
using .Cholesky
using .Correlation
using .Nussinov
using .Jacobi2D

export Config, Metrics, BenchCore
export TwoMM, ThreeMM, Cholesky, Correlation, Nussinov, Jacobi2D

# Export commonly used functions
export configure_environment, check_configuration, print_system_info
export MetricsCollector, BenchmarkResult, record!, print_results, export_csv, export_json

# Version info
const VERSION = v"1.0.0"

function __init__()
    # Configure BLAS on module load
    configure_environment()
end

end # module
