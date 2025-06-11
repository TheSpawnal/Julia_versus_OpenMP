# PolyBenchCholesky Usage Examples

# Basic usage in REPL:
# julia> include("PolyBenchCholesky.jl")
# julia> using .PolyBenchCholesky

# 1. Run all non-distributed implementations (default)
# julia> PolyBenchCholesky.main()

# 2. Run a specific implementation
# julia> PolyBenchCholesky.main(implementation="blas")
# julia> PolyBenchCholesky.main(implementation="seq")
# julia> PolyBenchCholesky.main(implementation="simd")
# julia> PolyBenchCholesky.main(implementation="threads")
# julia> PolyBenchCholesky.main(implementation="blocked")

# 3. Run with custom datasets (default is MINI, SMALL, MEDIUM, LARGE)
# julia> PolyBenchCholesky.main(datasets=["SMALL", "MEDIUM", "LARGE"])

# 4. Verify implementation correctness
# julia> PolyBenchCholesky.verify_implementations()
# julia> PolyBenchCholesky.verify_implementations("MEDIUM")  # With larger dataset

# ==== DISTRIBUTED IMPLEMENTATIONS =====
# First, add worker processes:
# julia> using Distributed
# julia> addprocs(4)  # Add 4 worker processes

# Then load the module on all workers:
# julia> @everywhere include("PolyBenchCholesky.jl")
# julia> @everywhere using .PolyBenchCholesky

# Now run distributed benchmarks:
# julia> PolyBenchCholesky.main(distributed=true)

# Or run a specific distributed implementation:
# julia> PolyBenchCholesky.main(implementation="distributed", distributed=true)
# julia> PolyBenchCholesky.main(implementation="dist_col", distributed=true)

# ===== SETTING THREAD COUNT =====
# For multithreaded implementation, set the number of threads before starting Julia:
# $ export JULIA_NUM_THREADS=8
# $ julia

# Or start Julia with thread count:
# $ julia --threads 8

# ===== EXAMPLE COMPLETE WORKFLOW =====
# 1. Start Julia with threads and prepare for distributed computing:
# $ julia --threads 8

# 2. In Julia REPL:
using Distributed
addprocs(4)  # Add 4 worker processes

# 3. Load the module everywhere:
@everywhere include("PolyBenchCholesky.jl")
@everywhere using .PolyBenchCholesky

# 4. Verify correctness first:
PolyBenchCholesky.verify_implementations("SMALL")

# 5. Run comprehensive benchmarks:
PolyBenchCholesky.main(distributed=true, datasets=["SMALL", "MEDIUM", "LARGE"])

# 6. Compare specific implementations:
PolyBenchCholesky.main(implementation="seq", datasets=["LARGE"])
PolyBenchCholesky.main(implementation="blas", datasets=["LARGE"])
PolyBenchCholesky.main(implementation="dist_col", distributed=true, datasets=["LARGE"])

# ===== DEBUGGING AND ANALYSIS =====
# To see the actual matrix transformations (small matrices only):
# julia> A = PolyBenchCholesky.init_array(5)
# julia> PolyBenchCholesky.print_matrix(A, name="Original Matrix")
# julia> PolyBenchCholesky.kernel_cholesky_seq!(copy(A))
# julia> PolyBenchCholesky.print_matrix(A, name="After Cholesky")

# ===== MEMORY PROFILING =====
# To track memory allocations:
# $ julia --track-allocation=user
# Then run your benchmarks as usual