# PolyBench2MM Usage Examples
#do you want use the memory allocation tracker ? In the cli : 
#$ julia track-allocation=user

#basic usage in REPL:
# julia> include("PolyBench2MM.jl")
# julia> using .PolyBench2MM

# 1. Run all non-distributed implementations (default)
# julia> PolyBench2MM.main()

# 2. Run a specific implementation
# julia> PolyBench2MM.main(implementation="blas")
# julia> PolyBench2MM.main(implementation="seq")
# julia> PolyBench2MM.main(implementation="simd")
# julia> PolyBench2MM.main(implementation="threads")

# 3. Run with custom datasets (default is MINI, SMALL, MEDIUM)
# julia> PolyBench2MM.main(datasets=["SMALL", "MEDIUM", "LARGE"])

# 4. Verify implementation correctness
# julia> PolyBench2MM.verify_implementations()

# ==== DISTRIBUTED IMPLEMENTATIONS =====
# First, add worker processes:
# julia> using Distributed
# julia> addprocs(4)  # Add 4 worker processes

# Then load the module on all workers:
# julia> @everywhere include("PolyBench2MM.jl")
# julia> @everywhere using .PolyBench2MM

# Now run distributed benchmarks:
# julia> PolyBench2MM.main(distributed=true)

# Or run a specific distributed implementation:
# julia> PolyBench2MM.main(implementation="distributed", distributed=true)
# julia> PolyBench2MM.main(implementation="dist3", distributed=true)

# The distributed mode will automatically find the optimal number of processes
# for MINI, SMALL, and MEDIUM datasets by testing different process counts

# ===== SETTING THREAD COUNT =====
# For multithreaded implementation, set the number of threads before starting Julia:
# $ export JULIA_NUM_THREADS=8
# $ julia

# Or start Julia with thread count:
# $ julia --threads 8