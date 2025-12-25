#!/bin/bash
# =============================================================================
# Julia PolyBench HPC Deployment - Essential Commands Reference
# =============================================================================
# This file contains all the essential commands for running the benchmark suite
# on both local machines and DAS-5 cluster.

# =============================================================================
# PART 1: LOCAL DEVELOPMENT COMMANDS
# =============================================================================

# --- System Configuration Check ---
julia -e '
using LinearAlgebra
println("=== System Configuration ===")
println("Julia version: ", VERSION)
println("Julia threads: ", Threads.nthreads())
println("BLAS threads:  ", BLAS.get_num_threads())
println("CPU threads:   ", Sys.CPU_THREADS)
println("Memory:        ", round(Sys.total_memory() / 1024^3, digits=1), " GB")
println("BLAS vendor:   ", BLAS.vendor())
'

# --- Starting Julia with Threads ---
julia -t 8                      # 8 threads
julia --threads=auto            # Auto-detect optimal
julia -t 4,1                    # 4 default + 1 interactive thread

# --- Environment Variables ---
export JULIA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1   # Critical: Set to 1 for multithreaded Julia

# --- Run Single Benchmark ---
julia -t 8 scripts/run_2mm.jl --dataset MEDIUM
julia -t 8 scripts/run_3mm.jl --dataset LARGE
julia -t 8 scripts/run_correlation.jl --dataset MEDIUM
julia -t 4 scripts/run_cholesky.jl --dataset SMALL

# --- Run All Benchmarks ---
julia -t 8 scripts/run_all.jl --datasets SMALL,MEDIUM,LARGE --output csv

# --- Thread Scaling Study ---
for t in 1 2 4 8 16; do
    echo "=== Threads: $t ===" >> scaling_results.log
    julia -t $t scripts/run_2mm.jl --dataset LARGE >> scaling_results.log 2>&1
done

# --- Verify Correctness ---
julia -t 4 -e '
include("src/kernels/ThreeMM.jl")
using .PolyBench3MM
verify_implementations("SMALL")
'

# --- Profile for Flame Graphs ---
julia -t 8 scripts/run_2mm.jl --dataset LARGE --profile

# Using ProfileView.jl:
julia -t 8 -e '
using Profile, ProfileView
include("src/kernels/TwoMM.jl")
# Warmup
# ... run once ...
Profile.clear()
@profile for _ in 1:100
    # ... run benchmark ...
end
ProfileView.svgwrite("flamegraph.svg")
'

# --- Memory Analysis ---
julia --track-allocation=user -t 8 scripts/run_2mm.jl --dataset MEDIUM
# View allocation results:
# Look for *.mem files in src/kernels/

# --- Export Results ---
julia -t 8 scripts/run_2mm.jl --dataset LARGE --output csv,json

# --- Visualization ---
python3 scripts/visualize_benchmarks.py results/benchmark_LARGE_*.csv

# =============================================================================
# PART 2: DAS-5 CLUSTER COMMANDS
# =============================================================================

# --- Interactive Node Request ---
srun -N 1 -n 1 -c 32 --time=01:00:00 --pty bash

# --- Check Available Resources ---
sinfo
squeue -u $USER

# --- Load Modules ---
module load julia/1.10
module list

# --- Single Node SLURM Script ---
cat > benchmark_single_node.slurm << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH --job-name=julia_polybench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# Load Julia module
module load julia/1.10

# Set threading environment
export JULIA_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1

# Navigate to project directory
cd $HOME/Julia_versus_OpenMP/julia_polybench

# Run benchmarks
echo "Starting benchmark at $(date)"
echo "Node: $(hostname)"
echo "Threads: $JULIA_NUM_THREADS"

julia scripts/run_all.jl --datasets MEDIUM,LARGE,EXTRALARGE --output csv

echo "Completed at $(date)"
SLURM_SCRIPT

# Submit job
sbatch benchmark_single_node.slurm

# --- Multi-Node Distributed Job ---
cat > benchmark_distributed.slurm << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH --job-name=julia_distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=distributed_%j.out
#SBATCH --error=distributed_%j.err

module load julia/1.10

# Use 1 thread per process (MPI-style)
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd $HOME/Julia_versus_OpenMP/julia_polybench

# Get node list
NODES=$(scontrol show hostnames $SLURM_NODELIST)
echo "Allocated nodes: $NODES"

# Create machine file
MACHINE_FILE="machines_$SLURM_JOB_ID.txt"
for node in $NODES; do
    echo "$node" >> $MACHINE_FILE
done

# Run with distributed Julia
julia --machine-file=$MACHINE_FILE scripts/run_distributed.jl --datasets LARGE,EXTRALARGE

rm $MACHINE_FILE
SLURM_SCRIPT

sbatch benchmark_distributed.slurm

# --- Monitor Jobs ---
squeue -u $USER
scancel <job_id>                # Cancel specific job
scancel -u $USER                # Cancel all your jobs

# --- Check Job Output ---
tail -f benchmark_*.out         # Live output
cat benchmark_*.err             # Check errors

# =============================================================================
# PART 3: TROUBLESHOOTING COMMANDS
# =============================================================================

# --- Check BLAS Configuration ---
julia -e '
using LinearAlgebra
println("BLAS vendor: ", BLAS.vendor())
println("BLAS threads: ", BLAS.get_num_threads())

# Set BLAS threads
BLAS.set_num_threads(1)
println("After setting: ", BLAS.get_num_threads())
'

# --- Debug Threading Issues ---
julia -t 8 -e '
using Base.Threads
println("Thread pool info:")
println("  nthreads(): ", nthreads())
println("  threadid(): ", threadid())

# Test threading
results = zeros(nthreads())
@threads for i in 1:nthreads()
    results[threadid()] = threadid()
end
println("Threads used: ", unique(results))
'

# --- Check Memory Usage ---
julia -e '
println("Total memory: ", round(Sys.total_memory() / 1024^3, digits=2), " GB")
println("Free memory:  ", round(Sys.free_memory() / 1024^3, digits=2), " GB")
'

# --- Warm-up Test ---
julia -t 8 -e '
using LinearAlgebra

n = 1000
A = rand(n, n)
B = rand(n, n)

# First run (includes JIT)
@time C = A * B

# Subsequent runs (actual performance)
@time C = A * B
@time C = A * B
'

# =============================================================================
# PART 4: QUICK REFERENCE
# =============================================================================

# Essential 10 Commands for Report Generation:

# 1. Configure environment
export JULIA_NUM_THREADS=8 && export OPENBLAS_NUM_THREADS=1

# 2. Verify all implementations
julia -t 8 -e 'include("scripts/verify_all.jl")'

# 3. Run small dataset (quick test)
julia -t 8 scripts/run_all.jl --datasets SMALL --output csv

# 4. Run full benchmark suite
julia -t 8 scripts/run_all.jl --datasets SMALL,MEDIUM,LARGE --output csv,json

# 5. Thread scaling analysis
for t in 1 2 4 8; do julia -t $t scripts/run_2mm.jl --dataset LARGE; done > scaling.log

# 6. Generate profile data
julia -t 8 scripts/run_2mm.jl --dataset LARGE --profile

# 7. Export to plots
python3 scripts/visualize_benchmarks.py results/*.csv --output-dir plots/

# 8. Submit DAS-5 job
sbatch benchmark_single_node.slurm

# 9. Check DAS-5 results
tail -f benchmark_*.out

# 10. Generate final report
julia scripts/generate_report.jl --results-dir results/ --output report.pdf

# =============================================================================
# NOTES
# =============================================================================

# Key Points:
# - Always set OPENBLAS_NUM_THREADS=1 when using Julia threading
# - Use julia -t N to set thread count (cannot change at runtime)
# - First run includes JIT compilation - always warmup before benchmarking
# - Use BenchmarkTools.jl for accurate timing
# - Check allocations - should be 0 for optimized kernels
# - Efficiency = (speedup / threads) * 100% for threaded strategies
# - Sequential efficiency should be 100%, not based on thread count
