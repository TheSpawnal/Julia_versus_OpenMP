#!/bin/bash
#SBATCH --job-name=julia_polybench_all
#SBATCH --output=julia_bench_%j.out
#SBATCH --error=julia_bench_%j.err
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq

# ==============================================================================
# DAS-5 Julia PolyBench - All Benchmarks at MEDIUM Dataset
# ==============================================================================
# Usage: sbatch das5_all_benchmarks.slurm
# Submit during off-hours for longer runs: sbatch --time=02:00:00 das5_all_benchmarks.slurm
# ==============================================================================

# CRITICAL: Source DAS-5 environment (required for module system)
. /etc/bashrc
. /etc/profile.d/lmod.sh

# Load modules
module load prun
module load julia/1.11.4 2>/dev/null || module load julia

# Verify Julia
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not available. Check 'module avail julia'"
    exit 1
fi

# Environment configuration
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export JULIA_DEPOT_PATH="$HOME/.julia"

# Project paths
PROJECT_DIR="$HOME/Julia_versus_OpenMP/julia_polybench"
RESULTS_DIR="$PROJECT_DIR/results"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Header
echo "======================================================================"
echo "DAS-5 Julia PolyBench - All Benchmarks"
echo "======================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Cores:      $SLURM_CPUS_PER_TASK"
echo "Threads:    $JULIA_NUM_THREADS"
echo "Julia:      $(julia --version)"
echo "Project:    $PROJECT_DIR"
echo "======================================================================"

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }

# Dataset configuration
DATASET="MEDIUM"
ITERATIONS=10
WARMUP=5

# Run each benchmark
BENCHMARKS=("2mm" "3mm" "cholesky" "correlation" "jacobi2d" "nussinov")

for bench in "${BENCHMARKS[@]}"; do
    SCRIPT="scripts/run_${bench}.jl"
    if [[ -f "$SCRIPT" ]]; then
        echo ""
        echo ">>> Running: $bench ($DATASET)"
        echo "----------------------------------------------------------------------"
        julia -t "$JULIA_NUM_THREADS" "$SCRIPT" \
            --dataset "$DATASET" \
            --iterations "$ITERATIONS" \
            --warmup "$WARMUP" \
            --output csv
    else
        echo "SKIP: $SCRIPT not found"
    fi
done

echo ""
echo "======================================================================"
echo "All benchmarks complete"
echo "End:        $(date)"
echo "Results in: $RESULTS_DIR"
echo "======================================================================"

# List generated CSV files
echo ""
echo "Generated files:"
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null | tail -10
