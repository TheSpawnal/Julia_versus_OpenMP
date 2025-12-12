#!/bin/bash
#SBATCH --job-name=julia_polybench
#SBATCH --output=julia_polybench_%j.out
#SBATCH --error=julia_polybench_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal

# DAS-5 Julia PolyBench Single-Node Benchmark Script
# Usage: sbatch das5_single_node.sh [benchmark] [dataset]
# Example: sbatch das5_single_node.sh cholesky LARGE

# Load Julia module (adjust version as needed on DAS-5)
module load julia/1.10

# Set thread count from SLURM allocation
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parse arguments
BENCHMARK=${1:-"cholesky"}
DATASET=${2:-"LARGE"}

# Navigate to project directory
cd $HOME/julia_polybench

echo "="
echo "DAS-5 Julia PolyBench Benchmark"
echo "="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Benchmark: $BENCHMARK"
echo "Dataset: $DATASET"
echo "="

# Run the selected benchmark
case $BENCHMARK in
    "cholesky")
        julia -t $JULIA_NUM_THREADS scripts/run_cholesky.jl --dataset $DATASET --iterations 10
        ;;
    "correlation")
        julia -t $JULIA_NUM_THREADS scripts/run_correlation.jl --dataset $DATASET --iterations 10
        ;;
    "jacobi2d")
        julia -t $JULIA_NUM_THREADS scripts/run_jacobi2d.jl --dataset $DATASET --iterations 10
        ;;
    "all")
        echo "Running all benchmarks..."
        julia -t $JULIA_NUM_THREADS scripts/run_cholesky.jl --dataset $DATASET
        julia -t $JULIA_NUM_THREADS scripts/run_correlation.jl --dataset $DATASET
        julia -t $JULIA_NUM_THREADS scripts/run_jacobi2d.jl --dataset $DATASET
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Available: cholesky, correlation, jacobi2d, all"
        exit 1
        ;;
esac

echo ""
echo "Benchmark complete. Results saved to results/ directory."
