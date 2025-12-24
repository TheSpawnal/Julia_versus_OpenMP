#!/bin/bash
#===============================================================================
# DAS-5 Thread Scaling Study Script
#
# Runs benchmarks across multiple thread counts for scaling analysis.
# Generates CSV output compatible with visualize_benchmarks.py
#
# USAGE:
#   sbatch das5_scaling_study.slurm [benchmark] [dataset]
#   sbatch das5_scaling_study.slurm 2mm LARGE
#
# OUTPUT:
#   results/scaling_study_{benchmark}_{dataset}_{timestamp}.csv
#===============================================================================

#SBATCH --job-name=scaling_study
#SBATCH --output=scaling_study_%j.out
#SBATCH --error=scaling_study_%j.err
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq

# Source environment
. /etc/bashrc
. /etc/profile.d/lmod.sh

# Load modules
module load prun
module load julia/1.10

# Parse arguments
BENCHMARK=${1:-"2mm"}
DATASET=${2:-"LARGE"}
THREAD_COUNTS="1 2 4 8 16"

# Set BLAS to single-threaded (Julia handles parallelism)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Navigate to project
cd $HOME/Julia_versus_OpenMP/julia_polybench

# Create output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="results/scaling_study_${BENCHMARK}_${DATASET}_${TIMESTAMP}.csv"
mkdir -p results

echo "========================================================================"
echo "DAS-5 Thread Scaling Study"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Benchmark: $BENCHMARK"
echo "Dataset: $DATASET"
echo "Thread counts: $THREAD_COUNTS"
echo "Output: $OUTPUT_FILE"
echo "========================================================================"

# Write CSV header
echo "benchmark,dataset,strategy,threads,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency,verified" > $OUTPUT_FILE

# Run for each thread count
for THREADS in $THREAD_COUNTS; do
    echo ""
    echo ">>> Running with $THREADS thread(s)..."
    
    export JULIA_NUM_THREADS=$THREADS
    
    # Run benchmark and capture output
    SCRIPT="scripts/run_${BENCHMARK}.jl"
    
    if [ -f "$SCRIPT" ]; then
        # Run and append to CSV (skip header line from each run)
        julia -t $THREADS $SCRIPT \
            --dataset $DATASET \
            --iterations 10 \
            --output csv 2>&1 | tee -a "${OUTPUT_FILE%.csv}_${THREADS}t.log"
        
        # Extract CSV data from results directory (most recent file)
        LATEST=$(ls -t results/${BENCHMARK}_${DATASET}_*.csv 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            # Append data rows (skip header)
            tail -n +2 "$LATEST" >> $OUTPUT_FILE
        fi
    else
        echo "Warning: Script not found: $SCRIPT"
    fi
done

echo ""
echo "========================================================================"
echo "Scaling study complete"
echo "Results: $OUTPUT_FILE"
echo "========================================================================"

# Generate quick summary
echo ""
echo "Quick Summary:"
echo "--------------"
column -t -s',' $OUTPUT_FILE | head -20
