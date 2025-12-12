#!/bin/bash
# Thread Scaling Analysis Script
# Usage: ./thread_scaling.sh [benchmark] [dataset]
# Example: ./thread_scaling.sh cholesky LARGE

BENCHMARK=${1:-"cholesky"}
DATASET=${2:-"LARGE"}
OUTPUT_DIR="results/scaling"

mkdir -p $OUTPUT_DIR

echo "Thread Scaling Analysis"
echo "Benchmark: $BENCHMARK"
echo "Dataset: $DATASET"
echo ""

# Thread counts to test
THREAD_COUNTS="1 2 4 8 16 32"

for THREADS in $THREAD_COUNTS; do
    echo "Running with $THREADS threads..."
    
    case $BENCHMARK in
        "cholesky")
            julia -t $THREADS scripts/run_cholesky.jl --dataset $DATASET --iterations 5 --output csv
            ;;
        "correlation")
            julia -t $THREADS scripts/run_correlation.jl --dataset $DATASET --iterations 5 --output csv
            ;;
        "jacobi2d")
            julia -t $THREADS scripts/run_jacobi2d.jl --dataset $DATASET --iterations 5 --output csv
            ;;
    esac
    
    # Move results to scaling directory
    mv results/${BENCHMARK}_${DATASET}_*.csv $OUTPUT_DIR/ 2>/dev/null
done

echo ""
echo "Scaling analysis complete."
echo "Results in: $OUTPUT_DIR"