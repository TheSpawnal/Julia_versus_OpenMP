#!/bin/bash
# DAS-5 Julia PolyBench Helper Functions
# Source this: source das5_helpers.sh

export JULIA_PROJECT_DIR="$HOME/Julia_versus_OpenMP/julia_polybench"

# Submit single benchmark
jl_submit() {
    local bench="${1:-2mm}"
    local dataset="${2:-MEDIUM}"
    local threads="${3:-16}"
    
    sbatch --job-name="jl_${bench}" \
           --output="${bench}_%j.out" \
           --time=00:15:00 \
           -N 1 --cpus-per-task=${threads} --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=${threads}; export OPENBLAS_NUM_THREADS=1; cd $JULIA_PROJECT_DIR; julia -t ${threads} scripts/run_${bench}.jl --dataset ${dataset} --output csv"
    
    echo "Submitted: $bench @ $dataset with $threads threads"
}

# Submit all benchmarks
jl_submit_all() {
    local dataset="${1:-MEDIUM}"
    for bench in 2mm 3mm cholesky correlation jacobi2d nussinov; do
        jl_submit "$bench" "$dataset" 16
    done
}

# Submit scaling study
jl_scaling() {
    local bench="${1:-2mm}"
    local dataset="${2:-MEDIUM}"
    for threads in 1 2 4 8 16; do
        jl_submit "$bench" "$dataset" "$threads"
    done
}

# Monitor jobs
jl_status() {
    squeue -u $USER
}

# Cancel all
jl_cancel() {
    scancel -u $USER
    echo "All jobs cancelled"
}

# View latest output
jl_tail() {
    local pattern="${1:-julia}"
    tail -f $(ls -t ${pattern}*.out 2>/dev/null | head -1)
}

# Aggregate results
jl_aggregate() {
    cd "$JULIA_PROJECT_DIR/results"
    local outfile="aggregate_$(date +%Y%m%d_%H%M%S).csv"
    head -1 $(ls *.csv 2>/dev/null | head -1) > "$outfile"
    for f in *.csv; do
        [[ "$f" != "$outfile" ]] && tail -n +2 "$f" >> "$outfile"
    done
    echo "Created: $outfile"
}

echo "Julia PolyBench helpers loaded"
echo "Commands: jl_submit, jl_submit_all, jl_scaling, jl_status, jl_cancel, jl_tail, jl_aggregate"
