#!/bin/bash
# Comprehensive benchmarking script for 2MM OpenMP implementations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 2MM OpenMP Comprehensive Benchmark ===${NC}"
echo "Date: $(date)"
echo "System: $(uname -a)"
echo "CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"
echo "Cores: $(nproc)"
echo ""

# Check for required tools
check_requirements() {
    command -v numactl >/dev/null 2>&1 || { echo "Warning: numactl not found. NUMA optimizations unavailable."; }
    command -v perf >/dev/null 2>&1 || { echo "Warning: perf not found. Performance counters unavailable."; }
}

# Function to run benchmark
run_benchmark() {
    local impl=$1
    local dataset=$2
    local threads=$3
    local output=""
    
    if [ -n "$threads" ]; then
        export OMP_NUM_THREADS=$threads
    fi
    
    # Run 5 times and get average
    total_time=0
    for i in {1..5}; do
        if [ -x "./$impl" ]; then
            time_output=$(./$impl 2>&1 | grep -E "[0-9]+\.[0-9]+" | head -1)
            if [ -n "$time_output" ]; then
                time_value=$(echo $time_output | grep -oE "[0-9]+\.[0-9]+")
                total_time=$(echo "$total_time + $time_value" | bc)
            fi
        fi
    done
    
    if [ $(echo "$total_time > 0" | bc) -eq 1 ]; then
        avg_time=$(echo "scale=6; $total_time / 5" | bc)
        echo "$avg_time"
    else
        echo "N/A"
    fi
}

# Build all implementations
build_all() {
    echo -e "${GREEN}Building all implementations...${NC}"
    for dataset in MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET; do
        echo "Building for $dataset..."
        make clean > /dev/null 2>&1
        make DATASET=$dataset all > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            mkdir -p "build_$dataset"
            mv 2mm_* "build_$dataset/" 2>/dev/null
        fi
    done
}

# Thread scaling test
thread_scaling_test() {
    local dataset=$1
    echo -e "\n${BLUE}Thread Scaling Test - $dataset${NC}"
    echo "Implementation,1,2,4,8,16,32"
    
    cd "build_$dataset" || return
    
    for impl in 2mm_omp_static 2mm_omp_tiled 2mm_omp_hybrid; do
        if [ -x "$impl" ]; then
            echo -n "$impl"
            for threads in 1 2 4 8 16 32; do
                if [ $threads -le $(nproc) ]; then
                    time=$(run_benchmark "$impl" "$dataset" "$threads")
                    echo -n ",$time"
                else
                    echo -n ",N/A"
                fi
            done
            echo ""
        fi
    done
    
    cd ..
}

# NUMA placement test
numa_test() {
    if ! command -v numactl >/dev/null 2>&1; then
        echo "NUMA test skipped - numactl not available"
        return
    fi
    
    echo -e "\n${BLUE}NUMA Placement Test${NC}"
    local dataset="LARGE_DATASET"
    cd "build_$dataset" || return
    
    # Default placement
    echo -n "Default placement: "
    time=$(run_benchmark "2mm_omp_tiled" "$dataset" "$(nproc)")
    echo "$time seconds"
    
    # Interleaved memory
    echo -n "Interleaved memory: "
    time=$(numactl --interleave=all ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "$time seconds"
    
    # Local memory only
    echo -n "Local memory: "
    time=$(numactl --localalloc ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "$time seconds"
    
    cd ..
}

# Performance counter analysis
perf_analysis() {
    if ! command -v perf >/dev/null 2>&1; then
        echo "Performance counter analysis skipped - perf not available"
        return
    fi
    
    echo -e "\n${BLUE}Performance Counter Analysis${NC}"
    local dataset="LARGE_DATASET"
    cd "build_$dataset" || return
    
    for impl in 2mm_seq 2mm_omp_tiled; do
        if [ -x "$impl" ]; then
            echo -e "\n${GREEN}$impl:${NC}"
            perf stat -e cache-misses,cache-references,instructions,cycles ./$impl 2>&1 | \
                grep -E "(cache-misses|cache-references|instructions|cycles)" | \
                grep -v "seconds"
        fi
    done
    
    cd ..
}

# Compiler comparison
compiler_comparison() {
    echo -e "\n${BLUE}Compiler Comparison${NC}"
    local dataset="LARGE_DATASET"
    
    for cc in gcc clang icc; do
        if command -v $cc >/dev/null 2>&1; then
            echo -e "\n${GREEN}Building with $cc...${NC}"
            make clean > /dev/null 2>&1
            make CC=$cc DATASET=$dataset 2mm_omp_tiled > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -n "$cc: "
                time=$(OMP_NUM_THREADS=$(nproc) ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
                echo "$time seconds"
            fi
        fi
    done
}

# Environment optimization test
env_optimization_test() {
    echo -e "\n${BLUE}Environment Optimization Test${NC}"
    local dataset="LARGE_DATASET"
    cd "build_$dataset" || return
    
    # Default
    echo -n "Default environment: "
    time=$(OMP_NUM_THREADS=$(nproc) ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "$time seconds"
    
    # With thread binding
    echo -n "With OMP_PROC_BIND=true: "
    time=$(OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=true ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "$time seconds"
    
    # With different scheduling
    echo -n "With OMP_SCHEDULE=guided: "
    time=$(OMP_NUM_THREADS=$(nproc) OMP_SCHEDULE="guided,4" ./2mm_omp_tiled 2>&1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "$time seconds"
    
    cd ..
}

# Main execution
main() {
    check_requirements
    
    # Build all implementations
    build_all
    
    # Basic performance comparison
    echo -e "\n${BLUE}Performance Comparison (using all cores)${NC}"
    echo "Dataset,Sequential,OMP_Static,OMP_Tiled,OMP_Tasks,OMP_Hybrid"
    
    for dataset in MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET; do
        echo -n "$dataset"
        cd "build_$dataset" || continue
        
        for impl in 2mm_seq 2mm_omp_static 2mm_omp_tiled 2mm_omp_tasks 2mm_omp_hybrid; do
            if [ -x "$impl" ]; then
                if [[ "$impl" == "2mm_seq" ]]; then
                    time=$(run_benchmark "$impl" "$dataset" "1")
                else
                    time=$(run_benchmark "$impl" "$dataset" "$(nproc)")
                fi
                echo -n ",$time"
            else
                echo -n ",N/A"
            fi
        done
        echo ""
        cd ..
    done
    
    # Thread scaling tests
    for dataset in SMALL_DATASET LARGE_DATASET; do
        thread_scaling_test $dataset
    done
    
    # NUMA test
    numa_test
    
    # Performance counter analysis
    perf_analysis
    
    # Compiler comparison
    compiler_comparison
    
    # Environment optimization
    env_optimization_test
    
    echo -e "\n${GREEN}Benchmark complete!${NC}"
}

# Run main
main