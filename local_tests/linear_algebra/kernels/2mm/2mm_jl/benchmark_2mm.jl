using BenchmarkTools
using Statistics
using Distributed

# Import kernels
include("kernel_2mm.jl")

# Setup distributed if needed
if nprocs() == 1
    addprocs(8)  # Use 8 workers to match your CPU's logical processors
end

# Make distributed kernels available
@everywhere include("distributed_kernels.jl")

function init_array(ni, nj, nk, nl)
    alpha = 1.5
    beta = 1.2
    
    A = zeros(ni, nk)
    B = zeros(nk, nj)
    C = zeros(nj, nl)
    D = zeros(ni, nl)
    tmp = zeros(ni, nj)
    
    # Initialize arrays like in the C benchmark
    for i in 1:ni
        for j in 1:nk
            A[i,j] = ((i*j+1) % ni) / ni
        end
    end
    
    for i in 1:nk
        for j in 1:nj
            B[i,j] = (i*(j+1) % nj) / nj
        end
    end
    
    for i in 1:nj
        for j in 1:nl
            C[i,j] = ((i*(j+3)+1) % nl) / nl
        end
    end
    
    for i in 1:ni
        for j in 1:nl
            D[i,j] = (i*(j+2) % nk) / nk
        end
    end
    
    return alpha, beta, A, B, C, D, tmp
end

function benchmark_2mm(ni, nj, nk, nl)
    # Initialize matrices
    alpha, beta, A, B, C, D, tmp = init_array(ni, nj, nk, nl)
    
    # Make copies for each implementation
    A_seq = copy(A)
    B_seq = copy(B)
    C_seq = copy(C)
    D_seq = copy(D)
    tmp_seq = copy(tmp)
    
    A_task = copy(A)
    B_task = copy(B)
    C_task = copy(C)
    D_task = copy(D)
    tmp_task = copy(tmp)
    
    A_coarse = copy(A)
    B_coarse = copy(B)
    C_coarse = copy(C)
    D_coarse = copy(D)
    tmp_coarse = copy(tmp)
    
    # Benchmark sequential implementation
    t_seq = @benchmark kernel_2mm_sequential!($ni, $nj, $nk, $nl, $alpha, $beta, $tmp_seq, $A_seq, $B_seq, $C_seq, $D_seq)
    
    # Benchmark task-based implementation
    t_task = @benchmark kernel_2mm_tasks!($ni, $nj, $nk, $nl, $alpha, $beta, $tmp_task, $A_task, $B_task, $C_task, $D_task)
    
    # Benchmark distributed coarse-grained implementation
    t_coarse = @benchmark kernel_2mm_dist_coarse!($ni, $nj, $nk, $nl, $alpha, $beta, $tmp_coarse, $A_coarse, $B_coarse, $C_coarse, $D_coarse)
    
    # Calculate median times
    t_seq_median = median(t_seq.times) / 1e9  # Convert to seconds
    t_task_median = median(t_task.times) / 1e9
    t_coarse_median = median(t_coarse.times) / 1e9
    
    # Calculate speedups
    speedup_task = t_seq_median / t_task_median
    speedup_coarse = t_seq_median / t_coarse_median
    
    # Calculate efficiency
    num_workers = nworkers()
    efficiency_task = speedup_task / Threads.nthreads() * 100
    efficiency_coarse = speedup_coarse / num_workers * 100
    
    # Print results
    println("Matrix sizes: A($(ni)×$(nk)), B($(nk)×$(nj)), C($(nj)×$(nl)), D($(ni)×$(nl))")
    println("Number of threads: $(Threads.nthreads()), Number of workers: $(num_workers)")
    println()
    println("Implementation | Time (s) | Speedup | Efficiency (%)")
    println("------------- | -------- | ------- | -------------")
    println("Sequential    | $(round(t_seq_median, digits=6)) | 1.00    | 100.00")
    println("Task-based    | $(round(t_task_median, digits=6)) | $(round(speedup_task, digits=2))    | $(round(efficiency_task, digits=2))")
    println("Coarse-grained| $(round(t_coarse_median, digits=6)) | $(round(speedup_coarse, digits=2))    | $(round(efficiency_coarse, digits=2))")
    
    return (t_seq_median, t_task_median, t_coarse_median)
end

function run_benchmarks()
    # Define dataset sizes from polybench_2mm.h
    datasets = [
        (40, 50, 70, 80),      # SMALL_DATASET
        (180, 190, 210, 220),   # MEDIUM_DATASET
        (800, 900, 1100, 1200)  # LARGE_DATASET
    ]
    
    for (ni, nj, nk, nl) in datasets
        println("\nRunning benchmark for dataset size: $(ni)×$(nj)×$(nk)×$(nl)")
        println("---------------------------------------------")
        benchmark_2mm(ni, nj, nk, nl)
    end
end