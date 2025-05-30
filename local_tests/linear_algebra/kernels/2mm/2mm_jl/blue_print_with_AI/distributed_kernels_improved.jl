using LinearAlgebra
using Statistics
# Enhanced worker function with block-based processing for better cache locality
@everywhere function kernel_2mm_worker_improved!(start_row, end_row, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # We'll use a block size that fits well in L1 cache
    block_size = 32  # Adjust based on target architecture
    
    # First matrix multiplication for a block of rows (tmp = alpha*A*B)
    for i in start_row:end_row
        for jb in 1:block_size:nj
            j_end = min(jb + block_size - 1, nj)
            
            # Initialize tmp block
            for j in jb:j_end
                tmp[i,j] = 0.0
            end
            
            # Blocked multiplication
            for kb in 1:block_size:nk
                k_end = min(kb + block_size - 1, nk)
                
                for k in kb:k_end
                    for j in jb:j_end
                        @inbounds tmp[i,j] += alpha * A[i,k] * B[k,j]
                    end
                end
            end
        end
    end
    
    # Second matrix multiplication for a block of rows (D = beta*D + tmp*C)
    for i in start_row:end_row
        for jb in 1:block_size:nl
            j_end = min(jb + block_size - 1, nl)
            
            # Scale D block by beta
            for j in jb:j_end
                @inbounds D[i,j] *= beta
            end
            
            # Blocked multiplication
            for kb in 1:block_size:nj
                k_end = min(kb + block_size - 1, nj)
                
                for k in kb:k_end
                    for j in jb:j_end
                        @inbounds D[i,j] += tmp[i,k] * C[k,j]
                    end
                end
            end
        end
    end
end

function kernel_2mm_dist_coarse_improved!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Calculate how many rows each worker should process
    num_workers = nworkers()
    rows_per_worker = ceil(Int, ni / num_workers)
    
    # Create a performance monitoring data structure
    perf_data = Dict()
    for w_idx in 1:num_workers
        perf_data[w_idx] = Dict(
            "start_time" => 0.0,
            "end_time" => 0.0,
            "cpu_usage" => 0.0
        )
    end
    
    # Create local buffers to store results
    tmp_local = zeros(size(tmp))
    D_local = copy(D)
    
    # First matrix multiplication: tmp = alpha*A*B
    @sync for w_idx in 1:num_workers
        # Determine row range for this worker
        start_row = (w_idx - 1) * rows_per_worker + 1
        end_row = min(w_idx * rows_per_worker, ni)
        
        if start_row <= end_row
            # Compute block on worker w
            w = workers()[w_idx]
            @async begin
                # Record start time
                perf_data[w_idx]["start_time"] = time()
                
                # Perform computation for first phase
                result = @spawnat w begin
                    local_tmp = zeros(end_row - start_row + 1, nj)
                    
                    for i in 1:(end_row - start_row + 1)
                        for j in 1:nj
                            for k in 1:nk
                                local_tmp[i,j] += alpha * A[start_row + i - 1, k] * B[k,j]
                            end
                        end
                    end
                    
                    local_tmp
                end
                
                # Collect results from this worker
                worker_tmp = fetch(result)
                tmp_local[start_row:end_row, :] = worker_tmp
            end
        end
    end
    
    # Copy results back to shared tmp matrix
    tmp .= tmp_local
    
    # Ensure first multiplication is complete before proceeding
    @sync begin end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    @sync for w_idx in 1:num_workers
        # Determine row range for this worker
        start_row = (w_idx - 1) * rows_per_worker + 1
        end_row = min(w_idx * rows_per_worker, ni)
        
        if start_row <= end_row
            # Compute block on worker w
            w = workers()[w_idx]
            @async begin
                # Perform second matrix multiplication
                result = @spawnat w begin
                    local_D = zeros(end_row - start_row + 1, nl)
                    
                    # Initialize with beta * D
                    for i in 1:(end_row - start_row + 1)
                        for j in 1:nl
                            local_D[i,j] = beta * D[start_row + i - 1, j]
                        end
                    end
                    
                    # Add tmp * C
                    for i in 1:(end_row - start_row + 1)
                        for j in 1:nl
                            for k in 1:nj
                                local_D[i,j] += tmp[start_row + i - 1, k] * C[k,j]
                            end
                        end
                    end
                    
                    local_D
                end
                
                # Collect results from this worker
                worker_D = fetch(result)
                D_local[start_row:end_row, :] = worker_D
                
                # Record end time
                perf_data[w_idx]["end_time"] = time()
            end
        end
    end
    
    # Copy results back to shared D matrix
    D .= D_local
    
    # Calculate and return performance metrics
    total_time = maximum([data["end_time"] - data["start_time"] for (_, data) in perf_data])
    
    # Calculate GFLOPS
    total_flops = ni * nj * nk * 2 + ni * nl * nj * 2 + ni * nl
    gflops = total_time > 0 ? total_flops / (total_time * 1e9) : 0.0
    
    return D, Dict(
        "total_time" => total_time,
        "gflops" => gflops,
        "worker_stats" => perf_data
    )
end

# Hierarchical parallelism: Distributed between nodes, multithreaded within nodes
function kernel_2mm_hierarchical!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Calculate how many rows each worker should process
    num_workers = nworkers()
    rows_per_worker = ceil(Int, ni / num_workers)
    
    # Create a performance monitoring data structure
    perf_data = Dict(
        "worker_times" => Dict(),
        "memory_stats" => Dict(),
        "threads_per_worker" => Dict()
    )
    
    # Create local buffers to store results
    tmp_local = zeros(size(tmp))
    D_local = copy(D)
    
    # First phase: tmp = alpha*A*B using both workers and threads
    @sync for w_idx in 1:num_workers
        # Determine row range for this worker
        start_row = (w_idx - 1) * rows_per_worker + 1
        end_row = min(w_idx * rows_per_worker, ni)
        
        if start_row <= end_row
            # Compute block on worker w
            w = workers()[w_idx]
            @async begin
                # Start time for this worker
                start_time = time()
                
                # Get thread count on this worker
                threads = @spawnat w Threads.nthreads()
                perf_data["threads_per_worker"][w_idx] = fetch(threads)
                
                # Worker function that uses threads internally
                result = @spawnat w begin
                    local_tmp = zeros(end_row - start_row + 1, nj)
                    
                    # First matrix multiplication using threads
                    Threads.@threads for i in 1:(end_row - start_row + 1)
                        for j in 1:nj
                            local_sum = 0.0
                            for k in 1:nk
                                local_sum += A[start_row + i - 1, k] * B[k, j]
                            end
                            local_tmp[i, j] = alpha * local_sum
                        end
                    end
                    
                    local_tmp
                end
                
                # Collect results
                worker_tmp = fetch(result)
                tmp_local[start_row:end_row, :] = worker_tmp
                
                # Record performance data
                perf_data["worker_times"][w_idx] = time() - start_time
            end
        end
    end
    
    # Copy results back to shared tmp matrix
    tmp .= tmp_local
    
    # Ensure first multiplication is complete before starting second
    @sync begin end
    
    # Second phase: D = beta*D + tmp*C using both workers and threads
    @sync for w_idx in 1:num_workers
        # Determine row range for this worker
        start_row = (w_idx - 1) * rows_per_worker + 1
        end_row = min(w_idx * rows_per_worker, ni)
        
        if start_row <= end_row
            # Compute block on worker w
            w = workers()[w_idx]
            @async begin
                # Worker function that uses threads internally
                result = @spawnat w begin
                    local_D = zeros(end_row - start_row + 1, nl)
                    
                    # Second matrix multiplication using threads
                    Threads.@threads for i in 1:(end_row - start_row + 1)
                        for j in 1:nl
                            # Scale D by beta
                            local_D[i, j] = beta * D[start_row + i - 1, j]
                            
                            # Add tmp * C
                            for k in 1:nj
                                local_D[i, j] += tmp[start_row + i - 1, k] * C[k, j]
                            end
                        end
                    end
                    
                    local_D
                end
                
                # Collect results
                worker_D = fetch(result)
                D_local[start_row:end_row, :] = worker_D
            end
        end
    end
    
    # Copy results back to shared D matrix
    D .= D_local
    
    # Calculate overall performance metrics
    total_time = maximum(values(perf_data["worker_times"]))
    
    # Calculate GFLOPS
    total_flops = ni * nj * nk * 2 + ni * nl * nj * 2 + ni * nl
    gflops = total_time > 0 ? total_flops / (total_time * 1e9) : 0.0
    
    # Return the result matrix and performance data
    return D, Dict(
        "total_time" => total_time,
        "gflops" => gflops,
        "detailed_stats" => perf_data
    )
end

# BLAS-based implementation for comparison
function kernel_2mm_blas!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Record start time and memory usage
    start_time = time()
    mem_before = Sys.free_memory()
    
    # First matrix multiplication: tmp = alpha*A*B
    mul!(tmp, A, B, alpha, 0.0)
    
    # Second matrix multiplication: D = beta*D + tmp*C
    D .*= beta
    mul!(D, tmp, C, 1.0, 1.0)
    
    # Record end time and memory usage
    end_time = time()
    mem_after = Sys.free_memory()
    
    # Calculate performance metrics
    elapsed = end_time - start_time
    memory_used = mem_before - mem_after
    total_flops = ni * nj * nk * 2 + ni * nl * nj * 2 + ni * nl
    gflops = total_flops / (elapsed * 1e9)
    
    return D, Dict(
        "time" => elapsed,
        "gflops" => gflops,
        "memory_used" => memory_used
    )
end