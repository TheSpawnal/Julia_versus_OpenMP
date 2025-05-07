# Enhanced sequential implementations with better cache utilization
function kernel_2mm_sequential_blocked!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Use blocking to improve cache locality
    block_size = 32  # Adjust based on target architecture
    
    # First matrix multiplication: tmp = alpha*A*B
    for ib in 1:block_size:ni
        i_end = min(ib + block_size - 1, ni)
        for jb in 1:block_size:nj
            j_end = min(jb + block_size - 1, nj)
            
            # Initialize tmp block
            for i in ib:i_end
                for j in jb:j_end
                    tmp[i,j] = 0.0
                end
            end
            
            # Compute tmp block
            for kb in 1:block_size:nk
                k_end = min(kb + block_size - 1, nk)
                for i in ib:i_end
                    for k in kb:k_end
                        for j in jb:j_end
                            @inbounds tmp[i,j] += alpha * A[i,k] * B[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    for ib in 1:block_size:ni
        i_end = min(ib + block_size - 1, ni)
        for jb in 1:block_size:nl
            j_end = min(jb + block_size - 1, nl)
            
            # Scale D block by beta
            for i in ib:i_end
                for j in jb:j_end
                    @inbounds D[i,j] *= beta
                end
            end
            
            # Compute D block
            for kb in 1:block_size:nj
                k_end = min(kb + block_size - 1, nj)
                for i in ib:i_end
                    for k in kb:k_end
                        for j in jb:j_end
                            @inbounds D[i,j] += tmp[i,k] * C[k,j]
                        end
                    end
                end
            end
        end
    end
    
    return D
end

# Improved task-based implementation with block-based parallelism
function kernel_2mm_block_tasks!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Use larger blocks for tasks to reduce task creation overhead
    block_size = 64
    
    # First matrix multiplication: tmp = alpha*A*B
    @sync begin
        for ib in 1:block_size:ni
            i_end = min(ib + block_size - 1, ni)
            @async begin
                for jb in 1:block_size:nj
                    j_end = min(jb + block_size - 1, nj)
                    
                    # Initialize tmp block
                    for i in ib:i_end
                        for j in jb:j_end
                            tmp[i,j] = 0.0
                        end
                    end
                    
                    # Compute tmp block
                    for kb in 1:block_size:nk
                        k_end = min(kb + block_size - 1, nk)
                        for i in ib:i_end
                            for k in kb:k_end
                                for j in jb:j_end
                                    @inbounds tmp[i,j] += alpha * A[i,k] * B[k,j]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    @sync begin
        for ib in 1:block_size:ni
            i_end = min(ib + block_size - 1, ni)
            @async begin
                for jb in 1:block_size:nl
                    j_end = min(jb + block_size - 1, nl)
                    
                    # Scale D block by beta
                    for i in ib:i_end
                        for j in jb:j_end
                            @inbounds D[i,j] *= beta
                        end
                    end
                    
                    # Compute D block
                    for kb in 1:block_size:nj
                        k_end = min(kb + block_size - 1, nj)
                        for i in ib:i_end
                            for k in kb:k_end
                                for j in jb:j_end
                                    @inbounds D[i,j] += tmp[i,k] * C[k,j]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return D
end

# Multithreaded implementation using Julia's native threading
function kernel_2mm_threaded!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Use blocking to improve cache locality
    block_size = 32
    
    # First matrix multiplication: tmp = alpha*A*B
    Threads.@threads for ib in 1:block_size:ni
        i_end = min(ib + block_size - 1, ni)
        for jb in 1:block_size:nj
            j_end = min(jb + block_size - 1, nj)
            
            # Initialize tmp block
            for i in ib:i_end
                for j in jb:j_end
                    tmp[i,j] = 0.0
                end
            end
            
            # Compute tmp block
            for kb in 1:block_size:nk
                k_end = min(kb + block_size - 1, nk)
                for i in ib:i_end
                    for k in kb:k_end
                        for j in jb:j_end
                            @inbounds tmp[i,j] += alpha * A[i,k] * B[k,j]
                        end
                    end
                end
            end
        end
    end
    
    # Ensure first multiplication is complete before starting second
    @sync begin
        # Just an empty sync block to create a barrier
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    Threads.@threads for ib in 1:block_size:ni
        i_end = min(ib + block_size - 1, ni)
        for jb in 1:block_size:nl
            j_end = min(jb + block_size - 1, nl)
            
            # Scale D block by beta
            for i in ib:i_end
                for j in jb:j_end
                    @inbounds D[i,j] *= beta
                end
            end
            
            # Compute D block
            for kb in 1:block_size:nj
                k_end = min(kb + block_size - 1, nj)
                for i in ib:i_end
                    for k in kb:k_end
                        for j in jb:j_end
                            @inbounds D[i,j] += tmp[i,k] * C[k,j]
                        end
                    end
                end
            end
        end
    end
    
    return D
end

# SIMD-optimized implementation
function kernel_2mm_simd!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Ensure matrices are properly aligned for SIMD operations
    # In a real implementation, you would use specialized array types
    
    # First matrix multiplication: tmp = alpha*A*B
    for i in 1:ni
        for j in 1:nj
            tmp_ij = 0.0
            # Use SIMD acceleration for the inner loop
            @simd for k in 1:nk
                @inbounds tmp_ij += A[i,k] * B[k,j]
            end
            tmp[i,j] = alpha * tmp_ij
        end
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    for i in 1:ni
        for j in 1:nl
            @inbounds D[i,j] *= beta
            d_ij = 0.0
            # Use SIMD acceleration for the inner loop
            @simd for k in 1:nj
                @inbounds d_ij += tmp[i,k] * C[k,j]
            end
            D[i,j] += d_ij
        end
    end
    
    return D
end

# Function to track performance metrics during execution (simplified)
function with_metrics(func, ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Record start time
    start_time = time()
    
    # Initial memory usage
    mem_before = Sys.free_memory()
    
    # Execute the kernel function
    result = func(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    
    # Record end time
    end_time = time()
    
    # Final memory usage
    mem_after = Sys.free_memory()
    
    # Calculate performance metrics
    elapsed = end_time - start_time
    memory_used = mem_before - mem_after
    
    # Calculate FLOPS
    # Each multiplication-addition is 2 FLOPS
    # First multiplication: ni * nj * nk * 2 operations
    # Second multiplication: ni * nl * nj * 2 operations
    # Plus ni*nl multiplications by beta
    total_flops = ni * nj * nk * 2 + ni * nl * nj * 2 + ni * nl
    gflops = total_flops / (elapsed * 1e9)
    
    return result, Dict(
        "time" => elapsed,
        "memory_used" => memory_used,
        "gflops" => gflops
    )
end