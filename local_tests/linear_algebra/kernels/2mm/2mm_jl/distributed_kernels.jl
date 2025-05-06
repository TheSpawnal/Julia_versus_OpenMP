# This will be loaded on all workers with @everywhere
@everywhere function kernel_2mm_worker!(start_row, end_row, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication for a block of rows
    for i in start_row:end_row
        for j in 1:nj
            tmp[i,j] = 0.0
            for k in 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication for a block of rows
    for i in start_row:end_row
        for j in 1:nl
            D[i,j] *= beta
            for k in 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
end

# Coarse-grained distributed implementation
function kernel_2mm_dist_coarse!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # Calculate how many rows each worker should process
    num_workers = nworkers()
    rows_per_worker = ceil(Int, ni / num_workers)
    
    @sync for w_idx in 1:num_workers
        # Determine row range for this worker
        start_row = (w_idx - 1) * rows_per_worker + 1
        end_row = min(w_idx * rows_per_worker, ni)
        
        if start_row <= end_row
            # Compute block on worker w
            w = workers()[w_idx]
            @async begin
                @spawnat w kernel_2mm_worker!(
                    start_row, end_row, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
            end
        end
    end
    
    return D
end