# Sequential implementation
function kernel_2mm_sequential!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha*A*B
    for i in 1:ni
        for j in 1:nj
            tmp[i,j] = 0.0
            for k in 1:nk
                tmp[i,j] += alpha * A[i,k] * B[k,j]
            end
        end
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    for i in 1:ni
        for j in 1:nl
            D[i,j] *= beta
            for k in 1:nj
                D[i,j] += tmp[i,k] * C[k,j]
            end
        end
    end
    
    return D
end

# Task-based implementation
function kernel_2mm_tasks!(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D)
    # First matrix multiplication: tmp = alpha*A*B
    @sync begin
        for i in 1:ni
            @async begin
                for j in 1:nj
                    tmp[i,j] = 0.0
                    for k in 1:nk
                        tmp[i,j] += alpha * A[i,k] * B[k,j]
                    end
                end
            end
        end
    end
    
    # Second matrix multiplication: D = beta*D + tmp*C
    @sync begin
        for i in 1:ni
            @async begin
                for j in 1:nl
                    D[i,j] *= beta
                    for k in 1:nj
                        D[i,j] += tmp[i,k] * C[k,j]
                    end
                end
            end
        end
    end
    
    return D
end

# Distributed implementations will be defined here but need worker setup first