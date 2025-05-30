
# 7. MPI-style Implementation using distributed approach
# NOTE: Commented out as it requires the MPI.jl package to be installed

# function kernel_2mm_mpi(alpha, beta, tmp, A, B, C, D)

#     comm = MPI.COMM_WORLD
#     rank = MPI.Comm_rank(comm)
#     nprocs = MPI.Comm_size(comm)
    
#     if rank == 0
#         ni, nk = size(A)
#         nj, nl = size(C)
#         # Broadcast dimensions
#         dims = [ni, nj, nk, nl]
#     else
#         dims = Array{Int}(undef, 4)
#     end
    
#     MPI.Bcast!(dims, 0, comm)
#     ni, nj, nk, nl = dims
    
#     # Distribute B and C to all processes
#     if rank != 0
#         B = Array{eltype(A)}(undef, nk, nj)
#         C = Array{eltype(A)}(undef, nj, nl)
#     end
#     MPI.Bcast!(B, 0, comm)
#     MPI.Bcast!(C, 0, comm)
    
#     # Determine local work for each process
#     rows_per_proc = div(ni, nprocs)
#     start_row = rank * rows_per_proc + 1
#     end_row = (rank == nprocs-1) ? ni : (rank+1) * rows_per_proc
#     local_rows = end_row - start_row + 1
    
#     # Local portions of matrices
#     A_local = (rank == 0) ? A[start_row:end_row, :] : Array{eltype(A)}(undef, local_rows, nk)
#     tmp_local = Array{eltype(A)}(undef, local_rows, nj)
#     D_local = (rank == 0) ? D[start_row:end_row, :] : Array{eltype(A)}(undef, local_rows, nl)
    
#     # Scatter A
#     if rank == 0
#         for p = 1:nprocs-1
#             p_start = p * rows_per_proc + 1
#             p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
#             p_rows = p_end - p_start + 1
#             MPI.Send(A[p_start:p_end, :], p, 0, comm)
#         end
#     else
#         MPI.Recv!(A_local, 0, 0, comm)
#     end
    
#     # First matrix multiplication: tmp_local = alpha * A_local * B
#     for i = 1:local_rows
#         for j = 1:nj
#             tmp_local[i,j] = zero(eltype(tmp))
#             for k = 1:nk
#                 tmp_local[i,j] += alpha * A_local[i,k] * B[k,j]
#             end
#         end
#     end
    
#     # Scatter D if not process 0
#     if rank == 0
#         for p = 1:nprocs-1
#             p_start = p * rows_per_proc + 1
#             p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
#             p_rows = p_end - p_start + 1
#             MPI.Send(D[p_start:p_end, :], p, 1, comm)
#         end
#     else
#         MPI.Recv!(D_local, 0, 1, comm)
#     end
    
#     # Second matrix multiplication: D_local = tmp_local * C + beta * D_local
#     for i = 1:local_rows
#         for j = 1:nl
#             D_local[i,j] *= beta
#             for k = 1:nj
#                 D_local[i,j] += tmp_local[i,k] * C[k,j]
#             end
#         end
#     end
    
#     # Gather results back to process 0
#     if rank == 0
#         for p = 1:nprocs-1
#             p_start = p * rows_per_proc + 1
#             p_end = (p == nprocs-1) ? ni : (p+1) * rows_per_proc
#             p_rows = p_end - p_start + 1
#             D_p = Array{eltype(D)}(undef, p_rows, nl)
#             MPI.Recv!(D_p, p, 2, comm)
#             D[p_start:p_end, :] = D_p
#         end
#     else
#         MPI.Send(D_local, 0, 2, comm)
#     end
    
#     return (rank == 0) ? D : nothing
# end