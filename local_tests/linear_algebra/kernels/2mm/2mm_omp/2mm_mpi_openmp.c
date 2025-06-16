/**
 * 2mm_mpi_openmp.c: MPI + OpenMP hybrid implementation of PolyBench 2MM
 * 
 * This implementation provides distributed memory parallelism with MPI
 * combined with shared memory parallelism using OpenMP.
 * 
 * Strategies:
 * - Column-wise distribution (like Julia's distributed implementation)
 * - Row-wise distribution option
 * - Overlapping communication with computation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

// Include polybench headers
#include </mnt/c/Users/aldej/Desktop/Julia_versus_OpenMP/polybench_utilities/npolybench.h>
#include "2mm.h"

// MPI tags
#define TAG_A 1001
#define TAG_B 1002
#define TAG_C 1003
#define TAG_D 1004
#define TAG_TMP 1005

// Distribution strategies
#define DIST_COLUMNS 0
#define DIST_ROWS 1
#define DIST_2D 2

// Helper macros
#define BLOCK_SIZE(total, rank, nprocs) \
    ((total) / (nprocs) + ((rank) < ((total) % (nprocs)) ? 1 : 0))

#define BLOCK_START(total, rank, nprocs) \
    ((rank) * ((total) / (nprocs)) + MIN((rank), ((total) % (nprocs))))

#define BLOCK_END(total, rank, nprocs) \
    (BLOCK_START(total, rank, nprocs) + BLOCK_SIZE(total, rank, nprocs))

/* Initialize arrays in parallel */
static void init_arrays_mpi(int ni, int nj, int nk, int nl,
                           DATA_TYPE *alpha, DATA_TYPE *beta,
                           DATA_TYPE *A_local, DATA_TYPE *B_local,
                           DATA_TYPE *C_local, DATA_TYPE *D_local,
                           int my_rank, int nprocs, int distribution)
{
    *alpha = 1.5;
    *beta = 1.2;
    
    if (distribution == DIST_COLUMNS) {
        // Each process owns columns of matrices
        int my_nj_start = BLOCK_START(nj, my_rank, nprocs);
        int my_nj_size = BLOCK_SIZE(nj, my_rank, nprocs);
        int my_nl_start = BLOCK_START(nl, my_rank, nprocs);
        int my_nl_size = BLOCK_SIZE(nl, my_rank, nprocs);
        
        // Initialize B columns (local)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nk; i++) {
            for (int j = 0; j < my_nj_size; j++) {
                int global_j = my_nj_start + j;
                B_local[i * my_nj_size + j] = (DATA_TYPE)(i * (global_j + 1) % nj) / nj;
            }
        }
        
        // Initialize C columns (local)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nj; i++) {
            for (int j = 0; j < my_nl_size; j++) {
                int global_j = my_nl_start + j;
                C_local[i * my_nl_size + j] = (DATA_TYPE)((i * (global_j + 3) + 1) % nl) / nl;
            }
        }
        
        // Initialize D columns (local)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < my_nl_size; j++) {
                int global_j = my_nl_start + j;
                D_local[i * my_nl_size + j] = (DATA_TYPE)(i * (global_j + 2) % nk) / nk;
            }
        }
        
        // A is replicated on all processes
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nk; j++) {
                A_local[i * nk + j] = (DATA_TYPE)((i * j + 1) % ni) / ni;
            }
        }
    }
    else if (distribution == DIST_ROWS) {
        // Each process owns rows of matrices
        int my_ni_start = BLOCK_START(ni, my_rank, nprocs);
        int my_ni_size = BLOCK_SIZE(ni, my_rank, nprocs);
        
        // Initialize A rows (local)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < my_ni_size; i++) {
            for (int j = 0; j < nk; j++) {
                int global_i = my_ni_start + i;
                A_local[i * nk + j] = (DATA_TYPE)((global_i * j + 1) % ni) / ni;
            }
        }
        
        // Initialize D rows (local)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < my_ni_size; i++) {
            for (int j = 0; j < nl; j++) {
                int global_i = my_ni_start + i;
                D_local[i * nl + j] = (DATA_TYPE)(global_i * (j + 2) % nk) / nk;
            }
        }
        
        // B and C are replicated on all processes
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nk; i++) {
            for (int j = 0; j < nj; j++) {
                B_local[i * nj + j] = (DATA_TYPE)(i * (j + 1) % nj) / nj;
            }
        }
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nj; i++) {
            for (int j = 0; j < nl; j++) {
                C_local[i * nl + j] = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl;
            }
        }
    }
}

/* Column-distributed 2MM kernel (similar to Julia's approach) */
static void kernel_2mm_mpi_columns(int ni, int nj, int nk, int nl,
                                  DATA_TYPE alpha, DATA_TYPE beta,
                                  DATA_TYPE *A, DATA_TYPE *B_local,
                                  DATA_TYPE *C_local, DATA_TYPE *D_local,
                                  DATA_TYPE *tmp_local,
                                  int my_rank, int nprocs,
                                  MPI_Comm comm)
{
    // Get local column ranges
    int my_nj_start = BLOCK_START(nj, my_rank, nprocs);
    int my_nj_size = BLOCK_SIZE(nj, my_rank, nprocs);
    int my_nl_start = BLOCK_START(nl, my_rank, nprocs);
    int my_nl_size = BLOCK_SIZE(nl, my_rank, nprocs);
    
    // First multiplication: tmp_local = alpha * A * B_local
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < my_nj_size; j++) {
            DATA_TYPE sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i * nk + k] * B_local[k * my_nj_size + j];
            }
            tmp_local[i * my_nj_size + j] = sum;
        }
    }
    
    // Need to gather tmp for second multiplication
    // Each process needs specific columns of tmp for its columns of C
    
    // Alltoall communication to redistribute tmp
    int *send_counts = (int*)malloc(nprocs * sizeof(int));
    int *recv_counts = (int*)malloc(nprocs * sizeof(int));
    int *send_displs = (int*)malloc(nprocs * sizeof(int));
    int *recv_displs = (int*)malloc(nprocs * sizeof(int));
    
    // Calculate communication pattern
    for (int p = 0; p < nprocs; p++) {
        send_counts[p] = ni * my_nj_size;
        recv_counts[p] = ni * BLOCK_SIZE(nj, p, nprocs);
    }
    
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
        send_displs[p] = send_displs[p-1] + send_counts[p-1];
        recv_displs[p] = recv_displs[p-1] + recv_counts[p-1];
    }
    
    // Allocate receive buffer for complete tmp
    DATA_TYPE *tmp_full = (DATA_TYPE*)malloc(ni * nj * sizeof(DATA_TYPE));
    
    MPI_Allgatherv(tmp_local, ni * my_nj_size, MPI_DOUBLE,
                   tmp_full, recv_counts, recv_displs, MPI_DOUBLE, comm);
    
    // Second multiplication: D_local = beta * D_local + tmp_full * C_local
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < my_nl_size; j++) {
            D_local[i * my_nl_size + j] *= beta;
            DATA_TYPE sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp_full[i * nj + k] * C_local[k * my_nl_size + j];
            }
            D_local[i * my_nl_size + j] += sum;
        }
    }
    
    // Cleanup
    free(tmp_full);
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
}

/* Row-distributed 2MM kernel */
static void kernel_2mm_mpi_rows(int ni, int nj, int nk, int nl,
                               DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE *A_local, DATA_TYPE *B,
                               DATA_TYPE *C, DATA_TYPE *D_local,
                               DATA_TYPE *tmp_local,
                               int my_rank, int nprocs,
                               MPI_Comm comm)
{
    // Get local row range
    int my_ni_start = BLOCK_START(ni, my_rank, nprocs);
    int my_ni_size = BLOCK_SIZE(ni, my_rank, nprocs);
    
    // First multiplication: tmp_local = alpha * A_local * B
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < my_ni_size; i++) {
        for (int j = 0; j < nj; j++) {
            DATA_TYPE sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += alpha * A_local[i * nk + k] * B[k * nj + j];
            }
            tmp_local[i * nj + j] = sum;
        }
    }
    
    // Second multiplication: D_local = beta * D_local + tmp_local * C
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < my_ni_size; i++) {
        for (int j = 0; j < nl; j++) {
            D_local[i * nl + j] *= beta;
            DATA_TYPE sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp_local[i * nj + k] * C[k * nl + j];
            }
            D_local[i * nl + j] += sum;
        }
    }
}

/* MPI-only version (for comparison) */
static void kernel_2mm_mpi_only(int ni, int nj, int nk, int nl,
                               DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE *A_local, DATA_TYPE *B,
                               DATA_TYPE *C, DATA_TYPE *D_local,
                               DATA_TYPE *tmp_local,
                               int my_rank, int nprocs,
                               MPI_Comm comm)
{
    // Get local row range
    int my_ni_start = BLOCK_START(ni, my_rank, nprocs);
    int my_ni_size = BLOCK_SIZE(ni, my_rank, nprocs);
    
    // First multiplication: tmp_local = alpha * A_local * B
    for (int i = 0; i < my_ni_size; i++) {
        for (int j = 0; j < nj; j++) {
            DATA_TYPE sum = 0.0;
            for (int k = 0; k < nk; k++) {
                sum += alpha * A_local[i * nk + k] * B[k * nj + j];
            }
            tmp_local[i * nj + j] = sum;
        }
    }
    
    // Second multiplication: D_local = beta * D_local + tmp_local * C
    for (int i = 0; i < my_ni_size; i++) {
        for (int j = 0; j < nl; j++) {
            D_local[i * nl + j] *= beta;
            DATA_TYPE sum = 0.0;
            for (int k = 0; k < nj; k++) {
                sum += tmp_local[i * nj + k] * C[k * nl + j];
            }
            D_local[i * nl + j] += sum;
        }
    }
}

/* Print distributed array (gather to root) */
static void print_array_mpi(int ni, int nl, DATA_TYPE *D_local,
                           int my_rank, int nprocs, MPI_Comm comm)
{
    if (my_rank == 0) {
        DATA_TYPE *D_full = (DATA_TYPE*)malloc(ni * nl * sizeof(DATA_TYPE));
        
        // Gather all data to root
        int *recv_counts = (int*)malloc(nprocs * sizeof(int));
        int *recv_displs = (int*)malloc(nprocs * sizeof(int));
        
        for (int p = 0; p < nprocs; p++) {
            recv_counts[p] = BLOCK_SIZE(ni, p, nprocs) * nl;
            recv_displs[p] = BLOCK_START(ni, p, nprocs) * nl;
        }
        
        MPI_Gatherv(D_local, BLOCK_SIZE(ni, my_rank, nprocs) * nl, MPI_DOUBLE,
                    D_full, recv_counts, recv_displs, MPI_DOUBLE,
                    0, comm);
        
        // Print
        POLYBENCH_DUMP_START;
        POLYBENCH_DUMP_BEGIN("D");
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nl; j++) {
                if ((i * ni + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
                fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D_full[i * nl + j]);
            }
        }
        POLYBENCH_DUMP_END("D");
        POLYBENCH_DUMP_FINISH;
        
        free(D_full);
        free(recv_counts);
        free(recv_displs);
    }
    else {
        MPI_Gatherv(D_local, BLOCK_SIZE(ni, my_rank, nprocs) * nl, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
    }
}

int main(int argc, char** argv)
{
    int my_rank, nprocs;
    int provided;
    
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Problem sizes
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    
    // Select distribution strategy
    int distribution = DIST_ROWS;  // Default to row distribution
    if (argc > 1) {
        if (strcmp(argv[1], "columns") == 0) distribution = DIST_COLUMNS;
        else if (strcmp(argv[1], "2d") == 0) distribution = DIST_2D;
    }
    
    // Print configuration
    if (my_rank == 0) {
        printf("MPI+OpenMP 2MM Benchmark\n");
        printf("Processes: %d\n", nprocs);
        printf("Threads per process: %d\n", omp_get_max_threads());
        printf("Distribution: %s\n", 
               distribution == DIST_COLUMNS ? "columns" : 
               distribution == DIST_ROWS ? "rows" : "2D");
        printf("Matrix sizes: A(%d×%d), B(%d×%d), C(%d×%d), D(%d×%d)\n",
               ni, nk, nk, nj, nj, nl, ni, nl);
    }
    
    // Allocate memory based on distribution
    DATA_TYPE alpha, beta;
    DATA_TYPE *A_local, *B_local, *C_local, *D_local, *tmp_local;
    
    if (distribution == DIST_COLUMNS) {
        // Column distribution
        int my_nj_size = BLOCK_SIZE(nj, my_rank, nprocs);
        int my_nl_size = BLOCK_SIZE(nl, my_rank, nprocs);
        
        A_local = (DATA_TYPE*)malloc(ni * nk * sizeof(DATA_TYPE));  // Replicated
        B_local = (DATA_TYPE*)malloc(nk * my_nj_size * sizeof(DATA_TYPE));
        C_local = (DATA_TYPE*)malloc(nj * my_nl_size * sizeof(DATA_TYPE));
        D_local = (DATA_TYPE*)malloc(ni * my_nl_size * sizeof(DATA_TYPE));
        tmp_local = (DATA_TYPE*)malloc(ni * my_nj_size * sizeof(DATA_TYPE));
    }
    else {
        // Row distribution
        int my_ni_size = BLOCK_SIZE(ni, my_rank, nprocs);
        
        A_local = (DATA_TYPE*)malloc(my_ni_size * nk * sizeof(DATA_TYPE));
        B_local = (DATA_TYPE*)malloc(nk * nj * sizeof(DATA_TYPE));  // Replicated
        C_local = (DATA_TYPE*)malloc(nj * nl * sizeof(DATA_TYPE));  // Replicated
        D_local = (DATA_TYPE*)malloc(my_ni_size * nl * sizeof(DATA_TYPE));
        tmp_local = (DATA_TYPE*)malloc(my_ni_size * nj * sizeof(DATA_TYPE));
    }
    
    // Initialize arrays
    init_arrays_mpi(ni, nj, nk, nl, &alpha, &beta,
                    A_local, B_local, C_local, D_local,
                    my_rank, nprocs, distribution);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timer
    double start_time = MPI_Wtime();
    
    // Run kernel
    if (distribution == DIST_COLUMNS) {
        kernel_2mm_mpi_columns(ni, nj, nk, nl, alpha, beta,
                              A_local, B_local, C_local, D_local, tmp_local,
                              my_rank, nprocs, MPI_COMM_WORLD);
    }
    else {
        #ifdef MPI_ONLY
        kernel_2mm_mpi_only(ni, nj, nk, nl, alpha, beta,
                           A_local, B_local, C_local, D_local, tmp_local,
                           my_rank, nprocs, MPI_COMM_WORLD);
        #else
        kernel_2mm_mpi_rows(ni, nj, nk, nl, alpha, beta,
                           A_local, B_local, C_local, D_local, tmp_local,
                           my_rank, nprocs, MPI_COMM_WORLD);
        #endif
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Stop timer
    double end_time = MPI_Wtime();
    
    if (my_rank == 0) {
        printf("Execution time: %0.6f seconds\n", end_time - start_time);
        
        // Performance statistics
        double total_flops = 2.0 * ni * nj * nk + 2.0 * ni * nl * nj;
        double gflops = total_flops / (end_time - start_time) / 1e9;
        printf("Performance: %.2f GFLOPS\n", gflops);
        
        // Memory bandwidth estimate
        double data_moved = sizeof(DATA_TYPE) * (
            ni * nk +      // Read A
            nk * nj +      // Read B
            nj * nl +      // Read C
            ni * nl * 2 +  // Read/Write D
            ni * nj * 2    // Read/Write tmp
        );
        double bandwidth = data_moved / (end_time - start_time) / 1e9;
        printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    }
    
    // Prevent dead-code elimination
    if (getenv("POLYBENCH_DUMP_ARRAYS") != NULL) {
        print_array_mpi(ni, nl, D_local, my_rank, nprocs, MPI_COMM_WORLD);
    }
    
    // Cleanup
    free(A_local);
    free(B_local);
    free(C_local);
    free(D_local);
    free(tmp_local);
    
    MPI_Finalize();
    return 0;
}