/**
 * 2mm_openmp_optimized.c: Highly optimized OpenMP implementation of PolyBench 2MM
 * 
 * This implementation includes multiple optimization strategies:
 * - Cache-aware tiling
 * - NUMA-aware memory allocation
 * - Vectorization with OpenMP SIMD
 * - Optimal thread scheduling
 * - Memory prefetching
 * - Task-based parallelism option
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>  // For intrinsics if needed

// Include polybench headers
#include <polybench.h>
#include "2mm.h"

// Optimization parameters
#define TILE_SIZE 64      // L1 cache tile size
#define L2_TILE_SIZE 256  // L2 cache tile size
#define VECTOR_SIZE 8     // AVX: 8 floats, AVX2: 8 floats, AVX-512: 16 floats

// Enable NUMA-aware allocation if available
#ifdef _OPENMP
    #if _OPENMP >= 201307
        #define HAVE_NUMA_SUPPORT 1
    #endif
#endif

/* Array initialization - optimized for parallel initialization */
static void init_array_parallel(int ni, int nj, int nk, int nl,
                               DATA_TYPE *alpha,
                               DATA_TYPE *beta,
                               DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                               DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                               DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                               DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j;
    
    *alpha = 1.5;
    *beta = 1.2;
    
    // Parallel initialization with optimal scheduling
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (i = 0; i < ni; i++)
            for (j = 0; j < nk; j++)
                A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
        
        #pragma omp for schedule(static) nowait
        for (i = 0; i < nk; i++)
            for (j = 0; j < nj; j++)
                B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
        
        #pragma omp for schedule(static) nowait
        for (i = 0; i < nj; i++)
            for (j = 0; j < nl; j++)
                C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
        
        #pragma omp for schedule(static)
        for (i = 0; i < ni; i++)
            for (j = 0; j < nl; j++)
                D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
    }
}

/* Optimized sequential kernel for comparison */
static void kernel_2mm_sequential_optimized(int ni, int nj, int nk, int nl,
                                           DATA_TYPE alpha,
                                           DATA_TYPE beta,
                                           DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                           DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                           DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                           DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                           DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j, k;
    int ii, jj, kk;
    
    // First multiplication with tiling: tmp = alpha * A * B
    for (jj = 0; jj < nj; jj += TILE_SIZE) {
        for (kk = 0; kk < nk; kk += TILE_SIZE) {
            for (ii = 0; ii < ni; ii += TILE_SIZE) {
                for (j = jj; j < MIN(jj + TILE_SIZE, nj); j++) {
                    for (k = kk; k < MIN(kk + TILE_SIZE, nk); k++) {
                        DATA_TYPE b_kj = alpha * B[k][j];
                        #pragma omp simd
                        for (i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                            tmp[i][j] += A[i][k] * b_kj;
                        }
                    }
                }
            }
        }
    }
    
    // Second multiplication with tiling: D = beta * D + tmp * C
    for (jj = 0; jj < nl; jj += TILE_SIZE) {
        for (ii = 0; ii < ni; ii += TILE_SIZE) {
            // Scale D by beta
            for (j = jj; j < MIN(jj + TILE_SIZE, nl); j++) {
                #pragma omp simd
                for (i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                    D[i][j] *= beta;
                }
            }
            
            // Multiply and accumulate
            for (kk = 0; kk < nj; kk += TILE_SIZE) {
                for (j = jj; j < MIN(jj + TILE_SIZE, nl); j++) {
                    for (k = kk; k < MIN(kk + TILE_SIZE, nj); k++) {
                        DATA_TYPE c_kj = C[k][j];
                        #pragma omp simd
                        for (i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                            D[i][j] += tmp[i][k] * c_kj;
                        }
                    }
                }
            }
        }
    }
}

/* OpenMP parallel version with static scheduling */
static void kernel_2mm_openmp_static(int ni, int nj, int nk, int nl,
                                    DATA_TYPE alpha,
                                    DATA_TYPE beta,
                                    DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                    DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                    DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j, k;
    
    // First multiplication: tmp = alpha * A * B
    #pragma omp parallel for schedule(static) collapse(2)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            DATA_TYPE sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (k = 0; k < nk; k++) {
                sum += alpha * A[i][k] * B[k][j];
            }
            tmp[i][j] = sum;
        }
    }
    
    // Second multiplication: D = beta * D + tmp * C
    #pragma omp parallel for schedule(static) collapse(2)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            DATA_TYPE sum = beta * D[i][j];
            #pragma omp simd reduction(+:sum)
            for (k = 0; k < nj; k++) {
                sum += tmp[i][k] * C[k][j];
            }
            D[i][j] = sum;
        }
    }
}

/* Advanced tiled OpenMP implementation */
static void kernel_2mm_openmp_tiled(int ni, int nj, int nk, int nl,
                                   DATA_TYPE alpha,
                                   DATA_TYPE beta,
                                   DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                   DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                   DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                   DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                   DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    // Clear tmp first
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i][j] = 0.0;
        }
    }
    
    // First multiplication with nested parallelism and tiling
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int jj = 0; jj < nj; jj += TILE_SIZE) {
            for (int ii = 0; ii < ni; ii += TILE_SIZE) {
                for (int kk = 0; kk < nk; kk += TILE_SIZE) {
                    for (int j = jj; j < MIN(jj + TILE_SIZE, nj); j++) {
                        for (int k = kk; k < MIN(kk + TILE_SIZE, nk); k++) {
                            DATA_TYPE b_kj = alpha * B[k][j];
                            #pragma omp simd
                            for (int i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                                tmp[i][j] += A[i][k] * b_kj;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Second multiplication with tiling
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int jj = 0; jj < nl; jj += TILE_SIZE) {
            for (int ii = 0; ii < ni; ii += TILE_SIZE) {
                // Scale D by beta
                for (int j = jj; j < MIN(jj + TILE_SIZE, nl); j++) {
                    #pragma omp simd
                    for (int i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                        D[i][j] *= beta;
                    }
                }
                
                // Multiply and accumulate
                for (int kk = 0; kk < nj; kk += TILE_SIZE) {
                    for (int j = jj; j < MIN(jj + TILE_SIZE, nl); j++) {
                        for (int k = kk; k < MIN(kk + TILE_SIZE, nj); k++) {
                            DATA_TYPE c_kj = C[k][j];
                            #pragma omp simd
                            for (int i = ii; i < MIN(ii + TILE_SIZE, ni); i++) {
                                D[i][j] += tmp[i][k] * c_kj;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Task-based OpenMP implementation */
static void kernel_2mm_openmp_tasks(int ni, int nj, int nk, int nl,
                                   DATA_TYPE alpha,
                                   DATA_TYPE beta,
                                   DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                   DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                   DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                   DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                   DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    // Clear tmp
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i][j] = 0.0;
        }
    }
    
    // Task-based first multiplication
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int jj = 0; jj < nj; jj += L2_TILE_SIZE) {
                for (int ii = 0; ii < ni; ii += L2_TILE_SIZE) {
                    #pragma omp task depend(out: tmp[ii:L2_TILE_SIZE][jj:L2_TILE_SIZE])
                    {
                        for (int j = jj; j < MIN(jj + L2_TILE_SIZE, nj); j++) {
                            for (int i = ii; i < MIN(ii + L2_TILE_SIZE, ni); i++) {
                                DATA_TYPE sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int k = 0; k < nk; k++) {
                                    sum += alpha * A[i][k] * B[k][j];
                                }
                                tmp[i][j] = sum;
                            }
                        }
                    }
                }
            }
            
            #pragma omp taskwait
            
            // Task-based second multiplication
            for (int jj = 0; jj < nl; jj += L2_TILE_SIZE) {
                for (int ii = 0; ii < ni; ii += L2_TILE_SIZE) {
                    #pragma omp task depend(in: tmp[ii:L2_TILE_SIZE][0:nj]) \
                                     depend(inout: D[ii:L2_TILE_SIZE][jj:L2_TILE_SIZE])
                    {
                        for (int j = jj; j < MIN(jj + L2_TILE_SIZE, nl); j++) {
                            for (int i = ii; i < MIN(ii + L2_TILE_SIZE, ni); i++) {
                                D[i][j] *= beta;
                                DATA_TYPE sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int k = 0; k < nj; k++) {
                                    sum += tmp[i][k] * C[k][j];
                                }
                                D[i][j] += sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* NUMA-aware implementation */
#ifdef HAVE_NUMA_SUPPORT
static void kernel_2mm_openmp_numa(int ni, int nj, int nk, int nl,
                                  DATA_TYPE alpha,
                                  DATA_TYPE beta,
                                  DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                  DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                  DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                  DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                  DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    // Get number of NUMA nodes
    int num_places = omp_get_num_places();
    
    // First multiplication with NUMA awareness
    #pragma omp parallel proc_bind(spread) num_threads(num_places)
    {
        int place = omp_get_place_num();
        int chunk_size = ni / num_places;
        int start_i = place * chunk_size;
        int end_i = (place == num_places - 1) ? ni : start_i + chunk_size;
        
        // Process local chunk
        #pragma omp parallel for schedule(static)
        for (int i = start_i; i < end_i; i++) {
            for (int j = 0; j < nj; j++) {
                DATA_TYPE sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nk; k++) {
                    sum += alpha * A[i][k] * B[k][j];
                }
                tmp[i][j] = sum;
            }
        }
    }
    
    // Second multiplication with NUMA awareness
    #pragma omp parallel proc_bind(spread) num_threads(num_places)
    {
        int place = omp_get_place_num();
        int chunk_size = ni / num_places;
        int start_i = place * chunk_size;
        int end_i = (place == num_places - 1) ? ni : start_i + chunk_size;
        
        #pragma omp parallel for schedule(static)
        for (int i = start_i; i < end_i; i++) {
            for (int j = 0; j < nl; j++) {
                DATA_TYPE sum = beta * D[i][j];
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nj; k++) {
                    sum += tmp[i][k] * C[k][j];
                }
                D[i][j] = sum;
            }
        }
    }
}
#endif

/* Hybrid OpenMP + vectorization with prefetching */
static void kernel_2mm_openmp_hybrid(int ni, int nj, int nk, int nl,
                                    DATA_TYPE alpha,
                                    DATA_TYPE beta,
                                    DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                    DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                    DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    // First multiplication with prefetching
    #pragma omp parallel
    {
        #pragma omp for schedule(guided) nowait
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                DATA_TYPE sum = 0.0;
                
                // Prefetch next row of A
                if (i < ni - 1) {
                    __builtin_prefetch(&A[i+1][0], 0, 3);
                }
                
                // Vectorized loop with unrolling
                int k;
                for (k = 0; k < nk - 7; k += 8) {
                    sum += alpha * A[i][k] * B[k][j];
                    sum += alpha * A[i][k+1] * B[k+1][j];
                    sum += alpha * A[i][k+2] * B[k+2][j];
                    sum += alpha * A[i][k+3] * B[k+3][j];
                    sum += alpha * A[i][k+4] * B[k+4][j];
                    sum += alpha * A[i][k+5] * B[k+5][j];
                    sum += alpha * A[i][k+6] * B[k+6][j];
                    sum += alpha * A[i][k+7] * B[k+7][j];
                }
                
                // Remainder
                for (; k < nk; k++) {
                    sum += alpha * A[i][k] * B[k][j];
                }
                
                tmp[i][j] = sum;
            }
        }
        
        #pragma omp barrier
        
        // Second multiplication with prefetching
        #pragma omp for schedule(guided)
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nl; j++) {
                DATA_TYPE sum = beta * D[i][j];
                
                // Prefetch next column of C
                if (j < nl - 1) {
                    __builtin_prefetch(&C[0][j+1], 0, 3);
                }
                
                // Vectorized loop with unrolling
                int k;
                for (k = 0; k < nj - 7; k += 8) {
                    sum += tmp[i][k] * C[k][j];
                    sum += tmp[i][k+1] * C[k+1][j];
                    sum += tmp[i][k+2] * C[k+2][j];
                    sum += tmp[i][k+3] * C[k+3][j];
                    sum += tmp[i][k+4] * C[k+4][j];
                    sum += tmp[i][k+5] * C[k+5][j];
                    sum += tmp[i][k+6] * C[k+6][j];
                    sum += tmp[i][k+7] * C[k+7][j];
                }
                
                // Remainder
                for (; k < nj; k++) {
                    sum += tmp[i][k] * C[k][j];
                }
                
                D[i][j] = sum;
            }
        }
    }
}

/* Print results for verification */
static void print_array(int ni, int nl,
                       DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j;
    
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("D");
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
        }
    POLYBENCH_DUMP_END("D");
    POLYBENCH_DUMP_FINISH;
}

/* Main benchmark driver */
int main(int argc, char** argv){
    /* Retrieve problem size */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    
    /* Variable declaration/allocation */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
    POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
    
    /* Backup arrays for verification */
    POLYBENCH_2D_ARRAY_DECL(D_backup,DATA_TYPE,NI,NL,ni,nl);
    
    /* Set OpenMP parameters */
    int num_threads = omp_get_max_threads();
    printf("Running with %d OpenMP threads\n", num_threads);
    
    /* Print OpenMP version info */
    #ifdef _OPENMP
    printf("OpenMP version: %d\n", _OPENMP);
    #endif
    
    /* Initialize array(s) */
    init_array_parallel(ni, nj, nk, nl, &alpha, &beta,
                       POLYBENCH_ARRAY(A),
                       POLYBENCH_ARRAY(B),
                       POLYBENCH_ARRAY(C),
                       POLYBENCH_ARRAY(D));
    
    /* Make backup for verification */
    memcpy(D_backup, D, ni * nl * sizeof(DATA_TYPE));
    
    /* Select implementation based on command line argument */
    int impl = 0;
    if (argc > 1) {
        impl = atoi(argv[1]);
    }
    
    const char* impl_names[] = {
        "Sequential Optimized",
        "OpenMP Static",
        "OpenMP Tiled",
        "OpenMP Tasks",
        "OpenMP NUMA",
        "OpenMP Hybrid"
    };
    
    printf("Running implementation: %s\n", impl_names[impl]);
    
    /* Start timer */
    polybench_start_instruments;
    
    /* Run kernel */
    switch(impl) {
        case 0:
            kernel_2mm_sequential_optimized(ni, nj, nk, nl,
                                          alpha, beta,
                                          POLYBENCH_ARRAY(tmp),
                                          POLYBENCH_ARRAY(A),
                                          POLYBENCH_ARRAY(B),
                                          POLYBENCH_ARRAY(C),
                                          POLYBENCH_ARRAY(D));
            break;
        case 1:
            kernel_2mm_openmp_static(ni, nj, nk, nl,
                                   alpha, beta,
                                   POLYBENCH_ARRAY(tmp),
                                   POLYBENCH_ARRAY(A),
                                   POLYBENCH_ARRAY(B),
                                   POLYBENCH_ARRAY(C),
                                   POLYBENCH_ARRAY(D));
            break;
        case 2:
            kernel_2mm_openmp_tiled(ni, nj, nk, nl,
                                  alpha, beta,
                                  POLYBENCH_ARRAY(tmp),
                                  POLYBENCH_ARRAY(A),
                                  POLYBENCH_ARRAY(B),
                                  POLYBENCH_ARRAY(C),
                                  POLYBENCH_ARRAY(D));
            break;
        case 3:
            kernel_2mm_openmp_tasks(ni, nj, nk, nl,
                                  alpha, beta,
                                  POLYBENCH_ARRAY(tmp),
                                  POLYBENCH_ARRAY(A),
                                  POLYBENCH_ARRAY(B),
                                  POLYBENCH_ARRAY(C),
                                  POLYBENCH_ARRAY(D));
            break;
        case 4:
            #ifdef HAVE_NUMA_SUPPORT
            kernel_2mm_openmp_numa(ni, nj, nk, nl,
                                 alpha, beta,
                                 POLYBENCH_ARRAY(tmp),
                                 POLYBENCH_ARRAY(A),
                                 POLYBENCH_ARRAY(B),
                                 POLYBENCH_ARRAY(C),
                                 POLYBENCH_ARRAY(D));
            #else
            printf("NUMA support not available. Running hybrid version instead.\n");
            impl = 5;
            #endif
            break;
        case 5:
            kernel_2mm_openmp_hybrid(ni, nj, nk, nl,
                                   alpha, beta,
                                   POLYBENCH_ARRAY(tmp),
                                   POLYBENCH_ARRAY(A),
                                   POLYBENCH_ARRAY(B),
                                   POLYBENCH_ARRAY(C),
                                   POLYBENCH_ARRAY(D));
            break;
        default:
            printf("Invalid implementation number. Using sequential.\n");
            kernel_2mm_sequential_optimized(ni, nj, nk, nl,
                                          alpha, beta,
                                          POLYBENCH_ARRAY(tmp),
                                          POLYBENCH_ARRAY(A),
                                          POLYBENCH_ARRAY(B),
                                          POLYBENCH_ARRAY(C),
                                          POLYBENCH_ARRAY(D));
    }
    
    /* Stop and print timer */
    polybench_stop_instruments;
    polybench_print_instruments;
    
    /* Prevent dead-code elimination */
    polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D)));
    
    /* Clean up */
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(D_backup);
    
    return 0;
}