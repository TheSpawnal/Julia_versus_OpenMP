/**
 * OpenMP implementation of 2mm kernel from PolyBench/C
 * Based on original implementation but with added OpenMP parallelization
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <string.h>
 #include <math.h>
 #include <omp.h>
 #include <windows.h>
 
 /* Include polybench common header. */
 /* Note: You might need to adjust the path */
 #include "C:\Users\aldej\Desktop\Julia_versus_OpenMP\polybench_utilities\polybench.h"
 
 /* Include benchmark-specific header. */
 #include "C:\Users\aldej\Desktop\Julia_versus_OpenMP\local_tests\linear_algebra\kernels\2mm\polybench_2mm.h"
 
 /* Array initialization. */
 static
 void init_array(int ni, int nj, int nk, int nl,
                 double *alpha,
                 double *beta,
                 double A[ni][nk],
                 double B[nk][nj],
                 double C[nj][nl],
                 double D[ni][nl])
 {
   int i, j;
 
   *alpha = 1.5;
   *beta = 1.2;
   for (i = 0; i < ni; i++)
     for (j = 0; j < nk; j++)
       A[i][j] = (double) ((i*j+1) % ni) / ni;
   for (i = 0; i < nk; i++)
     for (j = 0; j < nj; j++)
       B[i][j] = (double) (i*(j+1) % nj) / nj;
   for (i = 0; i < nj; i++)
     for (j = 0; j < nl; j++)
       C[i][j] = (double) ((i*(j+3)+1) % nl) / nl;
   for (i = 0; i < ni; i++)
     for (j = 0; j < nl; j++)
       D[i][j] = (double) (i*(j+2) % nk) / nk;
 }
 
 /* Sequential implementation (baseline) */
 static
 void kernel_2mm_sequential(int ni, int nj, int nk, int nl,
                           double alpha,
                           double beta,
                           double tmp[ni][nj],
                           double A[ni][nk],
                           double B[nk][nj],
                           double C[nj][nl],
                           double D[ni][nl])
 {
   int i, j, k;
 
   /* First matrix multiplication: tmp = alpha*A*B */
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nj; j++) {
       tmp[i][j] = 0.0;
       for (k = 0; k < nk; ++k)
         tmp[i][j] += alpha * A[i][k] * B[k][j];
     }
   }
   
   /* Second matrix multiplication: D = beta*D + tmp*C */
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nl; j++) {
       D[i][j] *= beta;
       for (k = 0; k < nj; ++k)
         D[i][j] += tmp[i][k] * C[k][j];
     }
   }
 }
 
 /* Fine-grained parallelism implementation */
 static
 void kernel_2mm_fine_grained(int ni, int nj, int nk, int nl,
                              double alpha,
                              double beta,
                              double tmp[ni][nj],
                              double A[ni][nk],
                              double B[nk][nj],
                              double C[nj][nl],
                              double D[ni][nl])
 {
   int i, j, k;
 
   /* First matrix multiplication: tmp = alpha*A*B */
   #pragma omp parallel for private(i, j, k)
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nj; j++) {
       tmp[i][j] = 0.0;
       for (k = 0; k < nk; ++k)
         tmp[i][j] += alpha * A[i][k] * B[k][j];
     }
   }
   
   /* Second matrix multiplication: D = beta*D + tmp*C */
   #pragma omp parallel for private(i, j, k)
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nl; j++) {
       D[i][j] *= beta;
       for (k = 0; k < nj; ++k)
         D[i][j] += tmp[i][k] * C[k][j];
     }
   }
 }
 
 /* Coarse-grained parallelism implementation */
 static
 void kernel_2mm_coarse_grained(int ni, int nj, int nk, int nl,
                                double alpha,
                                double beta,
                                double tmp[ni][nj],
                                double A[ni][nk],
                                double B[nk][nj],
                                double C[nj][nl],
                                double D[ni][nl])
 {
   int i, j, k;
   int num_threads = omp_get_max_threads();
   
   #pragma omp parallel num_threads(num_threads)
   {
     int thread_id = omp_get_thread_num();
     int chunk_size = (ni + num_threads - 1) / num_threads;
     int start_i = thread_id * chunk_size;
     int end_i = (thread_id + 1) * chunk_size;
     if (end_i > ni) end_i = ni;
     
     /* First matrix multiplication: tmp = alpha*A*B */
     for (i = start_i; i < end_i; i++) {
       for (j = 0; j < nj; j++) {
         tmp[i][j] = 0.0;
         for (k = 0; k < nk; ++k)
           tmp[i][j] += alpha * A[i][k] * B[k][j];
       }
     }
     
     /* Ensure all threads have finished the first computation */
     #pragma omp barrier
     
     /* Second matrix multiplication: D = beta*D + tmp*C */
     for (i = start_i; i < end_i; i++) {
       for (j = 0; j < nl; j++) {
         D[i][j] *= beta;
         for (k = 0; k < nj; ++k)
           D[i][j] += tmp[i][k] * C[k][j];
       }
     }
   }
 }
 
 /* Improved fine-grained parallelism with nested parallelism and scheduling */
 static
 void kernel_2mm_improved(int ni, int nj, int nk, int nl,
                          double alpha,
                          double beta,
                          double tmp[ni][nj],
                          double A[ni][nk],
                          double B[nk][nj],
                          double C[nj][nl],
                          double D[ni][nl])
 {
   int i, j, k;
 
   /* First matrix multiplication: tmp = alpha*A*B */
   #pragma omp parallel for private(i, j, k) schedule(dynamic, 16)
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nj; j++) {
       double sum = 0.0;
       for (k = 0; k < nk; ++k)
         sum += A[i][k] * B[k][j];
       tmp[i][j] = alpha * sum;
     }
   }
   
   /* Second matrix multiplication: D = beta*D + tmp*C */
   #pragma omp parallel for private(i, j, k) schedule(dynamic, 16)
   for (i = 0; i < ni; i++) {
     for (j = 0; j < nl; j++) {
       D[i][j] *= beta;
       for (k = 0; k < nj; ++k)
         D[i][j] += tmp[i][k] * C[k][j];
     }
   }
 }
 
 /* Utility function for timing */
 double get_time() {
   LARGE_INTEGER frequency;
   LARGE_INTEGER t;
   QueryPerformanceFrequency(&frequency);
   QueryPerformanceCounter(&t);
   return ((double)t.QuadPart / (double)frequency.QuadPart);
 }
 
 /* Benchmark function */
 void benchmark_2mm(int ni, int nj, int nk, int nl) {
   double alpha, beta;
   double start_time, end_time;
   
   /* Allocate matrices */
   double (*A)[nk] = (double(*)[nk])malloc(ni * nk * sizeof(double));
   double (*B)[nj] = (double(*)[nj])malloc(nk * nj * sizeof(double));
   double (*C)[nl] = (double(*)[nl])malloc(nj * nl * sizeof(double));
   double (*D)[nl] = (double(*)[nl])malloc(ni * nl * sizeof(double));
   double (*tmp)[nj] = (double(*)[nj])malloc(ni * nj * sizeof(double));
   
   /* For validation */
   double (*D_seq)[nl] = (double(*)[nl])malloc(ni * nl * sizeof(double));
   
   /* Initialize matrices */
   init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
   
   /* Copy D to D_seq for validation */
   memcpy(D_seq, D, ni * nl * sizeof(double));
   
   /* Warmup runs (not timed) */
   kernel_2mm_sequential(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D_seq);
   
   /* Number of iterations for averaging */
   int num_runs = 3;
   
   /* Sequential benchmark */
   double seq_time = 0.0;
   for (int run = 0; run < num_runs; run++) {
     init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D_seq);
     start_time = get_time();
     kernel_2mm_sequential(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D_seq);
     end_time = get_time();
     seq_time += (end_time - start_time);
   }
   seq_time /= num_runs;
   
   /* Fine-grained parallel benchmark */
   double fine_time = 0.0;
   for (int run = 0; run < num_runs; run++) {
     /* Reset matrices */
     init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
     
     start_time = get_time();
     kernel_2mm_fine_grained(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
     end_time = get_time();
     fine_time += (end_time - start_time);
     
     /* Validate results */
     for (int i = 0; i < ni; i++) {
       for (int j = 0; j < nl; j++) {
         if (fabs(D[i][j] - D_seq[i][j]) > 1e-10) {
           printf("Validation failed for fine-grained: D[%d][%d] = %f, D_seq[%d][%d] = %f\n",
                  i, j, D[i][j], i, j, D_seq[i][j]);
           break;
         }
       }
     }
   }
   fine_time /= num_runs;
   
   /* Coarse-grained parallel benchmark */
   double coarse_time = 0.0;
   for (int run = 0; run < num_runs; run++) {
     /* Reset matrices */
     init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
     
     start_time = get_time();
     kernel_2mm_coarse_grained(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
     end_time = get_time();
     coarse_time += (end_time - start_time);
     
     /* Validate results */
     for (int i = 0; i < ni; i++) {
       for (int j = 0; j < nl; j++) {
         if (fabs(D[i][j] - D_seq[i][j]) > 1e-10) {
           printf("Validation failed for coarse-grained: D[%d][%d] = %f, D_seq[%d][%d] = %f\n",
                  i, j, D[i][j], i, j, D_seq[i][j]);
           break;
         }
       }
     }
   }
   coarse_time /= num_runs;
   
   /* Improved parallel benchmark */
   double improved_time = 0.0;
   for (int run = 0; run < num_runs; run++) {
     /* Reset matrices */
     init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
     
     start_time = get_time();
     kernel_2mm_improved(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
     end_time = get_time();
     improved_time += (end_time - start_time);
     
     /* Validate results */
     for (int i = 0; i < ni; i++) {
       for (int j = 0; j < nl; j++) {
         if (fabs(D[i][j] - D_seq[i][j]) > 1e-10) {
           printf("Validation failed for improved: D[%d][%d] = %f, D_seq[%d][%d] = %f\n",
                  i, j, D[i][j], i, j, D_seq[i][j]);
           break;
         }
       }
     }
   }
   improved_time /= num_runs;
   
   /* Print results */
   printf("Matrix sizes: A(%d×%d), B(%d×%d), C(%d×%d), D(%d×%d)\n", 
          ni, nk, nk, nj, nj, nl, ni, nl);
   printf("Number of threads: %d\n\n", omp_get_max_threads());
   
   printf("Implementation | Time (s) | Speedup | Efficiency (%%)\n");
   printf("------------- | -------- | ------- | -------------\n");
   printf("Sequential    | %.6f | 1.00    | 100.00\n", seq_time);
   printf("Fine-grained  | %.6f | %.2f    | %.2f\n", 
          fine_time, seq_time/fine_time, 100*(seq_time/fine_time)/omp_get_max_threads());
   printf("Coarse-grained| %.6f | %.2f    | %.2f\n", 
          coarse_time, seq_time/coarse_time, 100*(seq_time/coarse_time)/omp_get_max_threads());
   printf("Improved      | %.6f | %.2f    | %.2f\n", 
          improved_time, seq_time/improved_time, 100*(seq_time/improved_time)/omp_get_max_threads());
   
   /* Free allocated memory */
   free(A);
   free(B);
   free(C);
   free(D);
   free(D_seq);
   free(tmp);
 }
 
 int main(int argc, char** argv) {
   /* Define dataset sizes from polybench_2mm.h */
   int datasets[3][4] = {
     {40, 50, 70, 80},      /* SMALL_DATASET */
     {180, 190, 210, 220},  /* MEDIUM_DATASET */
     {800, 900, 1100, 1200} /* LARGE_DATASET */
   };
   
   /* Set number of OpenMP threads */
   int num_threads = omp_get_num_procs(); /* Default to number of processors */
   if (argc > 1) {
     num_threads = atoi(argv[1]);
   }
   omp_set_num_threads(num_threads);
   
   /* Set dataset size */
   int dataset_idx = 0; /* Default to SMALL_DATASET */
   if (argc > 2) {
     dataset_idx = atoi(argv[2]);
     if (dataset_idx < 0 || dataset_idx > 2) {
       dataset_idx = 0;
     }
   }
   
   printf("Running 2mm benchmark with OpenMP\n");
   printf("=================================\n");
   
   int ni = datasets[dataset_idx][0];
   int nj = datasets[dataset_idx][1];
   int nk = datasets[dataset_idx][2];
   int nl = datasets[dataset_idx][3];
   
   benchmark_2mm(ni, nj, nk, nl);
   
   return 0;
 }