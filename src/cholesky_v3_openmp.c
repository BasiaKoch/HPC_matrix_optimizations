#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP header: required for the parallel/for/simd directives used below. */

#define MAX_N 100000

/*
 * v3_openmp: parallelise the trailing-submatrix update with OpenMP.
 *
 * Parallelisation strategy
 * ------------------------
 * The Cholesky p-loop has a strict serial dependency: step p+1 reads values
 * written in step p.  Within each step p, the row updates
 *   c[i*n + j] -= c[i*n + p] * c[p*n + j]   for i, j > p
 * are fully independent across rows i → perfect data parallelism.
 *
 * Thread-pool placement: the omp parallel region is opened ONCE before the
 * p-loop and kept alive for the whole factorisation.  Opening it inside the
 * p-loop would pay thread-creation overhead O(n) times, which dominates for
 * small n.  Instead, omp single serialises the diagonal/normalisation work
 * and omp for distributes the row updates across threads each step.
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open the parallel region once outside the p-loop.
     * All threads in the team are created here and reused for every step p.
     * The number of threads is controlled by OMP_NUM_THREADS at runtime. */
    #pragma omp parallel default(none) shared(c, n)
    {
        for (int p = 0; p < n; p++) {
            double *row_p = &c[p*n];

            /* omp single: exactly one thread executes the diagonal update and
             * row/column normalisation.  These writes must be visible to all
             * threads before the trailing update begins.
             * The implicit barrier at the end of the single region ensures
             * all threads wait here before proceeding to the omp for. */
            #pragma omp single
            {
                double diag = sqrt(row_p[p]);
                row_p[p] = diag;

                double inv_diag = 1.0 / diag;

                for (int j = p + 1; j < n; j++)
                    row_p[j] *= inv_diag;

                for (int i = p + 1; i < n; i++)
                    c[i*n + p] *= inv_diag;
            }
            /* --- implicit barrier: all threads synchronised here --- */

            /* omp for: distribute the (n-p-1) independent row updates evenly
             * across threads using static scheduling.
             * schedule(static): each thread gets a contiguous chunk of rows
             * of size roughly (n-p-1)/nthreads, assigned at compile time.
             * This is appropriate because every row update does the same
             * amount of work (n-p-1 FMAs), so load is balanced.
             * The implicit barrier at the end of the for ensures all threads
             * finish step p before any thread begins the omp single of step p+1. */
            #pragma omp for schedule(static)
            for (int i = p + 1; i < n; i++) {
                double *row_i = &c[i*n];
                double c_ip = row_i[p];   /* load once; invariant across j */

                /* omp simd: hint to the compiler to emit SIMD instructions
                 * (AVX-512 on icelake) for the inner FMA loop.
                 * The loop has no dependencies across j iterations, making
                 * it safe to vectorise. */
                #pragma omp simd
                for (int j = p + 1; j < n; j++)
                    row_i[j] -= c_ip * row_p[j];
            }
            /* --- implicit barrier: step p fully complete before p+1 --- */
        }
    }
    /* End of parallel region: all threads join here */

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
