#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP header: required for the parallel, single, for, and simd directives below. */

#ifndef BLOCK_NB
#define BLOCK_NB 96    /* default panel width for the blocked baseline; retained for consistency with the main full-node runs. */
#endif

#define MAX_N 100000

/*
 * v4_openmp_blocked: first blocked OpenMP version.
 *
 * Replaces the flat column-by-column OpenMP structure from v3 with a
 * panel-blocked algorithm to reduce synchronization frequency and improve
 * locality in the trailing update.
 *
 * Output layout:
 * lower triangle (i >= j): L[i,j]
 * upper triangle (i <  j): L^T[i,j] = L[j,i]   (filled at end)
 *
 * Per panel k..kend-1:
 * Phase 1 [serial]   factor diagonal block A[k:kend, k:kend]
 * Phase 2 [parallel] TRSM: solve block column below the panel
 * Phase 3 [parallel] SYRK: update trailing lower triangle
 *
 * Synchronisation count:
 * 3 * ceil(n / BLOCK_NB) implicit barriers
 * - end of omp single
 * - end of TRSM omp for
 * - end of SYRK omp for
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open parallel region once to avoid O(N/NB) thread creation overhead.
     * default(none) forces explicit data sharing declarations to prevent race conditions. */
    #pragma omp parallel default(none) shared(c, n)
    {
        for (int k = 0; k < n; k += BLOCK_NB) {
            int kend = (k + BLOCK_NB < n) ? (k + BLOCK_NB) : n;

            /* Phase 1: factor the diagonal block serially.
            * omp single: one thread computes the panel factor L11 while the others
            * wait at the implicit barrier. This phase updates only the lower triangle. */
            #pragma omp single
            {
                for (int p = k; p < kend; p++) {
                    double *row_p = &c[(size_t)p * n];

                    /* Diagonal pivot */
                    double diag = sqrt(row_p[p]);
                    row_p[p] = diag;
                    double inv_diag = 1.0 / diag;

                    /* Normalize COLUMN p within the current panel only. */
                    for (int i = p + 1; i < kend; i++) {
                        c[(size_t)i * n + p] *= inv_diag;
                    }

                    /* Within-panel trailing update.
                     * IMPORTANT: use lower-triangle entry c[j,p], not upper c[p,j].
                     * Previous panels' SYRK updates maintain the lower triangle only. */
                    for (int i = p + 1; i < kend; i++) {
                        double *row_i = &c[(size_t)i * n];
                        double c_ip = row_i[p];   /* L[i,p], already normalised */

                        /* omp simd enables vectorization for the inner update loop.
                         * Note: c[j*n+p] is stride-n (non-unit-stride), which can limit
                         * cache locality relative to packed stride-1 access (see v5 OPT-1). */
                        #pragma omp simd
                        for (int j = p + 1; j < kend; j++) {
                            row_i[j] -= c_ip * c[(size_t)j * n + p];
                        }
                    }
                }
            }
            /* implicit barrier: diagonal block complete before TRSM begins */

            /* Micro-optimization: skip parallel phases if this is the final block */
            if (kend >= n) {
                continue;
            }

            /* Phase 2: TRSM below the panel.
             * schedule(static): the work per row is uniform (kend-k columns),
             * so static scheduling distributes load evenly with minimal overhead. */
            #pragma omp for schedule(static)
            for (int i = kend; i < n; i++) {
                for (int p = k; p < kend; p++) {
                    double val = c[(size_t)i * n + p];
                    for (int s = k; s < p; s++) {
                        val -= c[(size_t)i * n + s] * c[(size_t)p * n + s];
                    }
                    c[(size_t)i * n + p] = val / c[(size_t)p * n + p];
                }
            }
            /* implicit barrier: L21 block fully solved before trailing update */

            /* Phase 3: trailing SYRK update of lower triangle only.
             * Work per row increases with i (row i=kend has 1 update; i=n-1 has n-kend).
             * schedule(guided) may not balance this increasing workload optimally;
             * see v5_openmp_blocked for schedule(static,1). */
            #pragma omp for schedule(guided)
            for (int i = kend; i < n; i++) {
                double *row_i = &c[(size_t)i * n];
                double *panel_i = &c[(size_t)i * n + k];
                int panel_width = kend - k;

                for (int j = kend; j <= i; j++) {
                    double *panel_j = &c[(size_t)j * n + k];
                    double dot = 0.0;

                    /* omp simd encourages vectorization of the stride-1 dot product. */
                    #pragma omp simd reduction(+:dot)
                    for (int p = 0; p < panel_width; p++) {
                        dot += panel_i[p] * panel_j[p];
                    }

                    row_i[j] -= dot;
                }
            }
            /* implicit barrier: Trailing update complete before next panel iteration */
        }

        /* Final lower -> upper copy so output matches coursework interface. 
         * schedule(static): reasonable load distribution for simple O(N^2) pass. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                c[(size_t)i * n + j] = c[(size_t)j * n + i];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
