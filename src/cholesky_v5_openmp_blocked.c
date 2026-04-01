#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP pragmas used below */

#ifndef BLOCK_NB
#define BLOCK_NB 96   /* default block size used in the main experiments */
#endif

#define MAX_N 100000

/*
 * v5_openmp_blocked: final blocked OpenMP implementation.
 *
 * Relative to v4, this version adds:
 *  - column packing in Phase 1,
 *  - private per-thread packing of the L11 block in Phase 2,
 *  - four-way unrolling in Phase 3,
 *  - schedule(static,1) in Phase 3 for better load balance.
 *
 * The blocked algorithm has three phases per panel:
 *  1. factor the diagonal block,
 *  2. solve below the panel,
 *  3. update the trailing lower triangle.
 *
 * As in v4, this gives 3 * ceil(n / BLOCK_NB) implicit barriers.
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open one parallel region for the whole factorisation.
     * default(none) forces explicit sharing rules. */
    #pragma omp parallel default(none) shared(c, n)
    {
        /* Private packed copy of the current L11 block for each thread.
         * Declared here so it is reused across panels. */
        double l11[BLOCK_NB * BLOCK_NB];

        for (int k = 0; k < n; k += BLOCK_NB) {
            int kend = (k + BLOCK_NB < n) ? (k + BLOCK_NB) : n;
            int panel_width = kend - k;

            /* Phase 1: factor the diagonal block.
             * One thread performs this work; the others wait at the
             * implicit barrier at the end of omp single. */
            #pragma omp single
            {
                for (int p = k; p < kend; p++) {
                    double diag = sqrt(c[(size_t)p * n + p]);
                    c[(size_t)p * n + p] = diag;
                    double inv_diag = 1.0 / diag;

                    /* Normalise column p within the current panel. */
                    for (int i = p + 1; i < kend; i++)
                        c[(size_t)i * n + p] *= inv_diag;

                    /* Pack column p into a contiguous buffer.
                     * This replaces long-stride reads with stride-1 access. */
                    double col_p[BLOCK_NB];
                    for (int j = p + 1; j < kend; j++)
                        col_p[j - p - 1] = c[(size_t)j * n + p];

                    /* Update the rest of the panel using the packed column.
                     * Use the lower-triangle value L[j,p], not c[p,j], because
                     * for later panels only the lower triangle is kept current. */
                    for (int i = p + 1; i < kend; i++) {
                        double *row_i = &c[(size_t)i * n];
                        double c_ip = row_i[p];

                        /* SIMD is safe here because j-iterations are independent. */
                        #pragma omp simd
                        for (int j = p + 1; j < kend; j++)
                            row_i[j] -= c_ip * col_p[j - p - 1];
                    }
                }
            }
            /* Implicit barrier: diagonal block is complete here. */

            if (kend >= n) continue;   /* no trailing matrix after the last panel */

            /* Pack the finished L11 block into a private buffer for this thread.
             * All threads read the same data, so this is race-free. */
            for (int p = k; p < kend; p++)
                for (int s = k; s <= p; s++)
                    l11[(p - k) * BLOCK_NB + (s - k)] = c[(size_t)p * n + s];

            /* Phase 2: TRSM below the panel.
             * schedule(static) is appropriate because each row has the same
             * number of columns to solve. */
            #pragma omp for schedule(static)
            for (int i = kend; i < n; i++) {
                double *row_i = &c[(size_t)i * n];
                for (int p = k; p < kend; p++) {
                    double val = row_i[p];

                    /* Subtract contributions from earlier columns in the panel.
                     * Read from packed l11 instead of long-stride accesses into c. */
                    #pragma omp simd reduction(-:val)
                    for (int s = k; s < p; s++)
                        val -= row_i[s] * l11[(p - k) * BLOCK_NB + (s - k)];

                    row_i[p] = val / l11[(p - k) * BLOCK_NB + (p - k)];
                }
            }
            /* Implicit barrier: the block below the panel is complete. */

            /* Phase 3: update the trailing lower triangle.
             * schedule(static,1) assigns rows round-robin, which helps because
             * the amount of work per row increases with i. */
            #pragma omp for schedule(static, 1)
            for (int i = kend; i < n; i++) {
                double *row_i = &c[(size_t)i * n];
                double *panel_i = row_i + k;   /* L[i, k:kend] */

                int j;

                /* Main loop: update four output columns at once.
                 * Four accumulators expose more independent work. */
                for (j = kend; j <= i - 3; j += 4) {
                    double *pj0 = &c[(size_t) j      * n + k];
                    double *pj1 = &c[(size_t)(j + 1) * n + k];
                    double *pj2 = &c[(size_t)(j + 2) * n + k];
                    double *pj3 = &c[(size_t)(j + 3) * n + k];
                    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;

                    #pragma omp simd reduction(+:d0,d1,d2,d3)
                    for (int p = 0; p < panel_width; p++) {
                        double pi = panel_i[p];
                        d0 += pi * pj0[p];
                        d1 += pi * pj1[p];
                        d2 += pi * pj2[p];
                        d3 += pi * pj3[p];
                    }

                    row_i[j    ] -= d0;
                    row_i[j + 1] -= d1;
                    row_i[j + 2] -= d2;
                    row_i[j + 3] -= d3;
                }

                /* Cleanup for the final 0..3 columns. */
                for (; j <= i; j++) {
                    double *pj = &c[(size_t)j * n + k];
                    double dot = 0.0;

                    #pragma omp simd reduction(+:dot)
                    for (int p = 0; p < panel_width; p++)
                        dot += panel_i[p] * pj[p];

                    row_i[j] -= dot;
                }
            }
            /* Implicit barrier: trailing update complete for this panel. */
        }

        /* Copy the lower triangle to the upper triangle so the output matches
         * the required interface: lower = L, upper = L^T. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                c[(size_t)i * n + j] = c[(size_t)j * n + i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}