#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP runtime header */

#ifndef BLOCK_NB
#define BLOCK_NB 128   /* panel width; tune with -DBLOCK_NB=N */
#endif

#define MAX_N 100000

/*
 * v5_openmp_blocked: panel-blocked Cholesky with OpenMP.
 *
 * Why the column-by-column approach (v3_openmp) failed to scale
 * ─────────────────────────────────────────────────────────────
 * v3_openmp synchronises after every Cholesky step p → O(n) barriers.
 * For n=2000 with 76 threads, ~4000 barriers × ~500 µs each ≈ 2 s of pure
 * synchronisation overhead, drowning out the ~10 ms of useful parallel work.
 *
 * Panel-blocked algorithm (right-looking, inspired by LAPACK dpotrf)
 * ──────────────────────────────────────────────────────────────────
 * Process nb columns at a time as a "panel":
 *
 *   for k = 0, nb, 2·nb, ...
 *     [serial]   Panel factorisation: unblocked Cholesky on the nb×nb
 *                diagonal block c[k:kend, k:kend], plus column normalisation
 *                of the below-panel strip c[kend:n, k:kend]  (≡ TRSM).
 *     [parallel] Trailing SYRK: for each row i ≥ kend, subtract the
 *                rank-nb outer product from c[i, kend:i+1].
 *
 * Synchronisation count: 2 × ceil(n/nb) barriers instead of 2n.
 * Example: n=2000, nb=128 → 32 barriers vs 4000.  The parallel SYRK
 * contributes O(n² · nb) operations per panel — sufficient work to keep
 * 76 threads busy and amortise barrier cost.
 *
 * Data layout (same as v3_openmp):
 *   lower triangle (i ≥ j): c[i*n+j] = L[i,j]  ← used by test/report
 *   upper triangle (i < j): scratch space, not part of output
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open the parallel region once: create the thread pool before the k-loop.
     * Keeping it open across all panel iterations avoids O(n/nb) thread
     * creation/destruction costs.  One thread runs the serial panel
     * factorisation; all threads share the parallel SYRK. */
    #pragma omp parallel default(none) shared(c, n)
    {
        for (int k = 0; k < n; k += BLOCK_NB) {
            /* kend: one-past-end of the current panel (clamped to n) */
            int kend = k + BLOCK_NB < n ? k + BLOCK_NB : n;

            /* ── Panel factorisation (serial) ─────────────────────────────
             * omp single: exactly one thread runs the unblocked right-looking
             * Cholesky on the diagonal block c[k:kend, k:kend].  It also
             * performs the TRSM step (column normalisation for rows kend..n-1),
             * so c[i,p] = L[i,p] is ready for all below-panel rows before
             * the parallel SYRK begins.
             * Implicit barrier at end of omp single ensures every thread sees
             * the completed panel and TRSM writes before entering the SYRK. */
            #pragma omp single
            {
                for (int p = k; p < kend; p++) {
                    double *row_p = &c[p * n];

                    /* Diagonal pivot: c[p,p] ← sqrt(c[p,p]) = L[p,p] */
                    double diag     = sqrt(row_p[p]);
                    row_p[p]        = diag;
                    double inv_diag = 1.0 / diag;

                    /* Row normalisation within panel (columns p+1..kend-1).
                     * Columns j ≥ kend are not needed by the SYRK (it uses
                     * the lower triangle c[j,p], not c[p,j]), so we skip
                     * normalising them and avoid touching cold cache lines. */
                    for (int j = p + 1; j < kend; j++)
                        row_p[j] *= inv_diag;

                    /* Column normalisation for ALL rows below diagonal.
                     * Rows p+1..kend-1: needed by the within-panel update.
                     * Rows kend..n-1:   TRSM — sets L[i,p] in the lower
                     *   triangle so the parallel SYRK can read c[i,p]=L[i,p]. */
                    for (int i = p + 1; i < n; i++)
                        c[i * n + p] *= inv_diag;

                    /* Within-panel trailing update: rows and cols p+1..kend-1.
                     * Below-panel rows (i ≥ kend) are deferred to the parallel
                     * SYRK so all threads share that dominant O(n²·nb) work. */
                    for (int i = p + 1; i < kend; i++) {
                        double *row_i = &c[i * n];
                        double  c_ip  = row_i[p];   /* L[i,p], already normalised */
                        for (int j = p + 1; j < kend; j++)
                            row_i[j] -= c_ip * row_p[j];
                    }
                }
            }
            /* --- implicit barrier: panel factorisation + TRSM complete --- */

            /* Skip SYRK if this is the last (or only) panel */
            if (kend >= n) continue;

            /* ── Parallel trailing SYRK ────────────────────────────────────
             * For each row i ≥ kend, subtract the rank-nb outer product from
             * the trailing submatrix lower triangle:
             *
             *   c[i, j] -= Σ_{p=k}^{kend-1}  c[i,p] · c[j,p]   j ∈ [kend, i]
             *
             * c[i,p] = L[i,p] and c[j,p] = L[j,p] (both in lower triangle,
             * set during column normalisation above).  Both panel strips
             * c[i, k:kend] and c[j, k:kend] are contiguous (stride 1), so
             * the inner p-loop vectorises efficiently with AVX-512 FMA.
             *
             * schedule(guided): row i requires (i - kend + 1) j-iterations.
             * The workload grows with i, giving a triangular distribution.
             * guided assigns larger chunks first and shrinks them dynamically,
             * balancing load across threads better than static scheduling
             * for this unequal-work triangular access pattern. */
            #pragma omp for schedule(guided)
            for (int i = kend; i < n; i++) {
                double *panel_i     = &c[i * n + k];  /* L[i, k:kend], stride 1 */
                double *row_i       = &c[i * n];
                int     panel_width = kend - k;

                for (int j = kend; j <= i; j++) {
                    double *panel_j = &c[j * n + k]; /* L[j, k:kend], stride 1 */
                    double  dot     = 0.0;

                    /* Dot product of two panel strips: both are stride-1 vectors
                     * of length panel_width (= nb, compile-time known).
                     * omp simd + reduction(+:dot): instructs the compiler to
                     * emit SIMD instructions (AVX-512 on icelake) for the FMA
                     * loop; the reduction accumulates partial sums across lanes. */
                    #pragma omp simd reduction(+:dot)
                    for (int p = 0; p < panel_width; p++)
                        dot += panel_i[p] * panel_j[p];

                    row_i[j] -= dot;
                }
            }
            /* --- implicit barrier at end of omp for: SYRK complete --- */
        }
    }
    /* End of parallel region: all threads join here */

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
