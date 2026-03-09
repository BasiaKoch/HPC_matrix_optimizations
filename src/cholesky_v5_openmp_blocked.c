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
 * Why the column-by-column approach (v3_openmp) scales poorly for large n
 * ────────────────────────────────────────────────────────────────────────
 * v3_openmp synchronises after every Cholesky step p → O(n) barriers.
 * For n=8000 at 76 threads (measured on CSD3 icelake) this limits speedup
 * to ~8× despite 76 cores, because each barrier stalls all threads waiting
 * for the single-threaded diagonal/normalisation work to finish.  For small
 * n (≤ 4000) scaling is reasonable; for large n it saturates quickly.
 *
 * Panel-blocked algorithm (right-looking, inspired by LAPACK dpotrf)
 * ──────────────────────────────────────────────────────────────────
 * Process nb columns at a time as a "panel":
 *
 *   for k = 0, nb, 2·nb, ...
 *     [serial]   Phase 1 — Panel factorisation: unblocked Cholesky on the
 *                nb×nb diagonal block c[k:kend, k:kend] only.  Column
 *                normalisation is restricted to within-panel rows (i < kend).
 *     [parallel] Phase 2 — TRSM: for each row i ≥ kend, solve the lower-
 *                triangular system to get L[i, k:kend]:
 *                  L[i,p] = (c[i,p] − Σ_{s=k}^{p−1} L[i,s]·L[p,s]) / L[p,p]
 *                This is the correct triangular solve; a simple division
 *                c[i,p] /= L[p,p] (without the sum) would be wrong for p > k
 *                because it omits the within-panel Schur contributions.
 *     [parallel] Phase 3 — Trailing SYRK: for each row i ≥ kend, subtract
 *                the rank-nb outer product from c[i, kend:i+1].
 *
 * Synchronisation count: 3 × ceil(n/nb) barriers instead of 2n.
 * Example: n=2000, nb=128 → 47 barriers vs 4000.  The parallel TRSM and
 * SYRK both contribute sufficient work to keep threads busy.
 *
 * Data layout (same as v3_openmp):
 *   lower triangle (i ≥ j): c[i*n+j] = L[i,j]
 *   upper triangle (i < j): c[i*n+j] = L^T[i,j] = L[j,i]  (filled at end)
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
     * factorisation; all threads share the parallel TRSM and SYRK. */
    #pragma omp parallel default(none) shared(c, n)
    {
        for (int k = 0; k < n; k += BLOCK_NB) {
            /* kend: one-past-end of the current panel (clamped to n) */
            int kend = k + BLOCK_NB < n ? k + BLOCK_NB : n;

            /* ── Phase 1: Panel factorisation (serial) ─────────────────────
             * omp single: exactly one thread runs an unblocked right-looking
             * Cholesky on the diagonal block c[k:kend, k:kend].
             *
             * Column normalisation is intentionally restricted to within-panel
             * rows (i < kend).  The below-panel rows (i ≥ kend) require a
             * proper triangular solve (TRSM) in Phase 2, not a simple division:
             * for p > k, the entry c[i,p] must first subtract contributions
             * from within-panel columns s=k..p-1 before dividing by L[p,p].
             *
             * Implicit barrier at end of omp single ensures every thread sees
             * the completed diagonal block before Phase 2 begins. */
            #pragma omp single
            {
                for (int p = k; p < kend; p++) {
                    double *row_p = &c[p * n];

                    /* Diagonal pivot: c[p,p] ← sqrt(c[p,p]) = L[p,p] */
                    double diag     = sqrt(row_p[p]);
                    row_p[p]        = diag;
                    double inv_diag = 1.0 / diag;

                    /* Row normalisation within panel (columns p+1..kend-1).
                     * Sets c[p,j] = L[j,p] for j in [p+1,kend), used by the
                     * within-panel trailing update below. */
                    for (int j = p + 1; j < kend; j++)
                        row_p[j] *= inv_diag;

                    /* Column normalisation within panel only (rows p+1..kend-1).
                     * Below-panel rows (i ≥ kend) are handled in Phase 2. */
                    for (int i = p + 1; i < kend; i++)
                        c[i * n + p] *= inv_diag;

                    /* Within-panel trailing update: rows and cols p+1..kend-1. */
                    for (int i = p + 1; i < kend; i++) {
                        double *row_i = &c[i * n];
                        double  c_ip  = row_i[p];   /* L[i,p], already normalised */
                        for (int j = p + 1; j < kend; j++)
                            row_i[j] -= c_ip * row_p[j];
                    }
                }
            }
            /* --- implicit barrier: diagonal block c[k:kend,k:kend] complete --- */

            /* Skip TRSM and SYRK if this is the last (or only) panel */
            if (kend >= n) continue;

            /* ── Phase 2: Parallel TRSM ────────────────────────────────────
             * For each row i ≥ kend, compute L[i, k:kend] by solving the
             * lower-triangular system L11 · x = a  (row-by-row TRSM):
             *
             *   L[i,p] = (c[i,p] − Σ_{s=k}^{p-1} L[i,s]·L[p,s]) / L[p,p]
             *
             * c[i,p] entering this phase holds the Schur complement value
             * accumulated from all previous panels (k'=0..k-1); the inner
             * s-sum subtracts the within-panel contributions s=k..p-1.
             *
             * c[p*n+s] = L[p,s] (lower triangle, set during Phase 1).
             * c[i*n+s] = L[i,s] (set by earlier iterations of the p-loop).
             * The p-loop is sequential per row i (triangular solve), but rows
             * are independent — hence parallelised with omp for over i.
             *
             * schedule(static): each row i does the same work (kend-k columns,
             * each with a short inner s-loop).  Static gives even distribution
             * with minimal overhead. */
            #pragma omp for schedule(static)
            for (int i = kend; i < n; i++) {
                for (int p = k; p < kend; p++) {
                    double val = c[i * n + p];
                    for (int s = k; s < p; s++)
                        val -= c[i * n + s] * c[p * n + s];
                    c[i * n + p] = val / c[p * n + p];
                }
            }
            /* --- implicit barrier: TRSM complete, L21 ready for SYRK --- */

            /* ── Phase 3: Parallel trailing SYRK ───────────────────────────
             * For each row i ≥ kend, subtract the rank-nb outer product from
             * the trailing submatrix lower triangle:
             *
             *   c[i, j] -= Σ_{p=k}^{kend-1}  c[i,p] · c[j,p]   j ∈ [kend, i]
             *
             * c[i,p] = L[i,p] and c[j,p] = L[j,p] (both set in Phase 2).
             * Both panel strips c[i, k:kend] and c[j, k:kend] are contiguous
             * (stride 1), so the inner p-loop vectorises efficiently with
             * AVX-512 FMA.
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

        /* ── Fill upper triangle ────────────────────────────────────────
         * The spec requires c[i,j] = L^T[i,j] = L[j,i] for all i < j.
         * This O(n^2) pass copies the lower triangle to the upper triangle
         * after all panels are complete; cost is dominated by the O(n^3)
         * factorisation.
         * schedule(static): each row i has (n-i-1) elements; static gives
         * a reasonable distribution without dynamic overhead. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                c[i * n + j] = c[j * n + i];
        /* --- implicit barrier at end of omp for --- */
    }
    /* End of parallel region: all threads join here */

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
