/*
 * v5_openmp_blocked: panel-blocked Cholesky with four cache/parallelism optimisations
 * over v4_openmp_blocked.
 *
 * Optimisation summary (each is independently motivated below):
 *
 *  OPT-1  Phase 1 — column packing to stride-1
 *         c[j*n+p] in the within-panel trailing update has stride n (64 KB for n=8000).
 *         Pre-copy into col_p[BLOCK_NB] converts every hot inner read to stride-1.
 *
 *  OPT-2  Phase 2 — private per-thread L11 cache
 *         c[p*n+s] in TRSM walks NB rows separated by n doubles (stride 64 KB), landing
 *         in L3 on every miss.  Each thread independently packs the NB×NB lower triangle
 *         of L11 into a private stack array l11[BLOCK_NB×BLOCK_NB] (≈72 KB at NB=96,
 *         fits in L2).  This reduces long-stride accesses and improves locality in TRSM.
 *
 *  OPT-3  Phase 3 — SYRK j-loop unrolled ×4 (register blocking)
 *         For each row i the inner p-loop loads panel_i[p] once and accumulates
 *         four independent dot products d0..d3 against panel_j, panel_{j+1}, panel_{j+2},
 *         panel_{j+3}.  This exposes four-way FMA parallelism; AVX-512 has two 8-wide
 *         FMA units, so four independent accumulators give the compiler more ILP to use.
 *
 *  OPT-4  Phase 3 — schedule(static,1) instead of schedule(guided)
 *         Work per row i increases linearly from 1 (i=kend) to (n-kend) (i=n-1).
 *         schedule(static,1) interleaves rows round-robin across threads, giving each
 *         thread a mix of heavy and light rows.  schedule(guided) assigns large initial
 *         chunks of light rows and small final chunks of heavy rows, which can leave
 *         the last thread doing disproportionate work.
 *
 * Synchronisation count: 3 * ceil(n/BLOCK_NB) implicit barriers (unchanged from v4).
 * L11 packing inside Phase 2 (after omp single barrier) is race-free: all threads
 * independently read the same finalised c[k:kend, k:kend] data.
 */

#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP header: required for the parallel, single, for, and simd directives below. */

#ifndef BLOCK_NB
#define BLOCK_NB 96   /* default panel width for the final kernel; best mean choice at 76 threads in the block sweep. */
#endif

#define MAX_N 100000

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open a single parallel region for the entire factorisation.
     * This avoids O(n/NB) thread-pool create/destroy overheads.
     * default(none) forces every shared variable to be listed explicitly,
     * preventing accidental data races. */
    #pragma omp parallel default(none) shared(c, n)
    {
        /* OPT-2: private per-thread L11 cache.
         * Declared inside the parallel region but outside the k-loop so it is
         * allocated once per thread (on the thread's own stack) and reused across
         * all panels.  Size: BLOCK_NB*BLOCK_NB*8 bytes = 73728 B at NB=96, which
         * fits in L2 (1.25 MB per core on icelake). */
        double l11[BLOCK_NB * BLOCK_NB];

        for (int k = 0; k < n; k += BLOCK_NB) {
            int kend       = (k + BLOCK_NB < n) ? (k + BLOCK_NB) : n;
            int panel_width = kend - k;

            /* ── Phase 1: factorise NB×NB diagonal block ────────────────────
             * omp single: exactly one thread executes this while the others
             * wait at the implicit barrier below.  Phase 1 is O(NB³) serial
             * work; for NB=96 this is ~884K FLOP — negligible vs Phase 3. */
            #pragma omp single
            {
                for (int p = k; p < kend; p++) {
                    /* Diagonal pivot */
                    double diag     = sqrt(c[(size_t)p * n + p]);
                    c[(size_t)p * n + p] = diag;
                    double inv_diag = 1.0 / diag;

                    /* Normalise column p within the panel: c[i,p] /= L[p,p].
                     * Covers only rows i < kend; below-panel rows are handled
                     * by the TRSM in Phase 2. */
                    for (int i = p + 1; i < kend; i++)
                        c[(size_t)i * n + p] *= inv_diag;

                    /* OPT-1: pack column p (rows p+1..kend-1) into col_p.
                     * c[j*n+p] has stride n = 64 KB between adjacent j-values
                     * (for n=8000), causing L2/L3 misses in the loop below.
                     * Copying to a stack array converts those reads to stride-1
                     * and allows the compiler to keep the entire column in
                     * registers or L1 cache across the i-loop. */
                    double col_p[BLOCK_NB];
                    for (int j = p + 1; j < kend; j++)
                        col_p[j - p - 1] = c[(size_t)j * n + p];

                    /* Within-panel trailing update using stride-1 col_p.
                     * IMPORTANT: use lower-triangle entry (col_p[j-p-1] = L[j,p]),
                     * not c[p*n+j] (upper triangle, stale for panels k > 0).
                     * omp simd vectorises the j-loop with AVX-512 FMA. */
                    for (int i = p + 1; i < kend; i++) {
                        double *row_i = &c[(size_t)i * n];
                        double  c_ip  = row_i[p];

                        /* omp simd: hint to generate SIMD instructions.
                         * With -O3 -march=native -ffast-math and AVX-512,
                         * GCC emits 8-wide FMA for this loop. */
                        #pragma omp simd
                        for (int j = p + 1; j < kend; j++)
                            row_i[j] -= c_ip * col_p[j - p - 1];
                    }
                }
            }
            /* Implicit barrier: L11 fully factorised; all threads may now read
             * the finalised c[k:kend, k:kend] safely. */

            if (kend >= n) continue;   /* last panel: no trailing submatrix */

            /* OPT-2 continued: each thread independently packs its private l11
             * from the now-finalised L11 block.
             * Redundant across threads but cheap: NB*(NB+1)/2 ≈ 4656 scalar reads
             * (36 KB) per panel, amortised over O(n²/NB) TRSM FLOP per thread.
             * l11[(p-k)*BLOCK_NB + (s-k)] = L[p,s] for k <= s <= p < kend.
             * All threads read the same c[] addresses — no write conflict. */
            for (int p = k; p < kend; p++)
                for (int s = k; s <= p; s++)
                    l11[(p - k) * BLOCK_NB + (s - k)] = c[(size_t)p * n + s];

            /* ── Phase 2: TRSM below the panel ─────────────────────────────
             * schedule(static): uniform work per row (panel_width columns each),
             * so static distribution incurs minimal scheduling overhead.
             * Reads l11 (L2-resident) instead of c[p*n+s] (L3-resident). */
            #pragma omp for schedule(static)
            for (int i = kend; i < n; i++) {
                double *row_i = &c[(size_t)i * n];
                for (int p = k; p < kend; p++) {
                    double val = row_i[p];

                    /* Subtract within-panel contributions from columns k..p-1.
                     * l11[(p-k)*BLOCK_NB + (s-k)] replaces c[p*n+s]:
                     * stride-NB (768 B between rows) vs stride-n (64 KB). */
                    #pragma omp simd reduction(-:val)
                    for (int s = k; s < p; s++)
                        val -= row_i[s] * l11[(p - k) * BLOCK_NB + (s - k)];

                    row_i[p] = val / l11[(p - k) * BLOCK_NB + (p - k)];
                }
            }
            /* Implicit barrier: L21 block (below-panel TRSM) fully computed. */

            /* ── Phase 3: SYRK trailing lower-triangle update ───────────────
             *
             * OPT-4: schedule(static,1) round-robin row assignment.
             * Work for row i = (i - kend + 1) dot products, increasing linearly.
             * Round-robin gives each of T threads rows kend, kend+T, kend+2T, ...
             * providing a balanced mix of heavy (large i) and light (small i) rows.
             *
             * OPT-3: j-loop unrolled ×4.  For each row i, process four output
             * columns j, j+1, j+2, j+3 in one pass over panel_i[0:NB].
             * panel_i[p] is loaded once per p and reused for four FMAs, increasing
             * instruction-level parallelism and SIMD reuse. */
            #pragma omp for schedule(static, 1)
            for (int i = kend; i < n; i++) {
                double *row_i   = &c[(size_t)i * n];
                double *panel_i = row_i + k;   /* L[i, k:kend], stride-1 */

                int j;
                /* OPT-3: 4-wide j-unrolled main loop.
                 * Four independent accumulators d0..d3 break the reduction
                 * chain and allow out-of-order execution units to stay busy. */
                for (j = kend; j <= i - 3; j += 4) {
                    double *pj0 = &c[(size_t) j      * n + k];
                    double *pj1 = &c[(size_t)(j + 1) * n + k];
                    double *pj2 = &c[(size_t)(j + 2) * n + k];
                    double *pj3 = &c[(size_t)(j + 3) * n + k];
                    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;

                    /* omp simd reduction: vectorise over p with 4 independent
                     * accumulators, improving opportunities for SIMD + ILP. */
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

                /* Scalar cleanup for the last 0..3 j-values. */
                for (; j <= i; j++) {
                    double *pj  = &c[(size_t)j * n + k];
                    double  dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (int p = 0; p < panel_width; p++)
                        dot += panel_i[p] * pj[p];
                    row_i[j] -= dot;
                }
            }
            /* Implicit barrier: trailing update complete for this panel. */
        }

        /* Copy lower triangle to upper to satisfy the coursework interface.
         * schedule(static): O(n²) uniform work, static distribution is optimal. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                c[(size_t)i * n + j] = c[(size_t)j * n + i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
