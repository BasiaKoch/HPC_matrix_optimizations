#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 100000

/*
 * v1_baseline: direct in-place Cholesky factorisation following the coursework
 * pseudocode as closely as possible.
 *
 * Rationale:
 * - keep the structure simple so correctness is easy to verify;
 * - provide a clean reference point for later optimisation stages;
 * - preserve both lower-triangular L and upper-triangular L^T in the output,
 *   matching the coursework interface exactly.
 */

double mphil_dis_cholesky(double *c, int n)
{
    /* Reject invalid problem sizes early so the routine fails predictably
     * rather than indexing beyond the supported coursework limit. */
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    /* MONOTONIC time is used so elapsed runtime is not affected by wall-clock
     * adjustments while benchmarking the factorisation. */
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Step through the matrix one pivot column at a time.
     * Column p depends on all previous columns, so this outer loop is inherently
     * serial in the baseline algorithm. */
    for (int p = 0; p < n; p++) {

        /* The working diagonal entry now holds the remaining Schur-complement
         * value for column p; its square root is L[p,p]. */
        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;

        /* Normalise the row to the right of the pivot.
         * This stores the upper-triangular part of L^T so the final output
         * already matches the required coursework layout. */
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] /= diag;
        }

        /* Normalise the column below the pivot.
         * These are the actual L[i,p] entries used in later updates. */
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] /= diag;
        }

        /* Apply the rank-1 update to the trailing submatrix:
         *   C[i,j] <- C[i,j] - L[i,p] * L[j,p].
         * The loop ordering here matches the coursework pseudocode directly.
         * It is easy to read, but it is not cache-friendly for row-major data;
         * later versions change this ordering for performance. */
        for (int j = p + 1; j < n; j++) {
            for (int i = p + 1; i < n; i++) {
                c[i*n + j] -= c[i*n + p] * c[p*n + j];
            }
        }
    }

    /* Return only the factorisation time, excluding any matrix setup done by
     * the caller, so benchmark results are comparable across versions. */
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
