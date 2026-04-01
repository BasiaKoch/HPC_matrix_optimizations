#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 100000
/*
 * v2_serial_opt
 * -------------
 * Second implementation stage: keep the baseline algorithm unchanged
 * mathematically, but improve single-thread performance through better
 * memory access in the trailing update.
 *
 * Changes relative to v1:
 * - interchange the trailing-update loops so the inner loop walks
 *   contiguous row-major memory;
 * - hoist c[i*n + p] out of the inner loop because it is invariant in j.
 *
 * Purpose of this version:
 * - test how much performance can be gained from simple serial
 *   locality/structure improvements before adding OpenMP.
 */
double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int p = 0; p < n; p++) {  /* move along the diagonal of the matrix */

        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;  /* update diagonal element */

        /* update row to right of diagonal element */
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] /= diag;
        }

        /* update column below diagonal element */
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] /= diag;
        }

        /* Update the submatrix below-right of the diagonal element.
         *
         * Loop interchange vs v1: outer=i, inner=j.
         * c is stored row-major, so c[i*n+j] is contiguous as j increases.
         * The inner loop therefore reads and writes contiguous memory,
         * which is much more cache-friendly than v1.
         *
         * c[i*n+p] is invariant in j, so it is hoisted out of the inner
         * loop to avoid reloading it on every iteration.
         */
        for (int i = p + 1; i < n; i++) {
            double c_ip = c[i*n + p];
            for (int j = p + 1; j < n; j++) {
                c[i*n + j] -= c_ip * c[p*n + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
