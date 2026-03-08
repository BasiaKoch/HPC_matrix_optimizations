#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 100000

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

        /* update submatrix below-right of diagonal element.
         *
         * Loop interchange vs v1: outer=i, inner=j.
         * c is stored row-major, so c[i*n+j] is contiguous as j increases.
         * Inner loop now reads and writes contiguous memory — one cache line
         * covers 8 consecutive doubles, so hardware prefetch works effectively.
         * v1 had outer=j, inner=i: c[i*n+j] strided by n doubles per step,
         * causing a cache miss on every iteration of the inner loop.
         *
         * c[i*n+p] is loop-invariant in j, so it is hoisted out of the inner
         * loop to avoid reloading it on every iteration. */
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
