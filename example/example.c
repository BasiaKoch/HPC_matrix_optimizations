/*
 * example.c — usage example for mphil_dis_cholesky
 *
 * Builds an n×n symmetric positive-definite covariance matrix using the
 * corr() function from the coursework brief, factorises it in-place, and
 * reports elapsed time, log-determinant, and performance in GFLOP/s.
 *
 * Build:
 *   make example VERSION=v5_openmp_blocked NB=128
 *
 * Run:
 *   ./example/example 4000
 *
 * CSD3 recommended settings:
 *   export OMP_NUM_THREADS=76
 *   export OMP_PROC_BIND=close
 *   export OMP_PLACES=cores
 *   ./example/example 8000
 */

#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * corr() — covariance kernel from the coursework brief.
 * Returns a decaying exponential covariance between grid points x and y
 * for a grid of size s.  The factor 16 sets the correlation length to s/4.
 */
static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

int main(int argc, char *argv[])
{
    int n = 1000;
    if (argc > 1) n = atoi(argv[1]);

    if (n < 1 || n > 100000) {
        fprintf(stderr, "Usage: %s [n]   (1 <= n <= 100000)\n", argv[0]);
        return 1;
    }

    printf("mphil_dis_cholesky example\n");
    printf("  n = %d   matrix bytes = %.1f MB\n",
           n, (double)n * n * sizeof(double) / 1e6);

    /* Allocate and fill the n×n SPD matrix with the corr() kernel. */
    double *c = malloc((size_t)n * n * sizeof(double));
    if (!c) {
        fprintf(stderr, "malloc failed for n=%d\n", n);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            c[(size_t)n*i + j] = corr(i, j, n);
        c[(size_t)n*i + i] = 1.0;   /* diagonal = 1 (variance = 1) */
    }

    /*
     * In-place Cholesky factorisation.
     * On return:
     *   lower triangle  c[i*n+j] (i >= j) contains L[i,j]
     *   upper triangle  c[i*n+j] (i <  j) contains L^T[i,j] = L[j,i]
     * Returns wall-clock time in seconds, or -1.0 on error.
     */
    double elapsed = mphil_dis_cholesky(c, n);
    if (elapsed < 0.0) {
        fprintf(stderr, "mphil_dis_cholesky returned error\n");
        free(c);
        return 1;
    }

    /*
     * log|C| = 2 * sum_{p=0}^{n-1} log(L[p,p])
     * (Coursework brief Eq. 4)
     */
    double logdet = 0.0;
    for (int p = 0; p < n; p++)
        logdet += log(c[(size_t)p*n + p]);
    logdet *= 2.0;

    /* Cholesky is O(n^3/3) floating-point operations. */
    double gflops = (double)n * n * n / 3.0 / elapsed / 1.0e9;

    printf("  elapsed = %.4f s\n", elapsed);
    printf("  log|C|  = %.6f\n",   logdet);
    printf("  GFlop/s = %.2f\n",   gflops);

    free(c);
    return 0;
}
