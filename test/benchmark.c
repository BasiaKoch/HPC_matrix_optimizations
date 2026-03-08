#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usage: ./test/benchmark <n> [reps]
 *
 * Runs Cholesky factorization on an n x n corr() matrix <reps> times
 * (default 3) and prints one CSV line per run to stdout:
 *
 *   n,threads,rep,time_s,gflops
 *
 * OMP_NUM_THREADS is read from the environment automatically by OpenMP.
 * The matrix is re-filled before each rep so we always time fresh data.
 *
 * Example — run and append to results file:
 *   OMP_NUM_THREADS=4 ./test/benchmark 4000 3 >> results/scaling.csv
 */

static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [reps]\n", argv[0]);
        return 1;
    }

    int n    = atoi(argv[1]);
    int reps = argc > 2 ? atoi(argv[2]) : 3;

    if (n < 1 || n > 100000) {
        fprintf(stderr, "n must be in [1, 100000]\n");
        return 1;
    }

    /* Read thread count from environment (set by OMP_NUM_THREADS) */
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    int threads = omp_threads ? atoi(omp_threads) : 1;

    double *c = malloc((size_t)n * n * sizeof(double));
    if (!c) { fprintf(stderr, "malloc failed\n"); return 1; }

    double flops = (double)n * n * n / 3.0;  /* n^3/3 ops for Cholesky */

    for (int r = 1; r <= reps; r++) {
        /* Re-fill matrix before each rep */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) c[i*n+j] = corr(i, j, n);
            c[i*n+i] = 1.0;
        }

        double t = mphil_dis_cholesky(c, n);
        double gflops = flops / t / 1.0e9;

        printf("%d,%d,%d,%.6f,%.4f\n", n, threads, r, t, gflops);
        fflush(stdout);
    }

    free(c);
    return 0;
}
