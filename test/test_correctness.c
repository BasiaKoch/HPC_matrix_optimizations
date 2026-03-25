/*
 * test_correctness.c — comprehensive correctness tests for mphil_dis_cholesky
 *
 * Test suite
 * ----------
 *  1. 2x2 spec example             exact element values + logdet
 *  2. 3x3 hand-computed            exact element values + logdet
 *  3. n=1 edge case
 *  4. Bounds guard                 n=0 and n=100001 must return -1.0
 *  5. Known-L reconstruction       generate L_ref, form A=L_ref*L_ref^T, compare
 *     sizes: 5, 95, 96, 97, 191, 192, 193, 255, 256, 257
 *  6. Numerically stressed SPD     diagonal L_ref with range 10^0 .. 10^{-12}
 *     sizes: 32, 96
 *  7. corr() reconstruction        coursework matrix; external logdet ref at
 *     n=50 and n=200, computed logdet reported at n=500
 *  8. Multi-thread agreement       2/4/8/76 threads vs 1-thread output
 *     sizes: 96, 200, 500
 *
 * Metrics on every nontrivial case
 * ---------------------------------
 *  max|L - L_ref|          (known-L and stressed, where L_ref is available)
 *  ||A - LL^T||_max        (all reconstruction tests)
 *  ||A - LL^T||_F / ||A||_F
 *  max_{i<j} |c[i*n+j] - c[j*n+i]|
 *  all L[i,i] > 0
 *  |logdet_computed - logdet_ref|  (derived from L_ref, or external ref for
 *                                  selected corr() sizes)
 *
 * Actual measured residuals are printed alongside every PASS/FAIL verdict.
 * logdet is a secondary sanity check only; it does not replace reconstruction.
 *
 * Build (performance flags):      make test VERSION=v5_openmp_blocked NB=96
 * Build (strict, no -ffast-math): make test-strict VERSION=v5_openmp_blocked NB=96
 * Returns: 0 if all tests pass, 1 otherwise.
 */

#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

/* Default panel width; overridden by -DBLOCK_NB=$(NB) at compile time. */
#ifndef BLOCK_NB
#  define BLOCK_NB 96
#endif

/* Tolerances */
#define TOL_EXACT        1e-10   /* exact rational results (2x2, 3x3)          */
#define TOL_KNOWN_L      1e-10   /* max|L - L_ref|, O(1) entries               */
#define TOL_RECON        1e-8    /* ||A - LL^T||_max                            */
#define TOL_REL_RECON    1e-10   /* ||A - LL^T||_F / ||A||_F                   */
#define TOL_SYM          1e-12   /* upper-triangle symmetry                     */
#define TOL_LOGDET       1e-6    /* |logdet_computed - logdet_ref|              */
#define TOL_THREAD       1e-10   /* max diff between t-thread and 1-thread L    */

/* ================================================================== */
/* Test harness                                                         */
/* ================================================================== */
static int tests_run    = 0;
static int tests_failed = 0;

static void check(const char *label, int ok)
{
    tests_run++;
    if (ok)
        printf("  PASS  %s\n", label);
    else {
        printf("  FAIL  %s\n", label);
        tests_failed++;
    }
}

/* ================================================================== */
/* Matrix generators                                                    */
/* ================================================================== */

/* Coursework corr() matrix: C[i,j] = 0.99*exp(-8*(i-j)^2/n^2), C[i,i]=1 */
static void fill_corr(double *c, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = (double)(i - j) / (double)n;
            c[(size_t)i*n+j] = 0.99 * exp(-0.5 * 16.0 * d * d);
        }
        c[(size_t)i*n+i] = 1.0;
    }
}

/*
 * Deterministic lower-triangular L_ref with positive diagonal.
 *   L[i,i] = 1 + 0.5*|sin(i+1)|          in [1.0, 1.5]
 *   L[i,j] = 0.3*sin(7i + 13j + 1)       for j < i, bounded in [-0.3, 0.3]
 */
static void fill_known_L(double *L, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++)
            L[(size_t)i*n+j] = 0.3 * sin((double)(7*i + 13*j + 1));
        L[(size_t)i*n+i] = 1.0 + 0.5 * fabs(sin((double)(i + 1)));
        for (int j = i+1; j < n; j++)
            L[(size_t)i*n+j] = 0.0;
    }
}

/*
 * Numerically stressed: purely diagonal L_ref spanning 10^0 .. 10^{-12}.
 * Off-diagonals are zero so A = diag(d_i^2) and Chol(A)[i,i] = d_i exactly.
 */
static void fill_stressed_L(double *L, int n)
{
    memset(L, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double t = (n > 1) ? (double)i / (double)(n - 1) : 0.0;
        L[(size_t)i*n+i] = pow(10.0, -12.0 * t);
    }
}

/* A = L * L^T  (uses lower triangle of L; stores full symmetric result) */
static void form_A(double *A, const double *L, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            int kmax = (i < j) ? i : j;
            for (int k = 0; k <= kmax; k++)
                s += L[(size_t)i*n+k] * L[(size_t)j*n+k];
            A[(size_t)i*n+j] = s;
        }
}

/* ================================================================== */
/* Metric checking                                                      */
/* ================================================================== */

/*
 * Compute and check all six metrics for a factorized result.
 *
 *  prefix     - short string prepended to each label (identifies the case)
 *  A_orig     - original SPD matrix (n×n row-major, must not be modified)
 *  fact       - factorized output: lower triangle = L, upper = L^T
 *  L_ref      - reference factor; NULL if unavailable
 *  n          - matrix dimension
 *  have_logdet_ref - 1 if logdet_ref is a valid external reference, 0 otherwise
 *  logdet_ref      - used only when have_logdet_ref=1; otherwise derived from
 *                    L_ref diagonal (if L_ref != NULL) or printed as INFO
 */
/* have_logdet_ref=0 means "derive from L_ref if available, else print INFO".
 * This avoids NaN arithmetic which -ffast-math/-ffinite-math-only breaks. */
static void check_all_metrics(const char *prefix,
                               const double *A_orig, const double *fact,
                               const double *L_ref, int n,
                               int have_logdet_ref, double logdet_ref)
{
    char lbl[320];

    /* ---- (a) Direct L comparison ---------------------------------- */
    if (L_ref) {
        double max_L_err = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) {
                double e = fabs(fact[(size_t)i*n+j] - L_ref[(size_t)i*n+j]);
                if (e > max_L_err) max_L_err = e;
            }
        snprintf(lbl, sizeof(lbl),
                 "%s  max|L-L_ref|=%.3e (tol=%.0e)", prefix, max_L_err, TOL_KNOWN_L);
        check(lbl, max_L_err < TOL_KNOWN_L);
    }

    /* ---- (b) Reconstruction max-norm and relative Frobenius ------- */
    double max_recon = 0.0, ssq_err = 0.0, ssq_A = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            int kmax = (i < j) ? i : j;
            for (int k = 0; k <= kmax; k++)
                s += fact[(size_t)i*n+k] * fact[(size_t)j*n+k];
            double e = fabs(s - A_orig[(size_t)i*n+j]);
            if (e > max_recon) max_recon = e;
            ssq_err += e * e;
            ssq_A   += A_orig[(size_t)i*n+j] * A_orig[(size_t)i*n+j];
        }
    double rel_recon = (ssq_A > 0.0) ? sqrt(ssq_err / ssq_A) : sqrt(ssq_err);

    snprintf(lbl, sizeof(lbl),
             "%s  ||A-LL^T||_max=%.3e (tol=%.0e)", prefix, max_recon, TOL_RECON);
    check(lbl, max_recon < TOL_RECON);

    snprintf(lbl, sizeof(lbl),
             "%s  ||A-LL^T||_F/||A||_F=%.3e (tol=%.0e)",
             prefix, rel_recon, TOL_REL_RECON);
    check(lbl, rel_recon < TOL_REL_RECON);

    /* ---- (c) Symmetry --------------------------------------------- */
    double max_sym = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            double e = fabs(fact[(size_t)i*n+j] - fact[(size_t)j*n+i]);
            if (e > max_sym) max_sym = e;
        }
    snprintf(lbl, sizeof(lbl),
             "%s  max_sym=%.3e (tol=%.0e)", prefix, max_sym, TOL_SYM);
    check(lbl, max_sym < TOL_SYM);

    /* ---- (d) Diagonal positivity ---------------------------------- */
    double min_diag = 1.0;
    int    diag_ok  = 1;
    for (int i = 0; i < n; i++) {
        double d = fact[(size_t)i*n+i];
        if (i == 0 || d < min_diag) min_diag = d;
        if (d <= 0.0) { diag_ok = 0; }
    }
    snprintf(lbl, sizeof(lbl),
             "%s  all L[i,i]>0  (min=%.3e)", prefix, min_diag);
    check(lbl, diag_ok);

    /* ---- (e) logdet ------------------------------------------------ */
    double logdet = 0.0;
    for (int i = 0; i < n; i++)
        logdet += log(fact[(size_t)i*n+i]);
    logdet *= 2.0;

    /* Determine reference: caller may supply one, or we derive from L_ref.
     * No NaN arithmetic — -ffast-math turns isnan() into a no-op. */
    double ref      = 0.0;
    int    have_ref = have_logdet_ref;
    if (!have_ref && L_ref) {
        for (int i = 0; i < n; i++)
            ref += log(L_ref[(size_t)i*n+i]);
        ref *= 2.0;
        have_ref = 1;
    } else if (have_ref) {
        ref = logdet_ref;
    }

    if (have_ref) {
        double diff = fabs(logdet - ref);
        snprintf(lbl, sizeof(lbl),
                 "%s  |logdet_diff|=%.3e (tol=%.0e)  got=%.6g ref=%.6g",
                 prefix, diff, TOL_LOGDET, logdet, ref);
        check(lbl, diff < TOL_LOGDET);
    } else {
        printf("  INFO  %s  logdet=%.10g (no reference)\n", prefix, logdet);
    }
}

/* ================================================================== */
/* Test 1 — 2x2 spec example                                           */
/* ================================================================== */
static void test_2x2(void)
{
    printf("\n=== Test 1: 2x2 spec example ===\n");
    double c[4] = {4.0, 2.0, 2.0, 26.0};
    double t = mphil_dis_cholesky(c, 2);
    check("L[0,0]=2",          fabs(c[0]-2.0) < TOL_EXACT);
    check("L[1,0]=1",          fabs(c[2]-1.0) < TOL_EXACT);
    check("L[1,1]=5",          fabs(c[3]-5.0) < TOL_EXACT);
    check("L^T[0,1]=1",        fabs(c[1]-1.0) < TOL_EXACT);
    double logdet = 2.0*(log(c[0])+log(c[3]));
    check("logdet=log(100)",   fabs(logdet-log(100.0)) < TOL_EXACT);
    check("elapsed>=0",        t >= 0.0);
}

/* ================================================================== */
/* Test 2 — 3x3 hand-computed                                          */
/* ================================================================== */
static void test_3x3(void)
{
    printf("\n=== Test 2: 3x3 hand-computed ===\n");
    double c[9] = {4.0,2.0,2.0, 2.0,3.0,1.0, 2.0,1.0,3.0};
    double t = mphil_dis_cholesky(c, 3);
    double sq2 = sqrt(2.0);
    check("L[0,0]=2",      fabs(c[0]-2.0) < TOL_EXACT);
    check("L[1,0]=1",      fabs(c[3]-1.0) < TOL_EXACT);
    check("L[1,1]=sqrt2",  fabs(c[4]-sq2) < TOL_EXACT);
    check("L[2,0]=1",      fabs(c[6]-1.0) < TOL_EXACT);
    check("L[2,1]=0",      fabs(c[7]-0.0) < TOL_EXACT);
    check("L[2,2]=sqrt2",  fabs(c[8]-sq2) < TOL_EXACT);
    check("L^T[0,1]=1",   fabs(c[1]-1.0) < TOL_EXACT);
    check("L^T[0,2]=1",   fabs(c[2]-1.0) < TOL_EXACT);
    check("L^T[1,2]=0",   fabs(c[5]-0.0) < TOL_EXACT);
    double logdet = 2.0*(log(c[0])+log(c[4])+log(c[8]));
    check("logdet=log(16)", fabs(logdet-log(16.0)) < TOL_EXACT);
    check("elapsed>=0",    t >= 0.0);
}

/* ================================================================== */
/* Test 3 — n=1 edge case                                              */
/* ================================================================== */
static void test_n1(void)
{
    printf("\n=== Test 3: n=1 edge case ===\n");
    double c[1] = {9.0};
    double t = mphil_dis_cholesky(c, 1);
    check("c[0]=3",     fabs(c[0]-3.0) < TOL_EXACT);
    check("elapsed>=0", t >= 0.0);
}

/* ================================================================== */
/* Test 4 — bounds guard                                               */
/* ================================================================== */
static void test_bounds(void)
{
    printf("\n=== Test 4: bounds guard ===\n");
    double c[1] = {4.0};
    check("n=0 → -1.0",      mphil_dis_cholesky(c, 0)      == -1.0);
    check("n=100001 → -1.0", mphil_dis_cholesky(c, 100001) == -1.0);
    check("n=1 → >=0",       mphil_dis_cholesky(c, 1)      >= 0.0);
    check("n=1: c[0]=2",     fabs(c[0]-2.0) < TOL_EXACT);
}

/* ================================================================== */
/* Test 5 — known-L reconstruction at block-boundary sizes             */
/* ================================================================== */
static void test_known_L(int n)
{
    printf("\n=== Test 5: known-L  n=%d ===\n", n);

    double *L_ref = malloc((size_t)n*n*sizeof(double));
    double *A     = malloc((size_t)n*n*sizeof(double));
    double *fact  = malloc((size_t)n*n*sizeof(double));
    if (!L_ref || !A || !fact) {
        printf("  SKIP n=%d (malloc failed)\n", n);
        free(L_ref); free(A); free(fact); return;
    }

    fill_known_L(L_ref, n);
    form_A(A, L_ref, n);
    memcpy(fact, A, (size_t)n*n*sizeof(double));
    double t = mphil_dis_cholesky(fact, n);

    char pfx[64], lbl[128];
    snprintf(pfx, sizeof(pfx), "n=%d", n);
    snprintf(lbl, sizeof(lbl), "elapsed>=0 n=%d", n);
    check(lbl, t >= 0.0);
    check_all_metrics(pfx, A, fact, L_ref, n, 0, 0.0);

    free(L_ref); free(A); free(fact);
}

/* ================================================================== */
/* Test 6 — numerically stressed diagonal SPD                          */
/* ================================================================== */
static void test_stressed(int n)
{
    printf("\n=== Test 6: stressed diagonal  n=%d  diag 1..1e-12 ===\n", n);

    double *L_ref = malloc((size_t)n*n*sizeof(double));
    double *A     = malloc((size_t)n*n*sizeof(double));
    double *fact  = malloc((size_t)n*n*sizeof(double));
    if (!L_ref || !A || !fact) {
        printf("  SKIP n=%d (malloc failed)\n", n);
        free(L_ref); free(A); free(fact); return;
    }

    fill_stressed_L(L_ref, n);
    form_A(A, L_ref, n);           /* diagonal: A[i,i] = L_ref[i,i]^2 */
    memcpy(fact, A, (size_t)n*n*sizeof(double));
    mphil_dis_cholesky(fact, n);

    char pfx[64];
    snprintf(pfx, sizeof(pfx), "stressed n=%d", n);
    /* logdet_ref derived automatically from L_ref diagonal by check_all_metrics */
    check_all_metrics(pfx, A, fact, L_ref, n, 0, 0.0);

    free(L_ref); free(A); free(fact);
}

/* ================================================================== */
/* Test 7 — corr() reconstruction + external logdet references         */
/* ================================================================== */
static const struct { int n; double logdet_ref; } CORR_LOGDET[] = {
    {  50, -196.1047097521 },
    { 200, -877.1028093966 },
};
static const int N_CORR = (int)(sizeof(CORR_LOGDET)/sizeof(CORR_LOGDET[0]));

static void test_corr(int n)
{
    printf("\n=== Test 7: corr()  n=%d ===\n", n);

    double *orig = malloc((size_t)n*n*sizeof(double));
    double *fact = malloc((size_t)n*n*sizeof(double));
    if (!orig || !fact) {
        printf("  SKIP n=%d (malloc failed)\n", n);
        free(orig); free(fact); return;
    }

    fill_corr(orig, n);
    memcpy(fact, orig, (size_t)n*n*sizeof(double));
    double t = mphil_dis_cholesky(fact, n);

    char lbl[64], pfx[64];
    snprintf(lbl, sizeof(lbl), "elapsed>=0 n=%d", n);
    check(lbl, t >= 0.0);

    int    have_ref   = 0;
    double logdet_ref = 0.0;
    for (int r = 0; r < N_CORR; r++)
        if (CORR_LOGDET[r].n == n) { logdet_ref = CORR_LOGDET[r].logdet_ref; have_ref = 1; break; }

    snprintf(pfx, sizeof(pfx), "corr n=%d", n);
    check_all_metrics(pfx, orig, fact, NULL, n, have_ref, logdet_ref);

    free(orig); free(fact);
}

/* ================================================================== */
/* Test 8 — multi-thread agreement                                     */
/* ================================================================== */
#ifdef _OPENMP
/*
 * Run at `nthreads` threads and compare the lower triangle of the result
 * against `ref_fact` (computed at 1 thread).  Also call check_all_metrics
 * so every thread count gets a full residual report.
 */
static void check_one_thread_count(int n, int nthreads,
                                   const double *A_orig, const double *ref_fact)
{
    double *work = malloc((size_t)n*n*sizeof(double));
    if (!work) {
        printf("  SKIP n=%d nt=%d (malloc failed)\n", n, nthreads);
        return;
    }

    memcpy(work, A_orig, (size_t)n*n*sizeof(double));
    omp_set_num_threads(nthreads);
    mphil_dis_cholesky(work, n);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            double d = fabs(work[(size_t)i*n+j] - ref_fact[(size_t)i*n+j]);
            if (d > max_diff) max_diff = d;
        }

    char lbl[256], pfx[64];
    snprintf(lbl, sizeof(lbl),
             "n=%d  nt=%d vs nt=1  max|diff|=%.3e (tol=%.0e)",
             n, nthreads, max_diff, TOL_THREAD);
    check(lbl, max_diff < TOL_THREAD);

    snprintf(pfx, sizeof(pfx), "n=%d nt=%d", n, nthreads);
    check_all_metrics(pfx, A_orig, work, NULL, n, 0, 0.0);

    free(work);
}

static void run_multithread_tests(void)
{
    int sizes[]        = {96, 200, 500};
    int thread_counts[] = {2, 4, 8, 76};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    int nc = (int)(sizeof(thread_counts)/sizeof(thread_counts[0]));

    for (int s = 0; s < ns; s++) {
        int n = sizes[s];
        printf("\n=== Test 8: multi-thread  n=%d ===\n", n);

        double *L_ref = malloc((size_t)n*n*sizeof(double));
        double *A     = malloc((size_t)n*n*sizeof(double));
        double *ref   = malloc((size_t)n*n*sizeof(double));
        if (!L_ref || !A || !ref) {
            printf("  SKIP n=%d (malloc failed)\n", n);
            free(L_ref); free(A); free(ref); continue;
        }

        fill_known_L(L_ref, n);
        form_A(A, L_ref, n);
        free(L_ref);

        /* 1-thread reference */
        memcpy(ref, A, (size_t)n*n*sizeof(double));
        omp_set_num_threads(1);
        mphil_dis_cholesky(ref, n);

        for (int t = 0; t < nc; t++)
            check_one_thread_count(n, thread_counts[t], A, ref);

        free(A); free(ref);
    }
}
#endif /* _OPENMP */

/* ================================================================== */
/* main                                                                 */
/* ================================================================== */
int main(void)
{
    printf("=== mphil_dis_cholesky correctness tests ===\n");
    printf("    NB=%d\n", BLOCK_NB);
    printf("    TOL_EXACT=%.0e  TOL_KNOWN_L=%.0e  TOL_RECON=%.0e"
           "  TOL_REL_RECON=%.0e  TOL_SYM=%.0e  TOL_LOGDET=%.0e\n\n",
           TOL_EXACT, TOL_KNOWN_L, TOL_RECON, TOL_REL_RECON, TOL_SYM, TOL_LOGDET);

    /* --- Tests 1-4: fixed small cases and bounds ------------------- */
    test_2x2();
    test_3x3();
    test_n1();
    test_bounds();

    /* --- Test 5: known-L at block-boundary sizes ------------------- */
    int known_sizes[] = {5, 95, 96, 97, 191, 192, 193, 255, 256, 257};
    for (int s = 0; s < (int)(sizeof(known_sizes)/sizeof(known_sizes[0])); s++)
        test_known_L(known_sizes[s]);

    /* --- Test 6: numerically stressed SPD -------------------------- */
    test_stressed(32);
    test_stressed(96);

    /* --- Test 7: corr() reconstruction ----------------------------- */
    test_corr(50);
    test_corr(200);
    test_corr(500);

    /* --- Test 8: multi-thread agreement ---------------------------- */
#ifdef _OPENMP
    run_multithread_tests();
#else
    printf("\n=== Test 8: multi-thread SKIPPED (no OpenMP) ===\n");
#endif

    printf("\n--- %d / %d tests passed ---\n",
           tests_run - tests_failed, tests_run);
    return (tests_failed == 0) ? 0 : 1;
}
