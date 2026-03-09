#include "mphil_dis_cholesky.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>   /* OpenMP runtime header */

#define MAX_N 100000

/*
 * v4_tuned: task-based OpenMP parallelisation with explicit data dependencies.
 *
 * Why v3_openmp failed to scale
 * ─────────────────────────────
 * v3_openmp places a global barrier after every pivot step (once via
 * omp single, once via omp for).  For n=2000 that is 4000 barriers;
 * each costs ~500 µs with 76 threads → ~2 s of pure synchronisation
 * overhead regardless of matrix size or thread count.
 *
 * The task-based approach (inspired by Liu et al. 2023, arXiv:2305.04635)
 * ────────────────────────────────────────────────────────────────────────
 * Each unit of work is submitted as an OpenMP task annotated with
 * depend() clauses.  The runtime builds a DAG and schedules tasks
 * without any global barriers.
 *
 * Key insight: pivot(p+1) only needs c[(p+1)*(p+1)] to be correct,
 * which is written by row_update(p, p+1).  It does NOT need all of
 * row_update(p, i) to finish.  This creates a genuine pipeline:
 *
 *   pivot(0)
 *   ├─ row_update(0,1) ──► pivot(1)
 *   │                       ├─ row_update(1,2) ──► pivot(2) ──► ...
 *   │                       └─ row_update(1,3)
 *   ├─ row_update(0,2) ──► row_update(1,2) (see above)
 *   └─ row_update(0,3) ──► row_update(1,3) (see above)
 *
 * Dependency tokens
 * ─────────────────
 * We reuse existing matrix elements as dependency tokens (no extra memory):
 *   c[p*n + p]   diagonal after pivot(p);  row_update(p,*) depends on it
 *   c[i*n + p]   column element written by row_update(p,i);
 *                row_update(p+1,i) and pivot(p+1) (i=p+1) depend on it
 *
 * Column normalisation is moved INSIDE each row_update task.
 * If it stayed in the pivot task, pivot(p) would have to wait for ALL
 * row_update(p-1,i) to finish — recreating a full barrier every step.
 */

double mphil_dis_cholesky(double *c, int n)
{
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "mphil_dis_cholesky: n=%d out of range [1, %d]\n", n, MAX_N);
        return -1.0;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Open the thread pool once.  One thread submits all tasks;
     * all threads (including the submitter) steal and execute them. */
    #pragma omp parallel
    /* omp single: exactly one thread runs the task-submission loop.
     * nowait: the submitter does not wait at the end of the single region —
     * it immediately starts stealing tasks alongside the other threads. */
    #pragma omp single nowait
    {
        /* ── Step p=0: no prior dependencies ────────────────────────── */
        double *row_0 = &c[0];

        /* Pivot task for p=0.
         * depend(inout: c[0]): marks the diagonal as written so that
         * row_update(0,i) tasks know when it is safe to read it. */
        #pragma omp task shared(c) firstprivate(row_0, n) \
                         depend(inout: c[0])
        {
            double diag    = sqrt(row_0[0]);
            row_0[0]       = diag;
            double inv_diag = 1.0 / diag;
            /* Normalise row 0 — no column normalisation here (see below). */
            for (int j = 1; j < n; j++)
                row_0[j] *= inv_diag;
        }

        /* Row-update tasks for p=0.
         * depend(in:  c[0])    : wait for pivot(0) to store diag in c[0].
         * depend(inout: c[i*n]): this task writes c[i*n+0..n-1];
         *   the inout on c[i*n+0] signals row_update(1,i) to wait for us. */
        for (int i = 1; i < n; i++) {
            #pragma omp task shared(c) firstprivate(i, row_0, n) \
                             depend(in: c[0]) depend(inout: c[i*n])
            {
                /* Column normalisation moved here from pivot — avoids pivot(1)
                 * depending on every row_update(0,i) to finish. */
                double inv_diag = 1.0 / c[0];   /* c[0] = diag stored by pivot(0) */
                c[i*n] *= inv_diag;
                double c_ip  = c[i*n];
                double *row_i = &c[i*n];
                for (int j = 1; j < n; j++)
                    row_i[j] -= c_ip * row_0[j];
            }
        }

        /* ── Steps p = 1 … n-1 ─────────────────────────────────────── */
        for (int p = 1; p < n; p++) {
            double *row_p = &c[p*n];

            /* Pivot task for step p.
             * depend(in:  c[p*n + p-1]): wait for row_update(p-1, p) —
             *   that task writes c[p*n+p-1] as its column-normalisation step,
             *   and also updates c[p*n+p] (the diagonal we need to sqrt).
             *   This is the ONLY previous task we must wait for; all others
             *   updating different rows can overlap with us.
             * depend(inout: c[p*n + p]): marks the diagonal as written. */
            #pragma omp task shared(c) firstprivate(p, row_p, n) \
                             depend(in: c[p*n + p-1]) depend(inout: c[p*n + p])
            {
                double diag    = sqrt(row_p[p]);
                row_p[p]       = diag;
                double inv_diag = 1.0 / diag;
                for (int j = p+1; j < n; j++)
                    row_p[j] *= inv_diag;
            }

            /* Row-update tasks for step p.
             * depend(in: c[p*n+p]):     wait for pivot(p).
             * depend(in: c[i*n + p-1]): wait for row_update(p-1, i) —
             *   ensures row i has been updated through step p-1 before we
             *   overwrite it in step p.  This is what creates the wavefront.
             * depend(inout: c[i*n + p]): signals the next step and pivot. */
            for (int i = p+1; i < n; i++) {
                #pragma omp task shared(c) firstprivate(p, i, row_p, n) \
                                 depend(in: c[p*n + p]) \
                                 depend(in: c[i*n + p-1]) \
                                 depend(inout: c[i*n + p])
                {
                    double inv_diag = 1.0 / c[p*n + p]; /* diag from pivot(p) */
                    c[i*n + p] *= inv_diag;              /* column normalisation */
                    double c_ip   = c[i*n + p];
                    double *row_i = &c[i*n];
                    for (int j = p+1; j < n; j++)
                        row_i[j] -= c_ip * row_p[j];
                }
            }
        }

        /* taskwait: block until every task in this task region has completed
         * before we measure the stop time and return. */
        #pragma omp taskwait
    }
    /* End of parallel region: all threads join here */

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    return (t_end.tv_sec  - t_start.tv_sec)
         + (t_end.tv_nsec - t_start.tv_nsec) * 1.0e-9;
}
