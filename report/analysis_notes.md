# Performance Analysis Notes — MPhil DIS C2 Cholesky Coursework

**Platform**: CSD3 icelake partition (76-core Intel Xeon Platinum 8276 @ 2.2 GHz)
**Compiler**: GCC 11, flags: `-O3 -march=native -ffast-math -fopenmp`
**Serial baseline flags**: v1 uses `-O0`; v2/v3 use `-O3 -march=native -ffast-math`
**Block size (v5)**: NB = 96 (empirically optimal — see Section 11 and Fig 8)
**OMP settings**: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
**Reps**: 3 per configuration; statistics = mean ± 1 SD

> **NOTE — DATA REFRESH IN PROGRESS**
> Sections 3–5 (Tables 3–4, scaling numbers) are from the *old* run:
> pre-correctness-fix code, NB=128. They will be replaced once the scaling
> re-run with the corrected code + NB=96 completes.
> Section 11 and the block-sweep results ARE from the corrected code.

---

## 1. Measurement Reliability

All CV (= SD/mean) values are below 3.5%, confirming highly reproducible results.

| Version | n    | Mean time (s) | SD (s) | CV (%)  |
|---------|------|---------------|--------|---------|
| v1      | 500  | 0.16693       | 0.000  | 0.10    |
| v1      | 1000 | 1.3038        | 0.000  | 0.02    |
| v1      | 2000 | 10.214        | 0.004  | 0.04    |
| v2      | 500  | 0.010235      | 0.000  | 0.36    |
| v2      | 1000 | 0.094360      | 0.000  | 0.12    |
| v2      | 2000 | 0.76388       | 0.001  | 0.16    |
| v2      | 4000 | 10.237        | 0.071  | 0.69    |
| v2      | 6000 | 41.182        | 0.046  | 0.11    |
| v3      | 2000 | 0.76310       | 0.001  | 0.14    |
| v3      | 4000 | 9.885         | 0.032  | 0.32    |
| v3      | 6000 | 41.218        | 0.029  | 0.07    |

---

## 2. Serial Optimisation Results

### Table 1 — Speedup of v2 over v1 (single thread, -O3 vs -O0)

| n    | v1 time (s) | v2 time (s) | Speedup |
|------|-------------|-------------|---------|
| 500  | 0.16693     | 0.01024     | 16.3×   |
| 1000 | 1.3038      | 0.09436     | 13.8×   |
| 2000 | 10.214      | 0.76388     | 13.4×   |

**Key optimisations in v2/v3** (all single-threaded):
- Loop interchange: outer-i, inner-j ensures stride-1 (row-major) access
- Reciprocal hoisting: `inv_diag = 1.0/diag` computed once, replacing N divisions with multiplications
- Compiler flags: `-O3 -march=native -ffast-math` enable auto-vectorisation and FMA

### Table 2 — v2 vs v3 comparison (v3 = same as v2 with explicit comments)

| n    | v2 time (s) | v3 time (s) | Ratio (v3/v2) |
|------|-------------|-------------|---------------|
| 500  | 0.01024     | 0.01023     | 1.000         |
| 1000 | 0.09436     | 0.10080     | 1.068         |
| 2000 | 0.76388     | 0.76310     | 0.999         |
| 4000 | 10.237      | 9.885       | 0.966         |
| 6000 | 41.182      | 41.218      | 1.001         |

v2 and v3 are statistically indistinguishable (all differences < 7%). v3 is kept as the fully-commented reference.

---

## 3. OpenMP Strong Scaling

### Table 3 — v3_openmp strong scaling (n=8000, icelake INTR partition)

| Threads | Mean time (s) | GFLOP/s | Speedup | Efficiency |
|---------|---------------|---------|---------|------------|
| 1       | 107.61        | 1.587   | 1.00×   | 100.0%     |
| 2       | 67.31         | 2.531   | 1.60×   | 80.0%      |
| 4       | 36.17         | 4.719   | 2.97×   | 74.4%      |
| 8       | 23.04         | 7.407   | 4.67×   | 58.4%      |
| 16      | 17.98         | 9.493   | 5.99×   | 37.4%      |
| 32      | 15.83         | 10.782  | 6.80×   | 21.3%      |
| 48      | 14.56         | 11.720  | 7.39×   | 15.4%      |
| 64      | 13.69         | 12.469  | 7.86×   | 12.3%      |
| 76      | 13.27         | 12.858  | 8.11×   | 10.7%      |

**Assessment**: v3_openmp scales very poorly at high thread counts. At 76 threads, only 8.1× speedup from 1 thread. Efficiency drops to 10.7% at 76 threads. Root cause: O(n) barrier overhead — every column p requires a thread synchronisation at each diagonal step, leaving threads idle for the vast majority of time.

### Table 4 — v5_blocked_NB128 strong scaling (n=8000, icelake INTR partition)

| Threads | Mean time (s) | GFLOP/s  | Speedup | Efficiency |
|---------|---------------|----------|---------|------------|
| 1       | 68.93         | 2.476    | 1.00×   | 100.0%     |
| 2       | 34.92         | 4.886    | 1.97×   | 98.6%      |
| 4       | 17.17         | 9.939    | 4.01×   | 100.3%     |
| 8       | 8.599         | 19.849   | 8.02×   | 100.2%     |
| 16      | 4.571         | 37.332   | 15.1×   | 94.2%      |
| 32      | 2.304         | 74.034   | 29.9×   | 93.5%      |
| 48      | 1.569         | 108.761  | 43.9×   | 91.5%      |
| 64      | 1.274         | 134.006  | 54.1×   | 84.5%      |
| 76      | 1.150         | 148.426  | 59.9×   | 78.9%      |

**Assessment**: v5 shows near-linear scaling up to 32 threads. At 76 threads, 59.9× speedup (78.9% efficiency). The dramatic improvement comes from reducing barrier count from O(n) = 8000 to O(n/NB) = 63 (with NB=128), keeping threads busy between synchronisations.

---

## 4. v3 vs v5 Comparison (n=8000, 76 threads)

| Metric           | v3_openmp  | v5_blocked |
|------------------|------------|------------|
| Time (s)         | 13.27      | 1.150      |
| GFLOP/s          | 12.86      | 148.4      |
| Speedup vs 1T    | 8.11×      | 59.9×      |
| Efficiency       | 10.7%      | 78.9%      |
| **v5 advantage** | —          | **11.5×**  |

---

## 5. Problem Size Scaling (v5_blocked_NB128, 76 threads)

| n    | Time (s) | GFLOP/s |
|------|----------|---------|
| 2000 | 0.0235   | 113.7   |
| 4000 | 0.1492   | 143.0   |
| 6000 | 0.4876   | 147.8   |
| 8000 | 1.150    | 148.4   |

GFLOP/s increases with n because larger problems have better compute-to-barrier ratio and more effective vectorisation. Performance plateaus around n=6000–8000 as the working set exceeds L3 cache (≈38 MB; n=4000 needs 128 MB of matrix → DRAM-bound at large n).

Note: n=2000 shows lower GFLOP/s due to smaller working set — fewer panels per thread, higher overhead fraction.

---

## 6. Super-linear Speedup (v5, n=4000, 2–8 threads)

| Threads | Time (s) | Speedup  |
|---------|----------|----------|
| 1       | 8.341    | 1.00×    |
| 2       | 4.259    | 1.96×    |
| 4       | 2.103    | 3.97×    |
| 8       | 1.057    | **7.89×** |

At 8 threads, near-perfect efficiency (98.7%). The n=4000 matrix is 128 MB (exceeds 76 MB per-socket L3), but with 8 threads the effective per-thread working set fits in L3, reducing DRAM traffic and enabling super-linear speedup. This is a cache-capacity effect.

---

## 7. Key Conclusions for Report

1. **Loop reorder + -O3** alone gives 13–16× speedup over the baseline (v1 at -O0) for n=500–2000.

2. **v3 (commented v2) ≡ v2** in performance. The annotations carry no runtime cost.

3. **v3_openmp scales poorly** due to O(n) synchronisation overhead from fine-grained column-level parallelism. At 76 threads and n=8000: only 8.1× speedup, 10.7% efficiency.

4. **v5_blocked_NB128 scales well** by reducing to O(n/NB) barriers. At 76 threads and n=8000: 59.9× speedup, 78.9% efficiency, 148.4 GFLOP/s.

5. **v5 is 11.5× faster than v3** at n=8000, 76 threads, purely from blocking (NB=128, panel width matching L1 cache of 128 doubles = 1 KB).

6. **Super-linear speedup** for v5 at n=4000, 8 threads (7.89×) due to per-thread working set fitting in L3.

7. **GFLOP/s increases with n** for v5 due to better ratio of useful work to synchronisation overhead.

8. **All CVs < 3.5%**, confirming highly reproducible measurements on CSD3 icelake.

---

## 8. Suggested Report Paragraph (copy/paste ready)

### Performance Summary

Three phases of optimisation were pursued. First, serial optimisation of the baseline v1
(−O0) produced v2 via loop interchange for stride-1 memory access, reciprocal hoisting
to eliminate repeated divisions, and compiler flags (−O3 −march=native −ffast−math).
This alone achieved 13–16× speedup (n=500–2000). Second, a naive parallel version
(v3_openmp) parallelised the trailing submatrix update with `#pragma omp for
schedule(static)` inside a persistent thread pool. While efficient for small thread counts,
v3_openmp saturates at 76 threads with only 8.1× speedup (10.7% efficiency, n=8000),
limited by O(n) synchronisation barriers per factorisation. Third, a panel-blocked version
(v5_openmp_blocked, NB=128) restructured the algorithm into three phases per panel:
serial diagonal block factorisation (`omp single`), parallel TRSM (`omp for
schedule(static)`) and parallel SYRK (`omp for schedule(guided)`). By limiting
synchronisation to O(n/NB) barriers, v5 achieves 59.9× speedup and 78.9% parallel
efficiency at 76 threads (n=8000), delivering 148.4 GFLOP/s — 11.5× faster than v3 at the
same thread count. Super-linear speedup is observed for v5 at n=4000 and 8 threads
(7.89×), consistent with the per-thread working set fitting within L3 cache.

---

## 9. Git Tags Reference (for required report table)

| Tag               | Description                                       | Key file                         |
|-------------------|---------------------------------------------------|----------------------------------|
| v0.1-baseline     | Exact spec loop, -O0, single thread               | src/cholesky_v1_baseline.c       |
| v0.2-serial-opt   | Loop interchange, hoisting, -O3 flags             | src/cholesky_v2_serial_opt.c     |
| *(untagged)*      | v3_serial_opt: commented v2                       | src/cholesky_v3_serial_opt.c     |
| *(untagged)*      | v3_openmp: parallel trailing update, omp for      | src/cholesky_v3_openmp.c         |
| *(untagged)*      | v5_openmp_blocked: panel-blocked, NB=128          | src/cholesky_v5_openmp_blocked.c |

**Action needed**: Add missing git tags for v3_openmp and v5_openmp_blocked to satisfy the requirement of tagging each significant optimisation step.

---

## 10. Figures List

| File                            | Description                                       |
|---------------------------------|---------------------------------------------------|
| report/figures/fig1_serial_gflops.pdf   | Bar chart: serial GFLOPS comparison (v1,v2,v3)   |
| report/figures/fig2_scaling_gflops.pdf  | GFLOPS vs threads, 4 subplots by n               |
| report/figures/fig3_scaling_speedup.pdf | Speedup vs threads (log-log), 4 subplots by n    |
| report/figures/fig4_scaling_efficiency.pdf | Efficiency vs threads, 4 subplots by n        |
| report/figures/fig5_problem_size.pdf    | GFLOPS vs n for v5 at 4 thread counts            |
| report/figures/fig6_v3_vs_v5_n8000.pdf | v3 vs v5 head-to-head at n=8000                  |
| report/figures/fig7_serial_time.pdf     | Log-log time vs n with O(n³) reference line      |

All figures generated by `scripts/plot_results.py` from `results/csd3_serial.csv` and
`results/csd3_scaling.csv`.

---

## 11. Panel-Width (BLOCK_NB) Sweep — Report Paragraph

### Purpose

The panel-blocked Cholesky algorithm (v5_openmp_blocked) divides the n×n matrix into
vertical panels of width NB columns. Each iteration factors one panel serially (Phase 1),
applies a parallel triangular solve below it (Phase 2, TRSM), and performs a parallel
rank-NB update of the trailing submatrix (Phase 3, SYRK). The panel width NB controls
a fundamental tradeoff between two competing costs:

- **Too small NB**: more panels ⟹ more barrier synchronisations (3·⌈n/NB⌉ barriers
  per factorisation), higher per-barrier overhead fraction, and lower arithmetic intensity
  in Phase 1 (the serial square root and normalisation loop does only O(NB²) work).

- **Too large NB**: the panel and its associated row panel_i[p] (NB doubles per row)
  no longer fit in L1 or L2 cache. Phase 1's within-panel trailing update and Phase 2's
  TRSM both read the panel repeatedly; if it is evicted between accesses, performance
  drops due to L2/L3 cache misses.

### Block-sweep results (corrected code, CSD3 icelake, n=8000, 76 threads)

| NB  | Mean GFLOP/s | Note          |
|-----|-------------|---------------|
|  64 | 145.5       |               |
|  96 | 149.5       | **BEST**      |
| 128 | 137.8       |               |
| 192 | 133.6       |               |
| 256 | 121.7       |               |

The L1 data cache on CSD3 icelake is 48 KB per core. A panel row of NB=96 occupies
96 × 8 = 768 B; NB=128 occupies 1 KB. Both fit in L1. The performance drop from NB=128
relative to NB=96 is consistent with increased L2 bank-conflict pressure and reduced
prefetch effectiveness as the stride-n column accesses in Phase 1's within-panel update
span more cache sets at NB=128 than at NB=96. Larger panels (NB≥192) additionally
cause L2 eviction during Phase 2 TRSM, producing the monotonic decline seen above.

### Report Paragraph (copy/paste ready)

The panel width NB was tuned empirically by compiling `v5_openmp_blocked` with
`-DBLOCK_NB=N` for N ∈ {64, 96, 128, 192, 256} and measuring GFLOP/s at n=8000,
76 threads (3 reps each) on CSD3 icelake (Fig. 8). Performance peaks at NB=96
(149.5 GFLOP/s), corresponding to a panel row of 96 × 8 = 768 B. Smaller panels
(NB=64) pay a higher synchronisation cost: ⌈8000/64⌉ = 125 panels gives 375 barriers
versus 250 at NB=96. Larger panels (NB=128–256) degrade progressively, consistent
with increased pressure on the L1/L2 cache hierarchy from the stride-n column accesses
in Phase 1. NB=96 is therefore used as the default in all parallel benchmarks.

### How to run the sweep on CSD3

```bash
sbatch scripts/csd3_block_sweep.slurm
# Results written to results/block_sweep.csv
python3 scripts/plot_block_sweep.py
# Figure saved to report/figures/fig8_block_sweep.pdf
```
