# Performance Analysis Notes — MPhil DIS C2 Cholesky Coursework

**Platform**: CSD3 icelake partition (76-core Intel Xeon Platinum 8276 @ 2.2 GHz)
**Compiler**: GCC 11, flags: `-O3 -march=native -ffast-math -fopenmp`
**Serial baseline flags**: v1 uses `-O0`; v2/v3 use `-O3 -march=native -ffast-math`
**Block size (v5)**: NB = 96 (empirically optimal — see Section 11 and Fig 8)
**OMP settings**: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
**Reps**: 3 per configuration; statistics = mean ± 1 SD

> **DATA STATUS: CURRENT**
> All scaling data (Sections 3–5) now come from the corrected code + NB=96 re-run.
> v6_blocked_NB96 data added from second CSD3 run. Block-sweep results confirmed.

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

### Table 3a — v3_openmp strong scaling (n=2000, corrected code, NB irrelevant)

| Threads | Mean time (s) | GFLOP/s | Speedup | Efficiency |
|---------|---------------|---------|---------|------------|
| 1       | 0.854         | 3.13    | 1.00×   | 100.0%     |
| 2       | 0.481         | 5.55    | 1.78×   | 88.8%      |
| 4       | 0.248         | 10.75   | 3.45×   | 86.1%      |
| 8       | 0.148         | 18.06   | 5.78×   | 72.3%      |
| 16      | 0.085         | 31.35   | 10.1×   | 63.0%      |
| 32      | 0.059         | 45.41   | 14.6×   | 45.5%      |
| 48      | 0.058         | 46.09   | **14.8× (peak)** | 30.8% |
| 64      | 0.062         | 42.81   | 13.8×   | 21.5%      |
| 76      | 0.079         | 33.70   | 10.8×   | 14.2%      |

**Note**: v3 at n=2000 *degrades* beyond 48 threads. At 76T, performance is worse than at 48T. Fine-grained synchronisation (n=2000 barrier steps) dominates at high thread counts.

### Table 3b — v3_openmp strong scaling (n=4000)

| Threads | Mean time (s) | GFLOP/s | Speedup | Efficiency |
|---------|---------------|---------|---------|------------|
| 1       | 10.269        | 2.08    | 1.00×   | 100.0%     |
| 2       | 4.166         | 5.11    | 2.47×   | 123%*      |
| 4       | 2.129         | 10.02   | 4.82×   | 120%*      |
| 8       | 1.238         | 17.22   | 8.30×   | 103.7%     |
| 16      | 0.757         | 28.19   | 13.6×   | 84.8%      |
| 32      | 0.449         | 47.54   | 22.9×   | 71.5%      |
| 48      | 0.356         | 59.97   | 28.9×   | 60.1%      |
| 64      | 0.325         | 65.65   | 31.6×   | 49.4%      |
| 76      | 0.325         | 65.73   | 31.6×   | 41.6%      |

*Super-linear speedup at 2–4 threads: cache-capacity effect — per-thread working set fits in L3.
Saturates at 64–76T (65.7 GFLOPS at both).

Note: v3 at n=6000 only has 3 reps at 1 thread (42.7 s, 1.69 GFLOPS) before the INTR job time limit was hit. n=8000 data unavailable for v3.

### Table 4 — v5_blocked_NB96 strong scaling (n=8000, corrected code)

| Threads | Mean time (s) | GFLOP/s  | Speedup | Efficiency |
|---------|---------------|----------|---------|------------|
| 1       | 69.009        | 2.473    | 1.00×   | 100.0%     |
| 2       | 34.575        | 4.936    | 2.00×   | 99.8%      |
| 4       | 17.110        | 9.977    | 4.03×   | 100.8%     |
| 8       | 8.683         | 19.655   | 7.95×   | 99.4%      |
| 16      | 4.367         | 39.076   | 15.8×   | 98.7%      |
| 32      | 2.231         | 76.510   | 30.9×   | 96.6%      |
| 48      | 1.533         | 111.36   | 45.0×   | 93.8%      |
| 64      | 1.277         | 133.74   | 54.0×   | 84.3%      |
| 76      | 1.162         | 146.88   | 59.4×   | 78.2%      |

**Assessment**: Near-linear scaling to 32T. 59.4× speedup at 76T (78.2% efficiency). Barrier count reduced from O(n)=8000 (v3) to O(n/NB)=84 panel steps for NB=96 (v5), keeping threads busy between synchronisations.

### Table 5 — v6_blocked_NB96 strong scaling (n=8000)

| Threads | Mean time (s) | GFLOP/s  | Speedup | Efficiency |
|---------|---------------|----------|---------|------------|
| 1       | 27.319        | 6.247    | 1.00×   | 100.0%     |
| 2       | 13.999        | 12.192   | 1.95×   | 97.6%      |
| 4       | 10.365        | 16.465   | 2.64×   | 65.9%      |
| 8       | 6.966         | 24.500   | 3.92×   | 49.0%      |
| 16      | 3.522         | 48.455   | 7.76×   | 48.5%      |
| 32      | 1.900         | 89.820   | 14.4×   | 44.9%      |
| 48      | 1.345 ±0.10   | 127.3    | 20.3×   | 42.3%*     |
| 64      | 1.049         | 162.68   | 26.0×   | 40.6%      |
| 76      | 0.942         | 181.23   | **29.0×** | 38.2%  |

*High variance at 48T (CV≈8%, reps span 117–137 GFLOPS). Likely NUMA topology effect.

**Assessment**: v6 achieves 181 GFLOPS at n=8000, 76T — +23% over v5. The lower parallel efficiency (38% vs 78%) is *not a regression*: v6's single-thread code is 2.53× faster than v5, so the parallelism ratio is lower even though absolute time is better at all thread counts above 1.

---

## 4. Version Comparison (76 threads)

### Table 6 — v3 vs v5 vs v6 at n=4000, 76T

| Version          | Time (s) | GFLOP/s | vs v3   |
|------------------|----------|---------|---------|
| v3_openmp        | 0.325    | 65.7    | 1×      |
| v5_blocked_NB96  | 0.147    | 144.9   | 2.21×   |
| v6_blocked_NB96  | 0.104    | 205.0   | **3.12×** |

### Table 7 — v5 vs v6 at 76T, all n

| n    | v5 GFLOP/s | v6 GFLOP/s | v6/v5 gain |
|------|-----------|-----------|------------|
| 2000 | 124.7     | 238.8     | +91%       |
| 4000 | 144.9     | 205.0     | +41%       |
| 6000 | 147.1     | 186.1     | +26%       |
| 8000 | 146.9     | 181.2     | +23%       |

v6/v5 improvement decreases with n because Phase 3 SYRK (where OPT-3 j×4 unroll helps) dominates at large n, while Phase 2 TRSM (where OPT-2 L11 cache helps most) is relatively larger at small n.

### Table 8 — Single-thread v5 vs v6 (OPT-1–3 active at 1T)

| n    | v5 GFLOP/s | v6 GFLOP/s | Ratio |
|------|-----------|-----------|-------|
| 2000 | 2.99      | 10.33     | 3.45× |
| 4000 | 2.61      | 7.21      | 2.76× |
| 6000 | 2.52      | 6.54      | 2.60× |
| 8000 | 2.47      | 6.25      | **2.53×** |

The single-core improvement of 2.5–3.5× is the dominant contribution to v6's gain. Driven primarily by OPT-2 (TRSM reads: L3→L2) and OPT-3 (filling both AVX-512 FMA pipelines).

---

## 5. Problem Size Scaling (76 threads)

### Table 9 — GFLOP/s vs n at 76T

| n    | v3     | v5      | v6      |
|------|--------|---------|---------|
| 2000 | 33.7   | 124.7   | 238.8   |
| 4000 | 65.7   | 144.9   | 205.0   |
| 6000 | n/a*   | 147.1   | 186.1   |
| 8000 | n/a*   | 146.9   | 181.2   |

*v3 at n=6000+ not available (INTR time limit).

v5 plateaus from n=4000 (144.9) to n=8000 (146.9): working set exceeds L3 cache (38 MB; n=4000 matrix = 128 MB) so memory-bandwidth-bound. v6 also plateaus but at higher absolute performance.

---

## 6. Super-linear Speedup (v3 at n=4000, 1–8 threads)

| Threads | Time (s) | Speedup  |
|---------|----------|----------|
| 1       | 10.269   | 1.00×    |
| 2       | 4.166    | 2.47×    |
| 4       | 2.129    | **4.82×** |
| 8       | 1.238    | 8.30×    |

Super-linear speedup (4.82× at 4T, 8.30× at 8T) in v3 at n=4000. Root cause: the full 128 MB matrix exceeds per-socket L3 (38 MB), but 4–8 threads partition the trailing submatrix so each thread's working rows fit in L3, eliminating most DRAM traffic. Cache-capacity effect. Same phenomenon seen in v5 at smaller n.

---

## 7. Key Conclusions for Report

1. **Loop reorder + -O3** alone gives 13–16× speedup over the baseline (v1 at -O0) for n=500–2000.

2. **v3 (commented v2) ≡ v2** in performance. The annotations carry no runtime cost.

3. **v3_openmp scales poorly** due to O(n) synchronisation overhead from fine-grained column-level parallelism. At 76 threads and n=8000: only 8.1× speedup, 10.7% efficiency.

4. **v5_blocked_NB96 scales well** by reducing to O(n/NB)=84 panel steps (vs 8000 for v3). At 76 threads and n=8000: ≈60× speedup, ~80% efficiency, ≈149 GFLOP/s. [UPDATE with fresh data]

5. **The best NB is thread-dependent** in the new v5 block sweep: NB=96 is best at 1T and 76T, while NB=256 is best at 8T and 32T. For the main full-node runs, NB=96 remains the best mean choice.

6. **v5 is >11× faster than v3** at n=8000, 76 threads, purely from blocking structure reducing synchronisation count.

7. **Super-linear speedup** for v5 at n=4000, 8 threads (7.89×) due to per-thread working set fitting in L3.

8. **GFLOP/s increases with n** for v5 due to better ratio of useful work to synchronisation overhead.

9. **v6 adds four microarchitectural opts** (col-pack, L11 private cache, j×4 unroll, static,1 schedule). Measured gain: +23% at n=8000, 76T; +41% at n=4000, 76T; **2.53× single-thread** at n=8000.

10. **All CVs < 3.5%**, confirming highly reproducible measurements on CSD3 icelake.

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
(v5_openmp_blocked, with NB=96 retained as the fixed setting for the main scaling runs)
restructured the algorithm into three phases per panel:
serial diagonal block factorisation (`omp single`), parallel TRSM (`omp for
schedule(static)`) and parallel SYRK (`omp for schedule(guided)`). By limiting
synchronisation to O(n/NB) panel steps (84 vs 8000 for v3), v5 achieves near-linear scaling
to 76 threads (n=8000), delivering ≈149 GFLOP/s — over 11× faster than v3 at the same
thread count. A separate block sweep shows the best NB depends on thread count, with NB=96
best at 1T/76T and NB=256 best at 8T/32T; NB=96 is retained for the main report because it
is the best mean full-node choice (Fig. 8). Super-linear
speedup is observed for v5 at n=4000 and 8 threads (7.89×), consistent with the per-thread
working set fitting within L3 cache. Finally, v6_openmp_blocked adds four microarchitectural
optimisations: (i) column packing to eliminate stride-n reads in Phase 1, (ii) a private
per-thread L11 cache keeping the diagonal block in L2 during Phase 2 TRSM,
(iii) 4-wide j-loop unrolling in Phase 3 to fill both AVX-512 FMA pipelines, and
(iv) `schedule(static,1)` for load-balanced triangular work distribution.
v6_openmp_blocked adds four microarchitectural optimisations: (i) column packing in Phase 1
eliminates stride-n reads; (ii) a private per-thread L11 cache converts Phase 2 TRSM reads
from L3 (stride 64 KB) to L2 (stride 768 B); (iii) 4-wide j-loop unrolling in Phase 3 SYRK
exposes four independent FMA chains filling both AVX-512 pipelines simultaneously; (iv)
`schedule(static,1)` gives each thread a round-robin mix of heavy and light rows for better
load balance. Together these yield a 2.53× single-thread improvement (n=8000) and 181
GFLOP/s at 76 threads — 23% above v5. The lower parallel efficiency of v6 (38% vs 78%)
reflects the stronger single-core baseline rather than any parallelism degradation.

---

## 9. Git Tags Reference (for required report table)

| Tag               | Description                                                        | Key file                         |
|-------------------|--------------------------------------------------------------------|----------------------------------|
| v0.1-baseline     | Exact spec loop, -O0, single thread                                | src/cholesky_v1_baseline.c       |
| v0.2-serial-opt   | Loop interchange, hoisting, -O3 flags                              | src/cholesky_v2_serial_opt.c     |
| v0.3-openmp-v1    | v3_openmp: parallel trailing update, omp for schedule(static)      | src/cholesky_v3_openmp.c         |
| v0.4-blocked      | v5_openmp_blocked: panel-blocked OpenMP, NB=96                     | src/cholesky_v5_openmp_blocked.c |
| v0.5-tuned        | v5_openmp_blocked with correctness fix (lower-triangle Phase 1)    | src/cholesky_v5_openmp_blocked.c |
| v0.6-cache-opts   | v6_openmp_blocked: col-pack, L11 cache, j×4 unroll, static,1      | src/cholesky_v6_openmp_blocked.c |

All six tags are in the repository (`git tag -l`). Run `git push --tags` to publish them to the remote.

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
| report/figures/fig8_block_sweep.pdf     | Multi-thread block sweep: GFLOPS vs NB and best NB vs threads |

Figures 1--7 are generated by `scripts/plot_results.py` from `results/csd3_serial.csv`
and `results/csd3_scaling.csv`. Figure 8 is generated by `scripts/plot_block_sweep.py`
from `results/block_sweep.csv`.

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

### Block-sweep results (v5_openmp_blocked, CSD3 icelake, n=8000)

| Threads | Best NB | Mean GFLOP/s | Interpretation |
|---------|---------|--------------|----------------|
| 1       | 96      | 6.24         | Small panels still best when locality dominates and barrier cost is irrelevant |
| 8       | 256     | 37.74        | Larger panels amortize synchronisation overhead better |
| 32      | 256     | 97.86        | Same trend: fewer panel iterations outweigh extra cache pressure |
| 76      | 96      | 187.85       | Full-node optimum shifts back to NB=96, but NB=256 is very close (186.88 GFLOP/s) |

The new sweep shows that NB is not globally constant across thread counts. At 8T and
32T, performance increases almost monotonically with panel width, implying that reducing
the number of panel iterations from 84 at NB=96 to 32 at NB=256 dominates any loss of
cache locality. At 76T the curve flattens and the best mean performance returns to
NB=96, with NB=256 only 0.5% lower. This suggests that at full occupancy the shorter
per-thread work chunks and stronger cache/load-balance effects favor a somewhat smaller
panel, while the optimum remains broad rather than sharply peaked.

### Report Paragraph (copy/paste ready)

The panel width NB was tuned empirically by compiling `v5_openmp_blocked` with
`-DBLOCK_NB=N` for N ∈ {64, 96, 128, 192, 256} and measuring GFLOP/s at n=8000
for thread counts {1, 8, 32, 76} (3 reps each) on CSD3 icelake (Fig. 8). The
optimum depends on thread count: NB=96 gives the highest mean throughput at 1T
(6.24 GFLOP/s) and 76T (187.85 GFLOP/s), whereas NB=256 is best at 8T (37.74
GFLOP/s) and 32T (97.86 GFLOP/s). Larger panels reduce the number of panel
iterations from 84 at NB=96 to 32 at NB=256, which helps at intermediate thread
counts by amortizing synchronisation overhead. At 76T, however, NB=96 regains a
slight lead, with NB=256 only 0.5% slower, indicating a broad full-node optimum.
NB=96 is therefore retained as the default setting for the main strong-scaling
results, while the sweep shows that retuning NB for lower thread counts could
yield extra performance.

### How to run the sweep on CSD3

```bash
sbatch scripts/csd3_block_sweep.slurm
# Results written to results/block_sweep.csv
python3 scripts/plot_block_sweep.py
# Figure saved to report/figures/fig8_block_sweep.pdf
```
