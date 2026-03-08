# Benchmarking Plan — C2 Cholesky Coursework

## Overview

The spec requires: performance data in figures/tables, scaling across matrix sizes
and core counts, code version + compile flags + platform + thread count recorded
for every data point.

This plan covers four experiments, the infrastructure to run them, and the exact
tables and plots needed for the report.

---

## Code Versions to Benchmark

| Git Tag           | What changes                                    | Compile flags                            |
|-------------------|-------------------------------------------------|------------------------------------------|
| v0.1-baseline     | Exact spec loop, no opt                         | `-O0 -g`                                 |
| v0.1-O3           | Same code, compiler optimisations only          | `-O3 -march=native -ffast-math`          |
| v0.2-serial-opt   | Loop reorder + precompute 1/diag                | `-O3 -march=native -ffast-math`          |
| v0.3-openmp-v1    | Basic OpenMP (persistent thread pool)           | `-O3 -march=native -ffast-math -fopenmp` |
| v0.4-sched        | Schedule variants (static vs dynamic vs guided) | `-O3 -march=native -ffast-math -fopenmp` |

Note: v0.1-O3 is NOT a separate git tag — it is v0.1-baseline recompiled with
different flags. Makefile should have a flag variable so this is easy to switch.

---

## Performance Metrics

For every run record:

| Metric     | Formula                        | Unit      |
|------------|--------------------------------|-----------|
| Time       | returned by mphil_dis_cholesky | seconds   |
| GFlop/s    | n³/3 / time / 1e9              | GFlop/s   |
| Speedup    | T(1 thread) / T(N threads)     | unitless  |
| Efficiency | Speedup / N_threads            | % (×100)  |

---

## Experiment 1 — Serial Optimisation Comparison

**Question**: How much do (a) compiler flags and (b) algorithmic changes improve
single-thread performance?

**Platform**: CSD3 icelake, 1 thread, `--exclusive`

**Versions**: v0.1-baseline(-O0), v0.1-baseline(-O3), v0.2-serial-opt(-O3)

**Matrix sizes**: n = 500, 1000, 2000, 4000, 8000

**Repetitions**: 3 runs per (version, n); report minimum time.

**Output CSV** (`results/exp1_serial.csv`):
```
version,flags,n,rep,time_s,gflops
v0.1-baseline,-O0 -g,1000,1,0.823,0.407
...
```

### Table for report (Table 1):
| n    | v0.1 -O0 (s) | v0.1 -O3 (s) | v0.2 -O3 (s) | Speedup v0.2 vs v0.1-O0 |
|------|--------------|--------------|--------------|--------------------------|
| 500  |              |              |              |                          |
| 1000 |              |              |              |                          |
| 2000 |              |              |              |                          |
| 4000 |              |              |              |                          |
| 8000 |              |              |              |                          |

### Plot for report (Plot 1):
- **Type**: log-log line plot
- **x-axis**: n (500 → 8000)
- **y-axis**: time in seconds (log scale)
- **Lines**: one per version (3 lines)
- **Also draw**: reference O(n³) slope line
- **Purpose**: shows absolute speedup from serial opts; slope confirms O(n³)

---

## Experiment 2 — Strong Scaling (Fixed n, Vary Threads)

**Question**: How does adding OpenMP threads speed up the computation?
Strong scaling = fixed problem size, more threads.

**Platform**: CSD3 icelake, `--exclusive`, all 76 cores allocated.

**Versions**: v0.2-serial-opt (as 1-thread baseline), v0.3-openmp-v1, v0.4-sched

**Thread counts**: 1, 2, 4, 8, 16, 32, 48, 64, 76

**Matrix sizes**: n = 2000, 4000, 8000, 12000
(Use multiple sizes to show where parallelism helps most — small n = overhead
dominates; large n = near-ideal scaling)

**Repetitions**: 3 runs per (version, n, threads); report minimum time.

**Output CSV** (`results/exp2_scaling.csv`):
```
version,flags,n,threads,rep,time_s,gflops,speedup,efficiency
v0.3-openmp-v1,-O3 -fopenmp,8000,1,1,34.2,15.8,1.00,1.00
v0.3-openmp-v1,-O3 -fopenmp,8000,4,1,9.1,59.5,3.76,0.94
...
```

### Table for report (Table 2) — at n=8000:
| Threads | Time (s) | GFlop/s | Speedup | Efficiency |
|---------|----------|---------|---------|------------|
| 1       |          |         | 1.00    | 100%       |
| 2       |          |         |         |            |
| 4       |          |         |         |            |
| 8       |          |         |         |            |
| 16      |          |         |         |            |
| 32      |          |         |         |            |
| 48      |          |         |         |            |
| 64      |          |         |         |            |
| 76      |          |         |         |            |

### Plots for report:

**Plot 2a — Speedup vs threads** (strong scaling):
- **Type**: line plot with log x-axis
- **x-axis**: number of threads (1, 2, 4, 8, 16, 32, 48, 64, 76)
- **y-axis**: speedup (T(1)/T(N))
- **Lines**: one per n value (n=2000, 4000, 8000, 12000)
- **Also draw**: ideal linear speedup (dashed)
- **Purpose**: shows parallel efficiency; large n approaches ideal more closely

**Plot 2b — Parallel efficiency vs threads**:
- **Type**: line plot
- **x-axis**: number of threads
- **y-axis**: efficiency = speedup/threads (%)
- **Lines**: one per n value
- **Purpose**: shows where efficiency drops off (Amdahl's law effect)

---

## Experiment 3 — Performance vs Problem Size (Parallel)

**Question**: How does the parallelised code scale with problem size? Does larger n
give better hardware utilisation (GFlop/s)?

**Platform**: CSD3 icelake, `--exclusive`

**Version**: v0.3-openmp-v1 (or best version)

**Matrix sizes**: n = 500, 1000, 2000, 4000, 8000, 12000, 16000

**Thread counts**: 1, 8, 32, 76

**Repetitions**: 3 per (n, threads); report minimum time.

**Output CSV** (`results/exp3_problem_size.csv`):
```
version,flags,n,threads,rep,time_s,gflops
...
```

### Plot for report (Plot 3):
- **Type**: two subplots or one combined
- **Plot 3a**: Time vs n (log-log), lines for 1, 8, 32, 76 threads
- **Plot 3b**: GFlop/s vs n, lines for 1, 8, 32, 76 threads
- **Purpose**: GFlop/s rising with n confirms cache effects; shows benefit of
  parallelism at each problem size

---

## Experiment 4 — OpenMP Schedule Comparison

**Question**: Does changing the OpenMP schedule clause improve performance?
`schedule(static)` vs `schedule(dynamic,chunk)` vs `schedule(guided)`

**Platform**: CSD3 icelake, `--exclusive`

**Version**: v0.4-sched (three sub-variants, each tagged separately or selected
via compile-time flag)

**Matrix sizes**: n = 4000, 8000

**Thread counts**: 16, 32, 76

**Repetitions**: 3 per combination

**Output CSV** (`results/exp4_schedule.csv`):
```
version,schedule,n,threads,rep,time_s,gflops
```

### Table for report (Table 3) — at n=8000, 76 threads:
| Schedule          | Time (s) | GFlop/s | Notes                    |
|-------------------|----------|---------|--------------------------|
| static            |          |         | even chunks, low overhead|
| dynamic,1         |          |         | fine-grain, high overhead|
| dynamic,64        |          |         | coarser chunks           |
| guided            |          |         | decreasing chunk size    |

---

## Infrastructure to Build

### 1. Benchmark program: `test/benchmark.c`

Purpose: runs the factorization N times and outputs one CSV line per run.

```
Usage: ./test/benchmark <n> <nrep>
Output (to stdout, one line per rep):
  version,n,rep,time_s,gflops
```

Prints a header only if environment variable BENCH_HEADER=1.
Matrix is re-filled with corr() before each rep (so we always time the
same computation, not a cached result).

### 2. Sweep shell script: `scripts/run_sweep.sh`

Loops over n values, calls benchmark, appends to CSV.

```bash
#!/bin/bash
# Usage: ./scripts/run_sweep.sh results/output.csv
OUT=$1
echo "version,n,threads,rep,time_s,gflops" > $OUT
for n in 500 1000 2000 4000 8000; do
    BENCH_HEADER=0 ./test/benchmark $n 3 >> $OUT
done
```

### 3. SLURM scripts

#### `scripts/bench_serial.slurm` — Experiment 1
```bash
#!/bin/bash
#SBATCH --job-name=chol_serial
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/serial_%j.out

module load gcc/11

export OMP_NUM_THREADS=1

# Run sweep for each n
for n in 500 1000 2000 4000 8000; do
    ./test/benchmark $n 3
done
```

Run once per version (v0.1-O0, v0.1-O3, v0.2) by recompiling Makefile CFLAGS
between submissions.

#### `scripts/bench_scaling.slurm` — Experiments 2 & 3
```bash
#!/bin/bash
#SBATCH --job-name=chol_scaling
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=results/scaling_%j.out

module load gcc/11

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Experiment 2: strong scaling — vary threads at fixed n
for n in 2000 4000 8000 12000; do
    for threads in 1 2 4 8 16 32 48 64 76; do
        export OMP_NUM_THREADS=$threads
        ./test/benchmark $n 3
    done
done
```

#### `scripts/bench_schedule.slurm` — Experiment 4
Same structure as above but compile three schedule variants.

### 4. Python plotting script: `scripts/plot_results.py`

Uses pandas + matplotlib. One function per plot type:
- `plot_serial_comparison(df)` → Plot 1
- `plot_strong_scaling(df)` → Plot 2a and 2b
- `plot_problem_size(df)` → Plot 3
- `plot_schedule_comparison(df)` → optional bar chart

---

## What Every Data Point Must Record (spec requirement)

The spec says every performance result must be reproducible. Each CSV row must
contain:

| Field         | Example                             |
|---------------|-------------------------------------|
| version       | v0.3-openmp-v1                      |
| git_hash      | cc9a2c6                             |
| flags         | -O3 -march=native -ffast-math -fopenmp |
| platform      | CSD3 icelake                        |
| n             | 8000                                |
| threads       | 32                                  |
| omp_proc_bind | close                               |
| rep           | 1                                   |
| time_s        | 4.721                               |
| gflops        | 28.6                                |

The benchmark program should print the git hash automatically:
```c
// In benchmark.c — embed at compile time:
#define GIT_HASH STRINGIFY(GIT_COMMIT_HASH)
// Pass via: -DGIT_COMMIT_HASH=$(git rev-parse --short HEAD)
```
Add to Makefile: `CFLAGS += -DGIT_COMMIT_HASH=$(shell git rev-parse --short HEAD)`

---

## Report Structure Mapped to Experiments

| Report Section                        | Experiment | Table    | Plot     |
|---------------------------------------|------------|----------|----------|
| Serial optimisation                   | Exp 1      | Table 1  | Plot 1   |
| OpenMP parallelisation strategy       | Exp 2      | Table 2  | Plot 2a,b|
| Performance vs problem size           | Exp 3      | —        | Plot 3   |
| Scheduling / further tuning           | Exp 4      | Table 3  | optional |
| Tagged commits (mandatory by spec)    | —          | Table 4  | —        |

### Table 4 (mandatory — tagged commits):
| Tag            | Description                            | Key change                    |
|----------------|----------------------------------------|-------------------------------|
| v0.1-baseline  | Single-threaded spec baseline          | Exact spec loop, O0           |
| v0.2-serial-opt| Serial optimised                       | Loop reorder, precompute 1/diag|
| v0.3-openmp-v1 | First working OpenMP version           | Persistent thread pool         |
| v0.4-sched     | OpenMP schedule tuning                 | static/dynamic/guided tested  |

---

## Time Estimates on CSD3 icelake (rough guide)

| Version         | n      | Threads | Est. time |
|-----------------|--------|---------|-----------|
| v0.1 (-O0)      | 4000   | 1       | ~25 s     |
| v0.1 (-O0)      | 8000   | 1       | ~200 s    |
| v0.2 (-O3)      | 8000   | 1       | ~15–30 s  |
| v0.3 (-O3+omp)  | 8000   | 76      | ~1–3 s    |
| v0.3 (-O3+omp)  | 12000  | 76      | ~3–8 s    |

**Limit v0.1-O0 sweep to n ≤ 4000** to avoid > 10 min per job.
Use `--time=00:30:00` for serial jobs, `--time=02:00:00` for full scaling sweep.

---

## Summary: Minimum Required Deliverables for Report

From the spec:
1. ✅ Table of tagged commits (Table 4)
2. ✅ Performance scaling across matrix sizes (Plot 1, Plot 3)
3. ✅ Performance scaling across number of cores (Plot 2a)
4. ✅ State CSD3 partition used (icelake — note in every figure caption)
5. ✅ Code version + compile flags + platform + threads on every data point
