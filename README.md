# Cholesky Coursework

This repository contains my MPhil DIS C2 High Performance Computing coursework.
The project is an in-place Cholesky factorisation routine for dense symmetric
positive-definite matrices on a single CSD3 Ice Lake node, written in C with
OpenMP.

I started from the baseline algorithm given in the brief, then improved it in
stages:

- `v1_baseline`: a direct implementation of the coursework loop at `-O0`
- `v2_serial_opt`: serial optimisation by fixing row-major memory access and
  hoisting loop-invariant values
- `v3_openmp`: my first OpenMP version, parallelising the trailing update
- `v4_openmp_blocked`: a blocked algorithm to reduce per-column
  synchronisation
- `v5_openmp_blocked`: the final version used for the main results, adding
  packing, better cache use, four-way unrolling, and improved scheduling

The tagged commits `v0.1-baseline` to `v0.5-tuned` track those stages.

## What The Library Does

The library exposes the coursework interface:

```c
#include "mphil_dis_cholesky.h"

double mphil_dis_cholesky(double *c, int n);
```

The routine expects:

- `c` to point to a 1D row-major `n x n` matrix of `double`
- the matrix to be symmetric positive-definite
- `1 <= n <= 100000`

The routine works in place:

- on entry, `c` contains the matrix `C`
- on exit, the lower triangle contains `L`
- on exit, the upper triangle contains `L^T`
- the return value is the wall-clock factorisation time in seconds

After the call, a log-determinant can be recovered as:

```c
logdet = 2.0 * sum(log(c[i*n + i]))
```

## What I Focused On

My main goal was not just to get a correct Cholesky implementation, but to show
an optimisation path that I could justify with measurements.

The main ideas I explored were:

- improving serial memory access before attempting parallelism
- keeping the OpenMP thread team alive across the whole factorisation
- moving from a flat column-by-column update to a blocked panel algorithm
- reducing long-stride accesses with packing
- improving the blocked update kernel with SIMD-friendly structure and better
  scheduling

For the main experiments, I used `v5_openmp_blocked` with `NB=96`.

## Requirements

- GCC 11 or later with OpenMP support (`-fopenmp`)
- GNU Make
- POSIX `clock_gettime` (standard on Linux/CSD3)

On CSD3 I used:

```bash
module purge
module load gcc/11
```

## Build

```bash
# Build the final benchmark version
make bench VERSION=v5_openmp_blocked NB=96

# Build and run the correctness tests
make test VERSION=v5_openmp_blocked NB=96

# Build the example program
make example VERSION=v5_openmp_blocked NB=96

# Build other stages if you want to compare them
make bench VERSION=v1_baseline
make bench VERSION=v2_serial_opt
make bench VERSION=v3_openmp
make bench VERSION=v4_openmp_blocked NB=96

# Remove build artefacts
make clean
```

The Makefile compiles `src/cholesky_<VERSION>.c` into `lib/libcholesky.a` and
links the test and example binaries against it.

## Repository Layout

- `src/`: implementation versions
- `include/`: public header
- `example/`: example program and example CSD3 submission script
- `test/`: correctness test suite and benchmark driver
- `scripts/`: CSD3 benchmark/correctness scripts and plotting helpers
- `results/`: committed CSV results and correctness logs
- `report/`: report source and submitted report files

## Example Usage

The example program in `example/example.c` builds the coursework `corr()`
matrix, runs the factorisation, and prints elapsed time, log-determinant, and
GFLOP/s.

```bash
make example VERSION=v5_openmp_blocked NB=96
./example/example 4000
```

On CSD3, the recommended full-node settings are:

```bash
export OMP_NUM_THREADS=76
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./example/example 8000
```

## Testing

```bash
make test VERSION=v5_openmp_blocked NB=96
make test-strict VERSION=v5_openmp_blocked NB=96
sbatch scripts/csd3_correctness.slurm
```

The test suite covers:

- exact 2x2 and 3x3 examples
- `n=1` and out-of-range guard checks
- known-`L` reconstruction cases near block boundaries
- numerically stressed diagonal SPD cases
- coursework `corr()` reconstruction tests
- selected external log-determinant references for `corr()`
- thread-agreement checks against a 1-thread reference

I kept both a performance build and a strict build without `-ffast-math` so I
could check that the fast build was not hiding correctness problems.

## Running Benchmarks On CSD3

```bash
# Serial comparison (v1, v2)
sbatch scripts/csd3_serial.slurm

# Strong scaling study (v3, v4, v5)
sbatch scripts/csd3_scaling.slurm

# Block-size sweep
sbatch --export=ALL,VERSION=v5_openmp_blocked,THREAD_LIST="1 8 32 76" \
    scripts/csd3_block_sweep.slurm
```

These scripts write CSV files into `results/`. The plotting scripts then turn
those CSVs into the figures used in the report.

Run submission scripts from the project root, where the `Makefile` lives.

## Performance Notes

In my main scaling runs on CSD3 Ice Lake, the final version
`v5_openmp_blocked` with `NB=96` reported:

| n    | GFLOP/s at 76 threads | Speedup vs 1 thread |
|------|------------------------|---------------------|
| 2000 | 238                    | 23x                 |
| 4000 | 205                    | 28x                 |
| 6000 | 186                    | 28x                 |
| 8000 | 183                    | 29x                 |

The block sweep showed that the best block size depends on thread count:

- `NB=96` was best in the committed sweep at 1 and 76 threads
- `NB=256` was best at 8 and 32 threads

I kept `NB=96` for the main scaling runs because those runs focused on the
full-node case and `NB=96` performed well there.

## Report

The report files are in `report/`:

- `report/report.pdf`
- `report/report.txt`
- `report/report.tex`

The report explains the optimisation path, the OpenMP strategy, the tagged
development stages, and the performance results used to support the final
implementation.

## AI Usage

I used AI tools during development as coding assistants. They were most useful
for drafting comments, checking ideas for tests, and helping me debug some
correctness and performance issues. I reviewed and adapted those suggestions,
and I take responsibility for the final code, experiments, and report.
