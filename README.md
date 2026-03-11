# cholesky

In-place Cholesky factorisation (`C = L Lᵀ`) optimised for multi-core CPUs,
developed for the MPhil DIS C2 HPC coursework (CSD3 icelake, 76-core Intel
nodes).

## Requirements

- GCC 11 or later with OpenMP support (`-fopenmp`)
- GNU Make
- POSIX `clock_gettime` (standard on Linux/CSD3)

On CSD3:
```bash
module purge
module load gcc/11
```

## Build

```bash
# Build the best (panel-blocked OpenMP + micro-opts) version — recommended for performance
make bench VERSION=v6_openmp_blocked NB=96

# Build and run correctness tests
make test VERSION=v6_openmp_blocked NB=96

# Build the example program
make example VERSION=v6_openmp_blocked NB=96

# Build any other version
make bench VERSION=v3_openmp
make bench VERSION=v1_baseline

# Remove all build artefacts
make clean
```

The build system compiles `src/cholesky_<VERSION>.c` into `lib/libcholesky.a`
and links it with the test/example binaries.

## Available Versions

| Version | Description |
|---------|-------------|
| `v1_baseline` | Exact spec loop, `-O0`, no optimisation |
| `v2_serial_opt` | Loop interchange, hoisting, reciprocal division, `-O3` |
| `v3_openmp` | First OpenMP parallel version: `omp for schedule(static)` |
| `v5_serial_blocked` | Panel-blocked serial reference (tune with `NB=N`) |
| `v5_openmp_blocked` | Panel-blocked OpenMP — baseline blocked version (tune with `NB=N`) |
| `v6_openmp_blocked` | Panel-blocked OpenMP + 4 cache/SIMD opts: col-pack, L11 cache, j×4 unroll, static,1 schedule |

## Usage

```c
#include "mphil_dis_cholesky.h"

// c: n*n matrix in row-major order, modified in-place
// Returns wall clock time in seconds, or -1.0 if n is out of range [1,100000]
double t = mphil_dis_cholesky(c, n);
```

After the call:
- Lower triangle `c[i*n+j]` for `i >= j` contains `L[i,j]`
- Upper triangle `c[i*n+j]` for `i < j` contains `L^T[i,j] = L[j,i]`
- `log|C| = 2 * Σ log(c[i*n+i])` for `i = 0..n-1`

See `example/example.c` for a complete usage example.

## Performance Guidance

For best performance use `v6_openmp_blocked` with the following environment
variables (set automatically by the provided SLURM scripts):

```bash
export OMP_NUM_THREADS=76        # one thread per physical core
export OMP_PROC_BIND=close       # bind threads to nearby cores
export OMP_PLACES=cores          # one thread per core (no SMT)
./example/example 8000
```

The panel width `NB=96` is empirically optimal on CSD3 icelake (96 doubles ≈
768 B; the packed NB×NB block is ~74 KB, fitting in the 2 MB L2 per core).
Smaller matrices (n < 500) are memory-bound at any thread count.

Measured peak performance on CSD3 icelake (76 threads, `v6_openmp_blocked`, NB=96):

| n    | GFlop/s | Speedup vs 1 thread |
|------|---------|---------------------|
| 2000 | 238     | 23×                 |
| 4000 | 205     | 28×                 |
| 6000 | 186     | 28×                 |
| 8000 | 181     | 29×                 |

## Running Benchmarks on CSD3

```bash
# Serial comparison (v1, v2, v3) — submits to icelake INTR queue
sbatch scripts/csd3_serial.slurm

# OpenMP strong scaling (v3_openmp vs v5_openmp_blocked, n=2000–8000, 1–76 threads)
sbatch scripts/csd3_scaling.slurm

# Results are written to results/csd3_serial.csv and results/csd3_scaling.csv
```

Submit scripts from the project root directory (where the `Makefile` lives),
not from inside `scripts/`.

## Running the Example on CSD3

```bash
# Build first, then submit
make example VERSION=v6_openmp_blocked NB=96
sbatch example/submit_csd3.slurm
```

## Testing

```bash
make test VERSION=v6_openmp_blocked NB=96
```

Runs four test suites: 2×2 spec example, 3×3 hand-computed, `L Lᵀ`
reconstruction at n=5/50/200, and bounds checking.
