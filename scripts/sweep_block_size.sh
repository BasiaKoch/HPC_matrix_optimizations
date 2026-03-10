#!/bin/bash
# sweep_block_size.sh — compile v5_openmp_blocked with different BLOCK_NB values
# and record performance for each panel width.
#
# Usage (on CSD3, inside a SLURM job or interactive session):
#   bash scripts/sweep_block_size.sh
#
# Output:
#   results/block_sweep.csv   columns: BLOCK_NB,n,threads,rep,time_s,gflops
#
# The script must be run from the project root (where Makefile lives).
# The benchmark binary (test/benchmark) re-fills the matrix before every rep,
# so each timing is independent.

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration — adjust if needed
# -----------------------------------------------------------------------
N=8000                          # Matrix size (large enough to be memory-bound)
THREADS=76                      # Thread count (all icelake physical cores)
REPS=3                          # Repetitions per (BLOCK_NB, n, threads) cell
BLOCK_SIZES="64 96 128 192 256" # Panel widths to sweep
OUT=results/block_sweep.csv

# Thread affinity settings — keep threads on nearby physical cores
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=$THREADS

# -----------------------------------------------------------------------
mkdir -p results

# Write CSV header (only if file does not exist yet)
if [ ! -f "$OUT" ]; then
    echo "BLOCK_NB,n,threads,rep,time_s,gflops" > "$OUT"
fi

echo "Block-size sweep: n=$N, threads=$THREADS, reps=$REPS"
echo "Panel widths: $BLOCK_SIZES"
echo "Output: $OUT"
echo ""

for NB in $BLOCK_SIZES; do
    echo "--- Building v5_openmp_blocked with BLOCK_NB=$NB ---"
    make bench VERSION=v5_openmp_blocked NB=$NB

    if [ ! -f ./test/benchmark ]; then
        echo "ERROR: build failed for NB=$NB" >&2
        exit 1
    fi

    echo "    Running n=$N, threads=$THREADS, reps=$REPS ..."
    # benchmark outputs:  n,threads,rep,time_s,gflops
    # Prepend BLOCK_NB to each line before appending to the CSV
    ./test/benchmark $N $REPS | sed "s/^/${NB},/" >> "$OUT"
done

echo ""
echo "Sweep complete. Results:"
cat "$OUT"
