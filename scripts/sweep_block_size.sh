#!/bin/bash
# sweep_block_size.sh — compile a blocked OpenMP version with different BLOCK_NB values
# and record performance for each panel width and thread count.
#
# Usage (on CSD3, inside a SLURM job or interactive session):
#   bash scripts/sweep_block_size.sh
#
# Output:
#   results/block_sweep.csv   columns: version,BLOCK_NB,n,threads,rep,time_s,gflops
#
# The script must be run from the project root (where Makefile lives).
# The benchmark binary (test/benchmark) re-fills the matrix before every rep,
# so each timing is independent.

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration — override with env vars if needed
# -----------------------------------------------------------------------
N=${N:-8000}                                 # Matrix size
THREAD_LIST=${THREAD_LIST:-${THREADS:-76}}   # Backward-compatible: THREADS still works
REPS=${REPS:-3}                              # Repetitions per (BLOCK_NB, n, threads) cell
BLOCK_SIZES=${BLOCK_SIZES:-"64 96 128 192 256"}
VERSION=${VERSION:-v4_openmp_blocked}        # Override as needed
OUT=${OUT:-results/block_sweep.csv}

# Thread affinity settings — keep threads on nearby physical cores
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

# -----------------------------------------------------------------------
mkdir -p results

# Always write a fresh header — overwrites any stale/incorrect previous run
echo "version,BLOCK_NB,n,threads,rep,time_s,gflops" > "$OUT"

echo "Block-size sweep: n=$N, reps=$REPS"
echo "Thread counts: $THREAD_LIST"
echo "Panel widths: $BLOCK_SIZES"
echo "Output: $OUT"
echo ""

for NB in $BLOCK_SIZES; do
    echo "--- Building $VERSION with BLOCK_NB=$NB ---"
    # Force a clean rebuild for every NB: Make tracks source-file timestamps,
    # not compiler flags, so without 'make clean' it reuses the previous binary
    # and all iterations silently run the same BLOCK_NB value.
    make clean
    make bench VERSION=$VERSION NB=$NB

    if [ ! -f ./test/benchmark ]; then
        echo "ERROR: build failed for NB=$NB" >&2
        exit 1
    fi

    for threads in $THREAD_LIST; do
        export OMP_NUM_THREADS="$threads"
        echo "    Running n=$N, threads=$threads, reps=$REPS ..."
        # benchmark outputs: n,threads,rep,time_s,gflops
        # Prepend version and BLOCK_NB before appending to the CSV.
        ./test/benchmark "$N" "$REPS" | sed "s/^/${VERSION},${NB},/" >> "$OUT"
    done
done

echo ""
echo "Sweep complete. Results:"
cat "$OUT"
