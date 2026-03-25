#!/bin/bash
#SBATCH --job-name=chol_correctness
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=correctness_%j.out
#SBATCH --error=correctness_%j.err
#
# Run the full correctness suite for mphil_dis_cholesky.
#
# What this script does
# ---------------------
#  1. Builds a strict binary (no -ffast-math) and a performance binary
#     (-ffast-math) for VERSION=v5_openmp_blocked NB=96.
#  2. Runs each binary at OMP_NUM_THREADS = 1 2 4 8 76.
#     The binary itself also sweeps 1/2/4/8/76 internally via
#     omp_set_num_threads (Test 8), so thread-vs-thread diffs appear
#     in every run regardless of the external OMP_NUM_THREADS setting.
#  3. All output is saved to results/correctness_<jobid>.log.
#
# Usage
# -----
#   sbatch scripts/csd3_correctness.slurm
#
# To test a different VERSION or NB:
#   sbatch --export=ALL,VERSION=v4_openmp_blocked,NB=128 scripts/csd3_correctness.slurm

set -euo pipefail

# ---- locate project root ------------------------------------------------
if [ -f "$SLURM_SUBMIT_DIR/Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$SLURM_SUBMIT_DIR/../Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    echo "ERROR: cannot find Makefile" >&2; exit 1
fi
echo "Working directory: $(pwd)"

module purge
module load gcc/11

mkdir -p results

# ---- configurable parameters -------------------------------------------
VERSION=${VERSION:-v5_openmp_blocked}
NB=${NB:-96}
THREAD_COUNTS=${THREAD_COUNTS:-"1 2 4 8 76"}

LOG="results/correctness_${SLURM_JOB_ID}.log"
STRICT_BIN="test/test_correctness_strict"
PERF_BIN="test/test_correctness"

# ---- build ---------------------------------------------------------------
echo "=== Building  VERSION=$VERSION  NB=$NB ===" | tee "$LOG"

# strict binary (no -ffast-math)
make test-strict VERSION="$VERSION" NB="$NB" 2>&1 | tee -a "$LOG"
if [ ! -f "$STRICT_BIN" ]; then
    echo "ERROR: strict build failed" | tee -a "$LOG"; exit 1
fi

# performance binary (-ffast-math, same flags as benchmarks)
make test VERSION="$VERSION" NB="$NB" 2>&1 | tee -a "$LOG"
if [ ! -f "$PERF_BIN" ]; then
    echo "ERROR: performance build failed" | tee -a "$LOG"; exit 1
fi

echo "" | tee -a "$LOG"

# ---- thread affinity (one thread per physical core, close placement) ----
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ---- run each binary at every thread count ------------------------------
for BIN_LABEL in "strict:${STRICT_BIN}" "perf:${PERF_BIN}"; do
    LABEL="${BIN_LABEL%%:*}"
    BIN="${BIN_LABEL##*:}"

    for T in $THREAD_COUNTS; do
        echo "================================================================" | tee -a "$LOG"
        echo "  build=$LABEL  OMP_NUM_THREADS=$T" | tee -a "$LOG"
        echo "================================================================" | tee -a "$LOG"
        OMP_NUM_THREADS=$T "$BIN" 2>&1 | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    done
done

# ---- summary ------------------------------------------------------------
echo "================================================================" | tee -a "$LOG"
echo "PASS/FAIL summary per run" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
grep -E "^---" "$LOG" | tee -a /dev/null

echo "" | tee -a "$LOG"
echo "Full log: $LOG"

# Fail the job if any test failed
if grep -q "^  FAIL" "$LOG"; then
    echo "RESULT: FAILURES DETECTED — see $LOG" | tee -a "$LOG"
    exit 1
else
    echo "RESULT: all tests passed" | tee -a "$LOG"
fi
