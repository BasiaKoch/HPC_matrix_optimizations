#!/bin/bash
#SBATCH --job-name=chol_scaling
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --qos=INTR
#SBATCH --output=csd3_scaling_%j.out
#SBATCH --error=csd3_scaling_%j.err

# Strong scaling study: fix n, vary thread count 1–76
# Platform: CSD3 icelake (76 cores/node)
# Versions: v3_openmp (flat OpenMP), v4_openmp_blocked (panel-blocked baseline), v5_openmp_blocked (cache/SIMD tuned)
#
# Usage examples:
#   Full scaling study (default):
#     sbatch scripts/csd3_scaling.slurm
#
#   Fig. 6 same-size comparison only (all versions at n=8000):
#     sbatch --export=ALL,STUDY=fig6 scripts/csd3_scaling.slurm
#
#   Fill missing v3 curve at n=6000 only (all thread counts):
#     sbatch --export=ALL,STUDY=v3_n6000_curve scripts/csd3_scaling.slurm
#
#   Faster Fig. 6 smoke run:
#     sbatch --export=ALL,STUDY=fig6,THREAD_LIST="1 8 32 76",REPS_V3=2,REPS_BLOCKED=2 scripts/csd3_scaling.slurm
#
#   Run a focused study and merge into main scaling CSV (replace overlap by version+n):
#     sbatch --export=ALL,STUDY=v3_n6000_curve,MERGE_INTO_MAIN=1 scripts/csd3_scaling.slurm

# Always run from project root (where Makefile lives)
if [ -f "$SLURM_SUBMIT_DIR/Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$SLURM_SUBMIT_DIR/../Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    echo "ERROR: cannot find project root" >&2; exit 1
fi
echo "Working directory: $(pwd)"

module purge
module load gcc/11

mkdir -p results

# -------------------------------------------------------------------
# Configurable parameters (override via --export=ALL,...)
# -------------------------------------------------------------------
STUDY=${STUDY:-full}                     # full | fig6 | v3_n6000_curve
REPS=${REPS:-3}                         # default reps
REPS_V3=${REPS_V3:-$REPS}               # reps for v3_openmp
REPS_BLOCKED=${REPS_BLOCKED:-$REPS}     # reps for v4/v5
THREAD_LIST=${THREAD_LIST:-"1 2 4 8 16 32 48 64 76"}
MERGE_INTO_MAIN=${MERGE_INTO_MAIN:-0}   # append OUT_CSV rows into results/csd3_scaling.csv
REPLACE_OVERLAP=${REPLACE_OVERLAP:-1}   # when merging, replace rows with matching (version,n)

# Per-version toggles (useful for targeted gap-fill runs)
RUN_V3=${RUN_V3:-1}
RUN_V4=${RUN_V4:-1}
RUN_V5=${RUN_V5:-1}

if [ "$STUDY" = "fig6" ]; then
    N_LIST_V3=${N_LIST_V3:-"8000"}
    N_LIST_BLOCKED=${N_LIST_BLOCKED:-"8000"}
    RUN_V3=${RUN_V3:-1}
    RUN_V4=${RUN_V4:-1}
    RUN_V5=${RUN_V5:-1}
elif [ "$STUDY" = "v3_n6000_curve" ]; then
    N_LIST_V3=${N_LIST_V3:-"6000"}
    N_LIST_BLOCKED=${N_LIST_BLOCKED:-""}
    RUN_V3=1
    RUN_V4=0
    RUN_V5=0
else
    N_LIST_V3=${N_LIST_V3:-"2000 4000 6000 8000"}
    N_LIST_BLOCKED=${N_LIST_BLOCKED:-"2000 4000 6000 8000"}
fi

if [ -z "${OUT_CSV:-}" ]; then
    if [ "$STUDY" = "fig6" ]; then
        OUT_CSV="results/csd3_scaling_fig6.csv"
    elif [ "$STUDY" = "v3_n6000_curve" ]; then
        OUT_CSV="results/csd3_scaling_v3_n6000_curve.csv"
    else
        OUT_CSV="results/csd3_scaling.csv"
    fi
fi

# Thread affinity: keep threads on nearby cores, one thread per core
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "version,n,threads,rep,time_s,gflops" > "$OUT_CSV"
echo "Study mode      : $STUDY"
echo "V3 sizes        : $N_LIST_V3"
echo "Blocked sizes   : $N_LIST_BLOCKED"
echo "Thread list     : $THREAD_LIST"
echo "Reps (v3)       : $REPS_V3"
echo "Reps (v4/v5)    : $REPS_BLOCKED"
echo "Run v3          : $RUN_V3"
echo "Run v4          : $RUN_V4"
echo "Run v5          : $RUN_V5"
echo "Output CSV      : $OUT_CSV"
echo "Merge into main : $MERGE_INTO_MAIN"
echo "Replace overlap : $REPLACE_OVERLAP"

build_version() {
    local version="$1"
    local nb="$2"

    make clean
    if [ -n "$nb" ]; then
        make bench VERSION="$version" NB="$nb"
    else
        make bench VERSION="$version"
    fi

    if [ ! -f ./test/benchmark ]; then
        echo "ERROR: build failed for $version" >&2
        exit 1
    fi
}

run_scaling_grid() {
    local csv_tag="$1"
    local n_list="$2"
    local reps="$3"

    for n in $n_list; do
        for threads in $THREAD_LIST; do
            export OMP_NUM_THREADS="$threads"
            ./test/benchmark "$n" "$reps" | sed "s/^/${csv_tag},/" >> "$OUT_CSV"
        done
    done
}

if [ "$RUN_V3" = "1" ]; then
    # v3_openmp: flat OpenMP baseline
    build_version "v3_openmp" ""
    run_scaling_grid "v3_openmp" "$N_LIST_V3" "$REPS_V3"
fi

if [ "$RUN_V4" = "1" ]; then
    # v4_openmp_blocked: panel-blocked baseline with NB=96
    build_version "v4_openmp_blocked" "96"
    run_scaling_grid "v4_blocked_NB96" "$N_LIST_BLOCKED" "$REPS_BLOCKED"
fi

if [ "$RUN_V5" = "1" ]; then
    # v5_openmp_blocked: v4 + col-pack, L11 cache, j×4 unroll, schedule(static,1)
    build_version "v5_openmp_blocked" "96"
    run_scaling_grid "v5_blocked_NB96" "$N_LIST_BLOCKED" "$REPS_BLOCKED"
fi

echo "Done. Results in $OUT_CSV"
cat "$OUT_CSV"

if [ "$MERGE_INTO_MAIN" = "1" ] && [ "$OUT_CSV" != "results/csd3_scaling.csv" ]; then
    if [ "$REPLACE_OVERLAP" = "1" ]; then
        if [ ! -f results/csd3_scaling.csv ]; then
            cp "$OUT_CSV" results/csd3_scaling.csv
        else
            TMP_MERGE=$(mktemp)
            # Keep existing rows that do NOT overlap (by version,n) with new rows.
            awk -F, 'NR==FNR { if (FNR>1) keys[$1 FS $2]=1; next }
                     FNR==1 { print; next }
                     !($1 FS $2 in keys)' \
                "$OUT_CSV" results/csd3_scaling.csv > "$TMP_MERGE"
            # Append new rows (without header).
            tail -n +2 "$OUT_CSV" >> "$TMP_MERGE"
            mv "$TMP_MERGE" results/csd3_scaling.csv
        fi
    else
        if [ ! -f results/csd3_scaling.csv ]; then
            cp "$OUT_CSV" results/csd3_scaling.csv
        else
            tail -n +2 "$OUT_CSV" >> results/csd3_scaling.csv
        fi
    fi
    echo "Merged into results/csd3_scaling.csv (REPLACE_OVERLAP=$REPLACE_OVERLAP)"
fi
