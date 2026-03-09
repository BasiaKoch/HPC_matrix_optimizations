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
# Versions: v3_openmp (barrier-based), v4_tuned (task-DAG), v5_openmp_blocked (panel-blocked)

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

REPS=3

# Thread affinity: keep threads on nearby cores, one thread per core
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "version,n,threads,rep,time_s,gflops" > results/csd3_scaling.csv

# Strong scaling: n=2000 fits in L3 (compute-bound), n=4000 exceeds L3 (bandwidth-bound)
# n=8000+ excluded: single-threaded run alone would exceed 1-hour INTR limit
for VERSION in v3_openmp; do
    make clean
    make bench VERSION=$VERSION
    if [ ! -f ./test/benchmark ]; then echo "ERROR: build failed for $VERSION" >&2; exit 1; fi

    for n in 2000 4000 6000 8000; do
        for threads in 1 2 4 8 16 32 48 64 76; do
            export OMP_NUM_THREADS=$threads
            ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_scaling.csv
        done
    done
done

# v5_openmp_blocked: panel-blocked with default panel width NB=128
# Use tag "v5_blocked_NB128" in CSV so plots can distinguish panel widths
make clean
make bench VERSION=v5_openmp_blocked NB=128
if [ ! -f ./test/benchmark ]; then echo "ERROR: build failed for v5_openmp_blocked" >&2; exit 1; fi
for n in 2000 4000 6000 8000; do
    for threads in 1 2 4 8 16 32 48 64 76; do
        export OMP_NUM_THREADS=$threads
        ./test/benchmark $n $REPS | sed "s/^/v5_blocked_NB128,/" >> results/csd3_scaling.csv
    done
done

echo "Done. Results in results/csd3_scaling.csv"
cat results/csd3_scaling.csv
