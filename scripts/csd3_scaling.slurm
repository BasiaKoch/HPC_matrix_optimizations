#!/bin/bash
#SBATCH --job-name=chol_scaling
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --qos=INTR
#SBATCH --output=results/csd3_scaling_%j.out
#SBATCH --error=results/csd3_scaling_%j.err

# Strong scaling study: fix n, vary thread count 1–76
# Experiments 2 & 3 from BENCHMARKING_PLAN.md
# Platform: CSD3 icelake (76 cores/node)
# Version: v3_openmp (change VERSION= for other versions)

module purge
module load gcc/11

mkdir -p results

VERSION=v3_openmp
REPS=3

# Thread affinity: keep threads on nearby cores, one thread per core
export OMP_PROC_BIND=close
export OMP_PLACES=cores

make bench VERSION=$VERSION -s

echo "version,n,threads,rep,time_s,gflops" > results/csd3_scaling.csv

# Experiment 2: strong scaling — fix n, vary threads
for n in 2000 4000 8000 12000; do
    for threads in 1 2 4 8 16 32 48 64 76; do
        export OMP_NUM_THREADS=$threads
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_scaling.csv
    done
done

echo "Done. Results in results/csd3_scaling.csv"
cat results/csd3_scaling.csv
