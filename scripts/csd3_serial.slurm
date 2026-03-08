#!/bin/bash
#SBATCH --job-name=chol_serial
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --qos=INTR
#SBATCH --output=results/csd3_serial_%j.out
#SBATCH --error=results/csd3_serial_%j.err

# Serial comparison: v1, v2, v3 at 1 thread
# Platform: CSD3 icelake
# Threads: 1 (OMP_NUM_THREADS=1 so any OpenMP version also runs single-threaded)

module purge
module load gcc/11

mkdir -p results

VERSIONS="v1_baseline v2_serial_opt v3_openmp"   # <-- add/change versions here
SIZES="500 1000 2000 4000 8000"
REPS=3

export OMP_NUM_THREADS=1

echo "version,n,threads,rep,time_s,gflops" > results/csd3_serial.csv

for VERSION in $VERSIONS; do
    make bench VERSION=$VERSION -s
    for n in $SIZES; do
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_serial.csv
    done
done

echo "Done. Results in results/csd3_serial.csv"
cat results/csd3_serial.csv
