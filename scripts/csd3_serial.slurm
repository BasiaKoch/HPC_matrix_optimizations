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
#SBATCH --output=csd3_serial_%j.out
#SBATCH --error=csd3_serial_%j.err

# Serial comparison: v1, v2, v3 at 1 thread
# Platform: CSD3 icelake
# Threads: 1 (OMP_NUM_THREADS=1 so any OpenMP version also runs single-threaded)

module purge
module load gcc/11

# Print gcc version so we know exactly what compiled the code
gcc --version

mkdir -p results

VERSIONS="v1_baseline v2_serial_opt v3_serial_opt"
REPS=3

export OMP_NUM_THREADS=1

echo "version,n,threads,rep,time_s,gflops" > results/csd3_serial.csv

for VERSION in $VERSIONS; do
    echo ">>> Building $VERSION ..."
    make clean
    make bench VERSION=$VERSION
    if [ ! -f ./test/benchmark ]; then
        echo "ERROR: failed to build $VERSION, skipping"
        continue
    fi

    # All versions capped at n=2000 for first exploration run.
    # Extend SIZES once timings are confirmed sensible.
    SIZES="500 1000 2000"

    echo ">>> Running $VERSION  SIZES=$SIZES  REPS=$REPS"
    for n in $SIZES; do
        echo "    $VERSION n=$n ..."
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_serial.csv
    done
done

echo "Done. Results in results/csd3_serial.csv"
cat results/csd3_serial.csv
