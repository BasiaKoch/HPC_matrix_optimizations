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

# Always run from project root, regardless of where sbatch was called from
cd "$(dirname "$0")/.."

module purge
module load gcc/11

# Print gcc version so we know exactly what compiled the code
gcc --version

mkdir -p results

REPS=3

export OMP_NUM_THREADS=1

echo "version,n,threads,rep,time_s,gflops" > results/csd3_serial.csv

# v1_baseline: -O0 + strided access is very slow; cap at n=2000
for VERSION in v1_baseline; do
    echo ">>> Building $VERSION ..."
    make clean
    make bench VERSION=$VERSION
    if [ ! -f ./test/benchmark ]; then echo "ERROR: failed to build $VERSION, skipping"; continue; fi
    SIZES="500 1000 2000"
    echo ">>> Running $VERSION  SIZES=$SIZES  REPS=$REPS"
    for n in $SIZES; do
        echo "    $VERSION n=$n ..."
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_serial.csv
    done
done

# v2, v3, v4: -O3; extend to n=4000 to see cache effects
for VERSION in v2_serial_opt v3_serial_opt; do
    echo ">>> Building $VERSION ..."
    make clean
    make bench VERSION=$VERSION
    if [ ! -f ./test/benchmark ]; then echo "ERROR: failed to build $VERSION, skipping"; continue; fi
    SIZES="500 1000 2000 4000"
    echo ">>> Running $VERSION  SIZES=$SIZES  REPS=$REPS"
    for n in $SIZES; do
        echo "    $VERSION n=$n ..."
        ./test/benchmark $n $REPS | sed "s/^/${VERSION},/" >> results/csd3_serial.csv
    done
done

# v4_serial_blocked: sweep block sizes to find best BS for this platform
for BS in 16 32 64 128; do
    VERSION=v4_serial_blocked
    echo ">>> Building $VERSION BS=$BS ..."
    make clean
    make bench VERSION=$VERSION BS=$BS
    if [ ! -f ./test/benchmark ]; then echo "ERROR: failed to build $VERSION BS=$BS, skipping"; continue; fi
    SIZES="500 1000 2000 4000"
    echo ">>> Running ${VERSION}_BS${BS}  SIZES=$SIZES  REPS=$REPS"
    for n in $SIZES; do
        echo "    ${VERSION}_BS${BS} n=$n ..."
        ./test/benchmark $n $REPS | sed "s/^/v4_blocked_BS${BS},/" >> results/csd3_serial.csv
    done
done

echo "Done. Results in results/csd3_serial.csv"
cat results/csd3_serial.csv
