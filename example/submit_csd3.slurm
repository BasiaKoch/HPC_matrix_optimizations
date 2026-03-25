#!/bin/bash
#SBATCH --job-name=chol_example
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --qos=INTR
#SBATCH --output=results/example_%j.out
#SBATCH --error=results/example_%j.err

# Example: run mphil_dis_cholesky on a 10000x10000 matrix using 76 threads.
# Build first with:
#   make example VERSION=v5_openmp_blocked NB=96
# Then submit from the project root with:
#   sbatch example/submit_csd3.slurm

set -euo pipefail

# Run from the project root regardless of where sbatch was called.
if [ -f "$SLURM_SUBMIT_DIR/Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$SLURM_SUBMIT_DIR/../Makefile" ]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    echo "ERROR: cannot find project root" >&2
    exit 1
fi

module purge
module load gcc/11

export OMP_NUM_THREADS=76
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p results
if [ ! -x ./example/example ]; then
    echo "ERROR: ./example/example not found; build it first with 'make example VERSION=v5_openmp_blocked NB=96'" >&2
    exit 1
fi
./example/example 10000
