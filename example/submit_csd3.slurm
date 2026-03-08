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

# Example: run mphil_dis_cholesky on a 10000x10000 matrix using 76 threads.
# See README.md for build instructions.

module purge
module load gcc/11

export OMP_NUM_THREADS=76
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p results
./example/example 10000
