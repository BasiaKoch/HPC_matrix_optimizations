#!/bin/bash
#SBATCH --job-name=chol_nb_sweep
#SBATCH -A FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --qos=INTR
#SBATCH --output=csd3_block_sweep_%j.out
#SBATCH --error=csd3_block_sweep_%j.err

# Block-size sweep for blocked OpenMP versions on CSD3 icelake.
# Submits from project root:
#   sbatch scripts/csd3_block_sweep.slurm
#
# Useful overrides:
#   sbatch --export=ALL,VERSION=v5_openmp_blocked scripts/csd3_block_sweep.slurm
#   sbatch --export=ALL,THREAD_LIST="1 8 32 76" scripts/csd3_block_sweep.slurm
#   sbatch --export=ALL,VERSION=v5_openmp_blocked,THREAD_LIST="1 8 32 76",BLOCK_SIZES="64 96 128 192 256" scripts/csd3_block_sweep.slurm

# Always run from the project root (where Makefile lives)
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

bash scripts/sweep_block_size.sh

echo "Done."
