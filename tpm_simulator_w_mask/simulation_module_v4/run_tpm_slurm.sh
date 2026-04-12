#!/bin/bash
#SBATCH --job-name=tpm_matlab
#SBATCH --partition=educluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# cd to submit dir (module root), create logs/output
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")"
fi
MODULE_DIR="$(pwd)"
mkdir -p "$MODULE_DIR/logs" "$MODULE_DIR/output"

module load matlab/R2024a

# Run main(); for single folder use main('microns_864691136811782003')
matlab -batch "cd('$MODULE_DIR'); main(); exit"