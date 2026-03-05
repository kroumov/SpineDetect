#!/bin/bash
#SBATCH --job-name=microns_mip_volumes
#SBATCH --partition=educluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/en580-syan28/MICRONS_DATA

SCRIPT_DIR="/home/en580-syan28/MICRONS_DATA"
cd "$SCRIPT_DIR" || exit 1
module load gcc/9.3.0 python/3.11.9 slurm
source "${SCRIPT_DIR}/microns_env/bin/activate" || exit 1

set -e
# One volume per MIP; each run creates microns_<id>, then we stitch and leave it there
for MIP in 2 3 4 5 6; do
    echo "MIP ${MIP}"
    sed -i "s/^MIP_LEVEL = .*/MIP_LEVEL = ${MIP}/" config.py
    python sample_random_volume.py
    VOL_DIR=$(ls -td microns_* 2>/dev/null | head -n 1)
    if [ -z "$VOL_DIR" ]; then
        echo "ERROR: No microns_* directory found."
        exit 1
    fi
    python accumulate_roi.py "${VOL_DIR}"
done