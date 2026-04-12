#!/bin/bash
#SBATCH --job-name=microns_batch
#SBATCH --partition=educluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# MIP levels to download; total volumes split evenly across these MIPs
MIPS=(3)
TOTAL_COUNT=1
DOWNLOAD_BASE="microns_data_download"
# Sampling ratio range [min, max]; each run picks a random value in range
NEURON_RATIO_MIN=0.01
NEURON_RATIO_MAX=0.05
VESSEL_RATIO_MIN=0.1
VESSEL_RATIO_MAX=0.9

SCRIPT_DIR="SpineDetect/tpm_simulator/sample_random_volume"
cd "$SCRIPT_DIR" || exit 1
# source "${SCRIPT_DIR}/microns_env/bin/activate" || exit 1

if [ -f "$DOWNLOAD_BASE" ]; then rm -f "$DOWNLOAD_BASE"; fi
mkdir -p "$DOWNLOAD_BASE"

NUM_MIPS=${#MIPS[@]}
PER_MIP=$((TOTAL_COUNT / NUM_MIPS))

for MIP in "${MIPS[@]}"; do
    MIP_DIR="${DOWNLOAD_BASE}/mip${MIP}"
    mkdir -p "$MIP_DIR"
    echo "MIP ${MIP} (${PER_MIP} volumes)"
    sed -i "s/^MIP_LEVEL = .*/MIP_LEVEL = ${MIP}/" config.py

    for ((v = 1; v <= PER_MIP; v++)); do
        echo "[MIP ${MIP}] ${v}/${PER_MIP}"
        NR=$(awk "BEGIN { srand(); printf \"%.4f\", $NEURON_RATIO_MIN + ($NEURON_RATIO_MAX - $NEURON_RATIO_MIN) * rand() }")
        VR=$(awk "BEGIN { srand(); printf \"%.4f\", $VESSEL_RATIO_MIN + ($VESSEL_RATIO_MAX - $VESSEL_RATIO_MIN) * rand() }")
        sed -i "s/^NEURON_SAMPLING_RATIO = .*/NEURON_SAMPLING_RATIO = $NR/" config.py
        sed -i "s/^VESSEL_SAMPLING_RATIO = .*/VESSEL_SAMPLING_RATIO = $VR/" config.py
        python sample_random_volume.py
        VOL_DIR=$(ls -td microns_* 2>/dev/null | head -n 1)
        if [ -z "$VOL_DIR" ]; then
            echo "ERROR: No microns_* directory."
            continue
        fi
        python accumulate_roi.py "$VOL_DIR"
        vol_name=$(printf "vol_%04d" "$v")
        mv "$VOL_DIR" "${MIP_DIR}/${vol_name}"   # save as mipN/vol_NNNN
    done
done

echo "Done: $DOWNLOAD_BASE"