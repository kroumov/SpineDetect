#!/bin/bash
# degrade_array.sh - SLURM Job Array task: process 4 chunks in parallel.
# Submitted by degrade.sh when --parallel.
# All output: logs/degrade/<timestamp>/ (via DEGRADE_RUN_DIR)

#SBATCH --job-name=tpm_degrade
#SBATCH --partition=educluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/degrade/array_%a.log
#SBATCH --error=logs/degrade/array_%a.log

CHUNKS_PER_TASK=4
set -e
CLOUD_DIR="${CLOUD_DIR:?CLOUD_DIR not set}"
CHUNKS_FILE="${CHUNKS_FILE:?CHUNKS_FILE not set}"
DEGRADE_RUN_DIR="${DEGRADE_RUN_DIR:-$CLOUD_DIR/logs/degrade}"
mkdir -p "$DEGRADE_RUN_DIR"
cd "$CLOUD_DIR" || exit 1

LOG_FILE="$DEGRADE_RUN_DIR/array_${SLURM_ARRAY_TASK_ID}.log"
export DEGRADE_DEBUG_LOG="$DEGRADE_RUN_DIR/debug_${SLURM_ARRAY_TASK_ID}.log"

start=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNKS_PER_TASK + 1 ))
end=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_TASK ))
FOLDERS=()
while IFS= read -r line; do
  [[ -n "$line" && -d "$line" ]] && FOLDERS+=("$line")
done < <(sed -n "${start},${end}p" "$CHUNKS_FILE")

[[ ${#FOLDERS[@]} -eq 0 ]] && { echo "ERROR: no chunks for task $SLURM_ARRAY_TASK_ID"; exit 1; }

echo "[$(date +%H:%M:%S)] Task $SLURM_ARRAY_TASK_ID: ${#FOLDERS[@]} chunks"
for chunk_path in "${FOLDERS[@]}"; do
  chunk_name=$(basename "$chunk_path")
  (
    bash "$CLOUD_DIR/degrade_one_chunk.sh" "$chunk_path" "$LOG_FILE"
    echo $? > "$DEGRADE_RUN_DIR/exit_${chunk_name}.$$"
  ) &
done
set +e
wait
wait_ret=$?
set -e

OK_COUNT=0
for chunk_path in "${FOLDERS[@]}"; do
  chunk_name=$(basename "$chunk_path")
  for f in "$DEGRADE_RUN_DIR/exit_${chunk_name}".*; do
    [[ -f "$f" ]] || continue
    [[ "$(cat "$f" 2>/dev/null)" == "0" ]] && ((OK_COUNT++)) || true
    rm -f "$f"
    break
  done
done
echo "[$(date +%H:%M:%S)] Task $SLURM_ARRAY_TASK_ID: $OK_COUNT/${#FOLDERS[@]} succeeded"
[[ $wait_ret -ne 0 ]] && exit 1
[[ $OK_COUNT -lt ${#FOLDERS[@]} ]] && exit 1
exit 0