#!/bin/bash
# Run full pipeline: download -> downsample -> chunk -> degrade -> noise -> interpolate
# - Single volume per run (only process this run's downloaded data)
# - Parallel: degrade, noise, interpolate run PARALLEL_JOBS chunks in parallel
# - Checkpoint/resume: on failure, resume from broken step
#
# Usage:
#   ./run_pipeline.sh              # run interactively
#   sbatch run_pipeline.sh         # submit to SLURM

#SBATCH --job-name=tpm_pipeline
#SBATCH --partition=educluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e
set -o pipefail  # Propagate failure through pipes (e.g. cmd | tee) so we exit on step error and can resume
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  CLOUD_DIR="$SLURM_SUBMIT_DIR"
else
  CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$CLOUD_DIR/config.sh"
cd "$CLOUD_DIR" || exit 1
export CLOUD_LOGS_DIR="$CLOUD_DIR/logs"
mkdir -p logs logs/pipeline

DOWNLOAD_DIR="${DATA_ROOT:-$PROJECT_ROOT}/data/download"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
PIPELINE_LOG="logs/pipeline/pipeline_${RUN_ID}.txt"
# Debug goes to pipeline log (same file); subscripts (e.g. noise) get DEBUG_LOG so they append here too
DEBUG_LOG="$CLOUD_DIR/$PIPELINE_LOG"
export DEBUG_LOG
[[ -n "${SLURM_JOB_ID:-}" ]] && exec 1> >(stdbuf -oL tee -a "$DEBUG_LOG") 2>&1
[[ -z "${SLURM_JOB_ID:-}" ]] && USE_TEE=true || USE_TEE=false
trap '[ -n "${DEBUG_LOG:-}" ] && echo "DEBUG $(date +%Y-%m-%d\ %H:%M:%S) EXIT rc=$?" >> "$DEBUG_LOG"' EXIT

echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') START SLURM_JOB_ID=$SLURM_JOB_ID CLOUD_DIR=$CLOUD_DIR" >> "$DEBUG_LOG"
echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') RUN_ID=$RUN_ID PIPELINE_LOG=$PIPELINE_LOG" >> "$DEBUG_LOG"

command -v module &>/dev/null && module load matlab/R2024a 2>/dev/null || true

# --- State helpers (simple key=value format) ---
read_state() {
  STEP=""; VOLUME=""; BEFORE_DL=""
  if [[ -f "$STATE_FILE" ]]; then
    STEP=$(grep '^step=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-)
    VOLUME=$(grep '^volume=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-)
    BEFORE_DL=$(grep '^before_download=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-)
  fi
}

write_state() {
  local step="$1" volume="$2" before_dl="$3"
  {
    echo "step=$step"
    echo "volume=$volume"
    echo "before_download=$before_dl"
    echo "run_id=$RUN_ID"
  } > "$STATE_FILE"
}

# --- Resume: delete residual download folders ---
delete_residual_download() {
  local before="$1"
  for d in "$DOWNLOAD_DIR"/microns_*; do
    [[ -d "$d" ]] || continue
    local name=$(basename "$d")
    if [[ " $before " != *" $name "* ]]; then
      echo "Resume: removing residual $name"
      rm -rf "$d"
    fi
  done
}

# --- Run steps (tee only when interactive; SLURM captures stdout to PIPELINE_LOG) ---
_run_step() {
  if $USE_TEE; then
    "$@" 2>&1 | tee -a "$PIPELINE_LOG"
  else
    "$@" 2>&1
  fi
}

run_download() {
  local before_list=""
  for d in "$DOWNLOAD_DIR"/microns_*; do
    [[ -d "$d" ]] && before_list="$before_list $(basename "$d")"
  done
  write_state "download" "" "$before_list"

  _run_step bash "$CLOUD_DIR/download.sh"
  local vol
  vol=$(tail -1 "$PIPELINE_LOG" | tr -d '\n\r' | xargs)
  if [[ -z "$vol" || "$vol" == *"WARN"* || "$vol" == *"ERROR"* ]]; then
    write_state "download" "" "$before_list"
    return 1
  fi
  write_state "downsample" "$vol" "$before_list"
  export VOLUME="$vol"
  return 0
}

run_downsample() {
  export CLOUD_VOLUME="$VOLUME"
  _run_step bash "$CLOUD_DIR/downsample.sh" || return $?
  write_state "chunk" "$VOLUME" ""
}

run_chunk() {
  export CLOUD_VOLUME="$VOLUME"
  for d in "$PROJECT_ROOT/data/chunk"/${VOLUME}_* "$PROJECT_ROOT/data/train/em"/${VOLUME}_*; do
    [[ -d "$d" ]] && rm -rf "$d"
  done
  _run_step bash "$CLOUD_DIR/chunk.sh" || return $?
  write_state "degrade" "$VOLUME" ""
}

run_degrade() {
  _run_step stdbuf -oL -eL bash "$CLOUD_DIR/degrade.sh" --volume "$VOLUME" --resume --parallel || return $?
  write_state "noise" "$VOLUME" ""
}

run_noise() {
  _run_step stdbuf -oL -eL bash "$CLOUD_DIR/noise.sh" --volume "$VOLUME" --resume || return $?
  write_state "interpolate" "$VOLUME" ""
}

run_interpolate() {
  export CLOUD_VOLUME="$VOLUME"
  _run_step stdbuf -oL -eL bash "$CLOUD_DIR/interpolate.sh" --volume "$VOLUME" --resume || return $?
  write_state "complete" "$VOLUME" ""
}

# --- Main ---
read_state
echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') STEP=$STEP VOLUME=$VOLUME" >> "$DEBUG_LOG"

if [[ "$STEP" == "download" ]]; then
  echo "=== Resume: cleaning residual download ==="
  delete_residual_download "$BEFORE_DL"
  STEP=""
fi

if [[ -z "$STEP" || "$STEP" == "complete" ]]; then
  rm -f "$STATE_FILE"
  echo "=== [1/6] Download ==="
  run_download || exit $?
  read_state
fi

[[ -n "$VOLUME" ]] || { echo "ERROR: no volume"; exit 1; }
echo "Processing volume: $VOLUME"

if [[ "$STEP" == "downsample" ]]; then
  echo "=== [2/6] Downsample ==="
  run_downsample || exit $?
  read_state
fi

if [[ "$STEP" == "chunk" ]]; then
  echo "=== [3/6] Chunk ==="
  run_chunk || exit $?
  read_state
fi

if [[ "$STEP" == "degrade" ]]; then
  echo "=== [4/6] Degrade (Job Array parallel) ==="
  run_degrade || { echo "DEBUG: degrade step failed, check logs/degrade/<timestamp>/ (latest run dir)"; exit $?; }
  read_state
fi

if [[ "$STEP" == "noise" ]]; then
  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') About to run_noise" >> "$DEBUG_LOG"
  echo "=== [5/6] Noise (parallel) ==="
  run_noise || { r=$?; echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') run_noise returned $r" >> "$DEBUG_LOG"; exit $r; }
  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') run_noise OK" >> "$DEBUG_LOG"
  read_state
fi

if [[ "$STEP" == "interpolate" ]]; then
  echo "=== [6/6] Interpolate (parallel) ==="
  run_interpolate || exit $?
fi

echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') Pipeline complete" >> "$DEBUG_LOG"
echo "=== Pipeline complete ==="
echo "Log: $PIPELINE_LOG"