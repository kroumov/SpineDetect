#!/bin/bash
# noise.sh - Add Poisson-Gaussian + pixel bleed to degrade outputs (pure bash).
# Usage: noise.sh [--volume NAME] [--resume] [--parallel]
#   --volume: only process chunks matching microns_{volume}_*
#   --resume: skip chunks that already have output
#   --parallel: run NOISE_PARALLEL_JOBS chunks in parallel
#
# When submitting from home with data on scratch4, set data root before sbatch:
#   export DATA_ROOT=/scratch4/en580/en580-syan28/tpm_simulator

set -e
source "$(dirname "$0")/config.sh"

# Prefer DEBUG_LOG (pipeline) or DATA_ROOT/logs (writable on compute nodes); fallback to LOGS_DIR
NOISE_DEBUG_LOG="${DEBUG_LOG:-}"
[[ -z "$NOISE_DEBUG_LOG" && -n "${DATA_ROOT:-}" ]] && NOISE_DEBUG_LOG="$DATA_ROOT/logs/noise_debug.log"
[[ -z "$NOISE_DEBUG_LOG" ]] && NOISE_DEBUG_LOG="$LOGS_DIR/noise_debug.log"
mkdir -p "$(dirname "$NOISE_DEBUG_LOG")" 2>/dev/null || true
echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh START DEGRADE_DIR=$DEGRADE_DIR" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true

COUNT=""
VOLUME=""
RESUME=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count|-n)  COUNT="$2"; shift 2 ;;
    --volume|-v) VOLUME="$2"; shift 2 ;;
    --resume)    RESUME=true; shift ;;
    --parallel)  PARALLEL=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOGS_DIR/noise"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/noise/noise_${TS}.log"

log() {
  local msg="$1"
  local line="[$(date +%H:%M:%S)] $msg"
  echo "$line" | tee -a "$LOG_FILE"
}

if [[ ! -d "$DEGRADE_DIR" ]]; then
  log "ERROR: degrade dir not found: $DEGRADE_DIR"
  exit 1
fi

# Resolve folders to process
if [[ -n "$VOLUME" ]]; then
  prefix="${VOLUME}_"
  FOLDERS=()
  for d in "$DEGRADE_DIR"/${prefix}*; do
    [[ -d "$d" ]] && FOLDERS+=("$d")
  done
  FOLDERS=($(printf '%s\n' "${FOLDERS[@]}" | sort))
else
  FOLDERS=()
  for d in "$DEGRADE_DIR"/microns_*_*; do
    [[ -d "$d" ]] && FOLDERS+=("$d")
  done
  FOLDERS=($(printf '%s\n' "${FOLDERS[@]}" | sort))
fi

[[ -n "$COUNT" ]] && FOLDERS=("${FOLDERS[@]:0:$COUNT}")

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  log "ERROR: No folders to process (degrade may have failed)."
  exit 1
fi

# Resume: filter out chunks that already have output
if $RESUME; then
  TODO=()
  for df in "${FOLDERS[@]}"; do
    name=$(basename "$df")
    out_folder="$OUTPUT_DIR/$name"
    if [[ ! -d "$out_folder" ]] || ! ls "$out_folder"/neurons_*.tiff 1>/dev/null 2>&1; then
      TODO+=("$df")
    fi
  done
  FOLDERS=("${TODO[@]}")
  if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    log "Resume: all chunks already done."
    exit 0
  fi
  log "Resume: ${#FOLDERS[@]} chunks remaining"
fi

mkdir -p "$OUTPUT_DIR"
log "Noise pipeline started, log: $LOG_FILE"
log "Degrade dir: $DEGRADE_DIR"
log "Output dir: $OUTPUT_DIR"
log "Folders to process: ${#FOLDERS[@]}"
$PARALLEL && log "Parallel: ${NOISE_PARALLEL_JOBS:-$PARALLEL_JOBS} jobs"

#command -v module &>/dev/null && module load matlab/R2024a 2>/dev/null || true
if command -v module &>/dev/null; then
  log "noise: loading matlab module"
  module load matlab/R2024a 2>>"$DBG_LOG" || true
else
  log "noise: module not found, setting MATLAB manually"

  MATLAB_BIN="/cis/local/linux/MATLAB/R2022a/bin/matlab"

  if [ -x "$MATLAB_BIN" ]; then
    export PATH="$(dirname "$MATLAB_BIN"):$PATH"
    log "noise: MATLAB added to PATH via $MATLAB_BIN"
  else
    echo "ERROR: MATLAB not found at $MATLAB_BIN" >&2
    exit 1
  fi
fi

OK_COUNT=0

process_one_chunk() {
  local degrade_folder="$1"
  local tiff_path="" f
  for f in "$degrade_folder"/neurons_*.tiff "$degrade_folder"/neurons_*.tif; do
    [[ -f "$f" ]] && { tiff_path="$f"; break; }
  done
  [[ -z "$tiff_path" ]] && return 1

  local chunk_name=$(basename "$degrade_folder")
  local manifest_tmp="$NOISE_MODULE_DIR/noise_args_${chunk_name}.txt"
  local degrade_info_path="$degrade_folder/degrade_info.txt"
  local out_folder="$OUTPUT_DIR/$chunk_name"
  mkdir -p "$out_folder"

  local in_abs out_dir_abs
  in_abs="$(cd "$(dirname "$tiff_path")" && pwd)/$(basename "$tiff_path")"
  out_dir_abs="$(cd "$out_folder" && pwd)"
  in_abs="${in_abs//\\/\/}"
  out_dir_abs="${out_dir_abs//\\/\/}"

  printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' \
    "$in_abs" "$out_dir_abs" \
    "$NOISE_MU" "$NOISE_SIGMA" "$NOISE_MU0" "$NOISE_SIGMA0" \
    "$NOISE_DARKCOUNT" "$NOISE_BLEEDP" "$NOISE_BLEEDW" \
    > "$manifest_tmp"

  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') process_one_chunk START $chunk_name" >> "${NOISE_DEBUG_LOG:-/dev/null}" 2>/dev/null || true
  cd "$NOISE_MODULE_DIR" || exit 1
  $MATLAB_CMD "addpath('.'); run_noise_from_paths('noise_args_${chunk_name}.txt');" >> "$LOG_FILE" 2>&1
  local ret=$?
  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') process_one_chunk DONE $chunk_name ret=$ret" >> "${NOISE_DEBUG_LOG:-/dev/null}" 2>/dev/null || true
  rm -f "$manifest_tmp"
  cd "$CLOUD_DIR" || true
  [[ $ret -ne 0 ]] && return $ret

  local degrade_content=""
  [[ -f "$degrade_info_path" ]] && degrade_content="$(cat "$degrade_info_path")"
  local noise_section="
--- Noise Model (noise_config) ---
mu:               $NOISE_MU
sigma:            $NOISE_SIGMA
mu0:              $NOISE_MU0
sigma0:           $NOISE_SIGMA0
darkcount:        $NOISE_DARKCOUNT
bleedp:           $NOISE_BLEEDP
bleedw:           $NOISE_BLEEDW
========================================"
  echo "${degrade_content}${noise_section}" > "$out_folder/noise_info.txt"
  return 0
}

export -f process_one_chunk
export OUTPUT_DIR NOISE_MODULE_DIR LOG_FILE CLOUD_DIR MATLAB_CMD
export NOISE_MU NOISE_SIGMA NOISE_MU0 NOISE_SIGMA0 NOISE_DARKCOUNT NOISE_BLEEDP NOISE_BLEEDW
export NOISE_DEBUG_LOG

if $PARALLEL; then
  N=${NOISE_PARALLEL_JOBS:-$PARALLEL_JOBS}
  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh parallel N=$N nfolders=${#FOLDERS[@]}" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true
  running=0
  wait_ret=0
  for df in "${FOLDERS[@]}"; do
    while (( running >= N )); do
      (wait -n 2>/dev/null || wait) || true
      (( running-- )) || true
    done
    process_one_chunk "$df" && ((OK_COUNT++)) || true &
    (( running++ ))
  done
  set +e
  wait
  wait_ret=$?
  set -e
  echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh wait done wait_ret=$wait_ret OK_COUNT=$OK_COUNT" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true
else
  for i in "${!FOLDERS[@]}"; do
    df="${FOLDERS[$i]}"
    idx=$((i + 1))
    total=${#FOLDERS[@]}
    log "--- [$idx/$total] $(basename "$df") ---"
    if process_one_chunk "$df"; then
      ((OK_COUNT++)) || true
      log "OK: $(basename "$df")"
    else
      log "FAIL: $(basename "$df")"
    fi
  done
fi

log "Finished: $OK_COUNT/${#FOLDERS[@]} succeeded"
echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh EXIT OK" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true
if $PARALLEL; then
  [[ ${wait_ret:-0} -ne 0 ]] && { echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh EXIT 1 wait_ret=$wait_ret" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true; exit 1; }
else
  [[ $OK_COUNT -lt ${#FOLDERS[@]} ]] && { echo "DEBUG $(date '+%Y-%m-%d %H:%M:%S') noise.sh EXIT 1 OK_COUNT=$OK_COUNT" >> "$NOISE_DEBUG_LOG" 2>/dev/null || true; exit 1; }
fi
exit 0