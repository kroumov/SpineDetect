#!/bin/bash
# interpolate.sh - Wrapper for pipelines/local/interpolate.py
# Usage: interpolate.sh [--volume NAME] [--resume] [--parallel]
#   --volume: only process chunks matching microns_{volume}_*
#   --resume: skip chunks that already have output
#   --parallel: run INTERPOLATE_PARALLEL_JOBS chunks in parallel

set -e
source "$(dirname "$0")/config.sh"
cd "$PROJECT_ROOT" || exit 1

VOLUME=""
RESUME=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volume|-v)  VOLUME="$2"; shift 2 ;;
    --resume)     RESUME=true; shift ;;
    --parallel)   PARALLEL=true; shift ;;
    *) shift ;;
  esac
done

# Activate venv (same logic as download.sh - include tpm_env)
for venv in "$PROJECT_ROOT/microns_env" "$PROJECT_ROOT/sample_random_volume/microns_env" "$PROJECT_ROOT/tpm_env"; do
  if [[ -f "$venv/bin/activate" ]]; then
    source "$venv/bin/activate"
    if python -c "import numpy" 2>/dev/null; then
      break
    fi
  fi
done

# Use DATA_ROOT when set (submit from home, data on scratch) so interpolate reads/writes same place as noise
NOISE_DIR="${DATA_ROOT:-$PROJECT_ROOT}/data/noise"
OUTPUT_DIR="${DATA_ROOT:-$PROJECT_ROOT}/data/train/tpm"
export CLOUD_NOISE_DIR="$NOISE_DIR"
export CLOUD_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$LOGS_DIR/interpolate"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/interpolate/interpolate_${TS}.log"

log() {
  echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

# Collect folders to process
if [[ -n "$VOLUME" ]]; then
  prefix="${VOLUME}_"
  FOLDERS=()
  for d in "$NOISE_DIR"/${prefix}*; do
    [[ -d "$d" ]] && FOLDERS+=("$d")
  done
  FOLDERS=($(printf '%s\n' "${FOLDERS[@]}" | sort))
else
  export CLOUD_VOLUME=""
  python "$INTERPOLATE_PY" "$@"
  exit $?
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  log "ERROR: No chunks matching ${VOLUME}_* in noise dir (noise may have failed)."
  exit 1
fi

# Resume: filter out chunks that already have output
if $RESUME; then
  TODO=()
  for nf in "${FOLDERS[@]}"; do
    name=$(basename "$nf")
    out_folder="$OUTPUT_DIR/$name"
    if [[ ! -d "$out_folder" ]] || ! ls "$out_folder"/neurons_*.tiff 1>/dev/null 2>&1; then
      TODO+=("$nf")
    fi
  done
  FOLDERS=("${TODO[@]}")
  if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    log "Resume: all chunks already done."
    exit 0
  fi
  log "Resume: ${#FOLDERS[@]} chunks remaining"
fi

log "Interpolate: ${#FOLDERS[@]} chunks"
$PARALLEL && log "Parallel: $PARALLEL_JOBS jobs"

run_one() {
  local folder="$1"
  local name=$(basename "$folder")
  export CLOUD_SINGLE_FOLDER="$name"
  python "$INTERPOLATE_PY" >> "$LOG_FILE" 2>&1
  return $?
}

export -f run_one
export INTERPOLATE_PY LOG_FILE CLOUD_SINGLE_FOLDER

OK_COUNT=0
if $PARALLEL; then
  N=${INTERPOLATE_PARALLEL_JOBS:-$PARALLEL_JOBS}
  running=0
  wait_ret=0
  for nf in "${FOLDERS[@]}"; do
    while (( running >= N )); do
      (wait -n 2>/dev/null || wait) || true
      (( running-- )) || true
    done
    run_one "$nf" && ((OK_COUNT++)) || true &
    (( running++ ))
  done
  set +e
  wait
  wait_ret=$?
  set -e
else
  for i in "${!FOLDERS[@]}"; do
    nf="${FOLDERS[$i]}"
    idx=$((i + 1))
    total=${#FOLDERS[@]}
    log "--- [$idx/$total] $(basename "$nf") ---"
    if run_one "$nf"; then
      ((OK_COUNT++)) || true
      log "OK: $(basename "$nf")"
    else
      log "FAIL: $(basename "$nf")"
    fi
  done
fi

log "Finished: $OK_COUNT/${#FOLDERS[@]} succeeded"
if $PARALLEL; then
  [[ ${wait_ret:-0} -ne 0 ]] && exit 1
else
  [[ $OK_COUNT -lt ${#FOLDERS[@]} ]] && exit 1
fi
exit 0