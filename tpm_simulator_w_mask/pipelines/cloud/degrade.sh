#!/bin/bash
# degrade.sh - Run NAOMi simulation on chunk data (pure bash).
# Usage: degrade.sh [--volume NAME] [--resume] [--parallel]
#   --volume: only process chunks matching microns_{volume}_*
#   --resume: skip chunks that already have output
#   --parallel: submit array job (4 chunks per task)
#
# All output: logs/degrade/<timestamp>/

set -e
source "$(dirname "$0")/config.sh"

FOLDER=""
COUNT=""
VOLUME=""
RESUME=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder|-f) FOLDER="$2"; shift 2 ;;
    --count|-n)  COUNT="$2"; shift 2 ;;
    --volume|-v)  VOLUME="$2"; shift 2 ;;
    --resume)    RESUME=true; shift ;;
    --parallel)  PARALLEL=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
DEGRADE_RUN_DIR="$LOGS_DIR/degrade/$TS"
mkdir -p "$DEGRADE_RUN_DIR"
LOG_FILE="$DEGRADE_RUN_DIR/degrade.log"

log() {
  echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

if [[ ! -d "$CHUNK_DIR" ]]; then
  log "ERROR: chunk dir not found: $CHUNK_DIR"
  exit 1
fi

if [[ -n "$FOLDER" ]]; then
  if [[ "$FOLDER" == /* ]]; then
    [[ -d "$FOLDER" ]] || { log "ERROR: folder not found: $FOLDER"; exit 1; }
    FOLDERS=("$FOLDER")
  else
    CANDIDATE="$CHUNK_DIR/$FOLDER"
    [[ -d "$CANDIDATE" ]] || { log "ERROR: folder not found: $CANDIDATE"; exit 1; }
    FOLDERS=("$CANDIDATE")
  fi
  [[ -z "$COUNT" ]] && COUNT=1
elif [[ -n "$VOLUME" ]]; then
  FOLDERS=()
  prefix="${VOLUME}_"
  for d in "$CHUNK_DIR"/${prefix}*; do
    [[ -d "$d" ]] && FOLDERS+=("$d")
  done
  FOLDERS=($(printf '%s\n' "${FOLDERS[@]}" | sort))
  if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    log "No chunks matching ${VOLUME}_* in chunk dir."
    exit 0
  fi
  [[ -n "$COUNT" ]] && FOLDERS=("${FOLDERS[@]:0:$COUNT}")
else
  FOLDERS=()
  for d in "$CHUNK_DIR"/microns_*_*; do
    [[ -d "$d" ]] && FOLDERS+=("$d")
  done
  FOLDERS=($(printf '%s\n' "${FOLDERS[@]}" | sort))
  if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    log "No microns_*_* folders in chunk dir."
    exit 0
  fi
  [[ -n "$COUNT" ]] && FOLDERS=("${FOLDERS[@]:0:$COUNT}")
fi

OUTPUT_DIR="$PROJECT_ROOT/data/degrade"
if $RESUME; then
  TODO=()
  for chunk_path in "${FOLDERS[@]}"; do
    name=$(basename "$chunk_path")
    out_folder="$OUTPUT_DIR/$name"
    if [[ ! -d "$out_folder" ]] || ! ls "$out_folder"/neurons_*.tiff 1>/dev/null 2>&1; then
      TODO+=("$chunk_path")
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
log "Degrade started, run dir: $DEGRADE_RUN_DIR"
log "Chunk dir: $CHUNK_DIR, output: $OUTPUT_DIR, folders: ${#FOLDERS[@]}"

command -v module &>/dev/null && module load matlab/R2024a 2>/dev/null || true

OK_COUNT=0
mkdir -p "$(dirname "$MANIFEST_DEGRADE")"

if $PARALLEL; then
  CHUNKS_PER_TASK=4
  N_TASKS=$(( (${#FOLDERS[@]} + CHUNKS_PER_TASK - 1) / CHUNKS_PER_TASK ))
  CHUNKS_FILE="$DEGRADE_RUN_DIR/chunks.txt"
  printf '%s\n' "${FOLDERS[@]}" > "$CHUNKS_FILE"
  CHUNKS_FILE="$(cd "$(dirname "$CHUNKS_FILE")" && pwd)/$(basename "$CHUNKS_FILE")"
  DEGRADE_RUN_DIR_ABS="$(cd "$DEGRADE_RUN_DIR" && pwd)"

  ARRAY_JOB_ID=$(sbatch --parsable --export=ALL,CHUNKS_FILE="$CHUNKS_FILE",CLOUD_DIR="$CLOUD_DIR",DEGRADE_RUN_DIR="$DEGRADE_RUN_DIR_ABS" \
    --output="$DEGRADE_RUN_DIR_ABS/array_%a.log" --error="$DEGRADE_RUN_DIR_ABS/array_%a.log" \
    --array="1-$N_TASKS" "$CLOUD_DIR/degrade_array.sh")
  log "Array job $ARRAY_JOB_ID ($N_TASKS tasks), waiting..."

  while true; do
    if ! squeue -j "$ARRAY_JOB_ID" -h 2>/dev/null | grep -q .; then
      break
    fi
    sleep 30
  done

  log "Array job finished, proceeding to next step"
  rm -f "$CHUNKS_FILE"
else
  export DEGRADE_DEBUG_LOG="$DEGRADE_RUN_DIR/debug.log"
  for i in "${!FOLDERS[@]}"; do
    chunk_path="${FOLDERS[$i]}"
    idx=$((i + 1))
    total=${#FOLDERS[@]}
    log "--- [$idx/$total] $(basename "$chunk_path") ---"
    if bash "$CLOUD_DIR/degrade_one_chunk.sh" "$chunk_path" "$LOG_FILE"; then
      ((OK_COUNT++)) || true
      log "OK: $(basename "$chunk_path")"
    else
      log "FAIL: $(basename "$chunk_path")"
    fi
  done
  log "Finished: $OK_COUNT/${#FOLDERS[@]} succeeded"
  [[ $OK_COUNT -lt ${#FOLDERS[@]} ]] && exit 1
fi