#!/bin/bash
# degrade.sh - Run NAOMi simulation on chunk data (pure bash).
# Usage: degrade.sh [--volume NAME] [--resume] [--parallel]
#   --volume: only process chunks matching microns_{volume}_* (e.g. microns_864691135430623536_0001)
#   --resume: skip chunks that already have output
#   --parallel: run PARALLEL_JOBS chunks in parallel
#
# Output: like local - MATLAB writes [MAIN], [STEP 1], [BRIDGE] etc. to main log for immediate visibility.
# Debug fallback: on errors, check /tmp/degrade_debug.log

set -e
# DEBUG: capture raw args before parsing (for troubleshooting)
_DEGRADE_ARGS_SAVED="$*"
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
LOG_FILE="$DEGRADE_RUN_DIR/degrade_${TS}.log"

log() {
  local msg="$1"
  local line="[$(date +%H:%M:%S)] $msg"
  echo "$line" | tee -a "$LOG_FILE"
}

if [[ ! -d "$CHUNK_DIR" ]]; then
  log "ERROR: chunk dir not found: $CHUNK_DIR"
  exit 1
fi

# Resolve folders to process
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
  if [[ -n "$COUNT" ]]; then
    FOLDERS=("${FOLDERS[@]:0:$COUNT}")
  fi
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
  if [[ -n "$COUNT" ]]; then
    FOLDERS=("${FOLDERS[@]:0:$COUNT}")
  fi
fi

# Resume: filter out chunks that already have output
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
log "Degrade pipeline started, log: $LOG_FILE"
log "DEBUG: degrade.sh invoked with args: $_DEGRADE_ARGS_SAVED"
log "Chunk dir: $CHUNK_DIR"
log "Output dir: $OUTPUT_DIR"
log "Folders to process: ${#FOLDERS[@]}"
if $PARALLEL; then
  log "DEBUG: parallel mode, N=$PARALLEL_JOBS"
else
  log "DEBUG: sequential mode (no --parallel), one chunk at a time"
fi
log "Debug fallback: /tmp/degrade_debug.log"
echo "[$(date +%H:%M:%S)] degrade.sh args=$_DEGRADE_ARGS_SAVED" >> /tmp/degrade_debug.log 2>/dev/null || true

command -v module &>/dev/null && module load matlab/R2024a 2>/dev/null || true

OK_COUNT=0
mkdir -p "$(dirname "$MANIFEST_DEGRADE")"

# DEBUG: verify script exists and paths
log "DEBUG: CLOUD_DIR=$CLOUD_DIR"
log "DEBUG: degrade_one_chunk.sh exists? $([ -f "$CLOUD_DIR/degrade_one_chunk.sh" ] && echo yes || echo no)"
log "DEBUG: LOG_FILE=$LOG_FILE"

# Parallel: 4 array tasks, each task runs 4 chunks in parallel (128G, 8 CPU per task)
if $PARALLEL; then
  CHUNKS_PER_TASK=4
  N_TASKS=$(( (${#FOLDERS[@]} + CHUNKS_PER_TASK - 1) / CHUNKS_PER_TASK ))
  CHUNKS_FILE="$DEGRADE_RUN_DIR/chunks_${TS}.txt"
  printf '%s\n' "${FOLDERS[@]}" > "$CHUNKS_FILE"
  CHUNKS_FILE="$(cd "$(dirname "$CHUNKS_FILE")" && pwd)/$(basename "$CHUNKS_FILE")"
  DEGRADE_RUN_DIR_ABS="$(cd "$DEGRADE_RUN_DIR" && pwd)"
  log "DEBUG: Job Array mode: $N_TASKS tasks, 4 chunks per task (128G, 8 CPU each)"
  log "Submitting array job: sbatch --array=1-$N_TASKS degrade_array.sh (logs in $DEGRADE_RUN_DIR_ABS)"

  ARRAY_JOB_ID=$(sbatch --parsable --export=ALL,CHUNKS_FILE="$CHUNKS_FILE",CLOUD_DIR="$CLOUD_DIR",DEGRADE_RUN_DIR="$DEGRADE_RUN_DIR_ABS" \
    --output="$DEGRADE_RUN_DIR_ABS/array_%A_%a.log" --error="$DEGRADE_RUN_DIR_ABS/array_%A_%a.log" \
    --array="1-$N_TASKS" "$CLOUD_DIR/degrade_array.sh")
  log "Array job ID: $ARRAY_JOB_ID"

  while true; do
    if ! squeue -j "$ARRAY_JOB_ID" -h 2>/dev/null | grep -q .; then
      break
    fi
    sleep 30
  done

  # One row per array task: use -X (allocation only) to avoid counting sub-steps
  SACCT_OUT=$(sacct -j "$ARRAY_JOB_ID" -X --format=JobID,State,ExitCode -n -P 2>/dev/null || true)
  if [[ -z "$SACCT_OUT" || ! "$SACCT_OUT" =~ [0-9] ]]; then
    # Fallback: no -X or empty; use full sacct and count only array task rows (e.g. 18071479_1 .. 18071479_4)
    SACCT_OUT=$(sacct -j "$ARRAY_JOB_ID" --format=JobID,State,ExitCode -n -P 2>/dev/null | grep -E "^${ARRAY_JOB_ID}_[0-9]+" || true)
  fi
  FAILED=$(echo "$SACCT_OUT" | grep -c -E 'FAILED|CANCELLED|NODE_FAIL' || true)
  OK_TASKS=$(echo "$SACCT_OUT" | grep -c COMPLETED || true)
  log "Finished: $OK_TASKS/$N_TASKS array tasks succeeded (array $ARRAY_JOB_ID)"
  log "DEBUG: sacct FAILED=$FAILED OK_TASKS=$OK_TASKS N_TASKS=$N_TASKS"
  { echo "DEBUG: sacct -X output:"; echo "$SACCT_OUT"; } >> "$LOG_FILE"
  rm -f "$CHUNKS_FILE"
  if [[ ${FAILED:-0} -gt 0 ]]; then
    log "Exiting: sacct reported failure count=$FAILED"
    exit 1
  fi
  if [[ $OK_TASKS -lt $N_TASKS ]]; then
    log "Exiting: completed tasks $OK_TASKS < $N_TASKS"
    exit 1
  fi
else
  for i in "${!FOLDERS[@]}"; do
    chunk_path="${FOLDERS[$i]}"
    idx=$((i + 1))
    total=${#FOLDERS[@]}
    log "--- [$idx/$total] $(basename "$chunk_path") ---"
    log "DEBUG: calling degrade_one_chunk.sh (sequential, foreground)"
    if bash "$CLOUD_DIR/degrade_one_chunk.sh" "$chunk_path" "$LOG_FILE"; then
      ((OK_COUNT++)) || true
      log "OK: $(basename "$chunk_path")"
    else
      log "FAIL: $(basename "$chunk_path") exit=$?"
      log "DEBUG: check /tmp/degrade_debug.log for details"
    fi
  done
fi

$PARALLEL || log "Finished: $OK_COUNT/${#FOLDERS[@]} succeeded"
$PARALLEL || [[ $OK_COUNT -lt ${#FOLDERS[@]} ]] && exit 1
