#!/bin/bash
# degrade_one_chunk.sh - Run NAOMi for one chunk (standalone, no export -f).
# Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>
#   chunk_path: absolute path to chunk folder (e.g. .../chunk/microns_xxx_0001)
#   main_log_path: where MATLAB writes [MAIN], [STEP 1] etc.
#
# Debug: DEGRADE_DEBUG_LOG (set by degrade.sh/degrade_array.sh) -> logs/degrade/<timestamp>/
# Fallback: main_log dir if DEGRADE_DEBUG_LOG not set

set -e
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLOUD_DIR/config.sh"

chunk_path="${1:?Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>}"
main_log="${2:?Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>}"

# Debug log: use DEGRADE_DEBUG_LOG, else fallback to logs/degrade/ under main_log's dir
if [[ -n "${DEGRADE_DEBUG_LOG:-}" ]]; then
  DBG_LOG="$DEGRADE_DEBUG_LOG"
else
  _log_dir="$(dirname "$main_log")"
  DBG_LOG="${_log_dir}/debug_$(basename "$main_log" .log).log"
fi
debug() { echo "[$(date +%H:%M:%S)] $*" >> "$DBG_LOG"; }

_ts="$(date +%H:%M:%S)"
_msg="[$_ts] degrade_one_chunk.sh ENTERED arg1=$1 arg2=$2"
echo "$_msg" >> "$main_log"
echo "$_msg"

debug "degrade_one_chunk: chunk_path=$chunk_path main_log=$main_log"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: sourced config, chunk=$(basename "$chunk_path")" >> "$main_log"

#if command -v module &>/dev/null; then
#  debug "degrade_one_chunk: loading matlab module"
#  module load matlab/R2024a 2>>"$DBG_LOG" || true
#else
#  debug "degrade_one_chunk: module command not found, assuming matlab in PATH"
#fi
if command -v module &>/dev/null; then
  debug "degrade_one_chunk: loading matlab module"
  module load matlab/R2024a 2>>"$DBG_LOG" || true
else
  debug "degrade_one_chunk: module not found, setting MATLAB manually"

  MATLAB_BIN="/cis/local/linux/MATLAB/R2022a/bin/matlab"

  if [ -x "$MATLAB_BIN" ]; then
    export PATH="$(dirname "$MATLAB_BIN"):$PATH"
    debug "degrade_one_chunk: MATLAB added to PATH via $MATLAB_BIN"
  else
    echo "ERROR: MATLAB not found at $MATLAB_BIN" >&2
    exit 1
  fi
fi

_matlab_path=$(command -v matlab 2>/dev/null || true)
debug "degrade_one_chunk: which matlab=$_matlab_path"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: MATLAB_CMD=$MATLAB_CMD matlab_path=${_matlab_path:-NOT_FOUND}" >> "$main_log"
[[ -z "$_matlab_path" ]] && { debug "degrade_one_chunk: matlab not found, aborting"; echo "[$(date +%H:%M:%S)] degrade_one_chunk: ERROR matlab not in PATH" >> "$main_log"; exit 1; }

chunk_name=$(basename "$chunk_path")
OUTPUT_DIR="$PROJECT_ROOT/data/degrade"
manifest_tmp="$MODULE_DIR/tmp/degrade_args_${chunk_name}.txt"
mkdir -p "$(dirname "$manifest_tmp")"

chunk_abs="$(cd "$chunk_path" && pwd)"
output_abs="$(cd "$OUTPUT_DIR" && pwd)"
log_abs="$(cd "$(dirname "$main_log")" && pwd)/$(basename "$main_log")"
chunk_abs="${chunk_abs//\\/\/}"
output_abs="${output_abs//\\/\/}"
log_abs="${log_abs//\\/\/}"

printf '%s\n%s\n%s\n' "$chunk_abs" "$output_abs" "$log_abs" > "$manifest_tmp"
mf_rel="tmp/$(basename "$manifest_tmp")"

debug "degrade_one_chunk: about to run MATLAB, manifest=$manifest_tmp"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: about to run MATLAB" >> "$main_log"
cd "$MODULE_DIR" || { debug "degrade_one_chunk: cd MODULE_DIR failed"; exit 1; }

debug "degrade_one_chunk: invoking MATLAB for $chunk_name"
$MATLAB_CMD "addpath('scripts'); run_from_path_files('$mf_rel')" >> "$main_log" 2>&1
ret=$?
rm -f "$manifest_tmp"
cd "$CLOUD_DIR" || true

debug "degrade_one_chunk: MATLAB exit=$ret for $chunk_name"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: MATLAB exit=$ret for $chunk_name" >> "$main_log"
[[ $ret -ne 0 ]] && debug "degrade_one_chunk: FAILED exit=$ret"
exit $ret