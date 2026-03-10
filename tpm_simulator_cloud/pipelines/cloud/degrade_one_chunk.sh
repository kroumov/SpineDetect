#!/bin/bash
# degrade_one_chunk.sh - Run NAOMi for one chunk (standalone, no export -f).
# Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>
#   chunk_path: absolute path to chunk folder (e.g. .../chunk/microns_xxx_0001)
#   main_log_path: where MATLAB writes [MAIN], [STEP 1] etc. (same file as local)
#
# Debug fallback: on error, also appends to /tmp/degrade_debug.log

# DEBUG: first line - write to main_log immediately (bypass stdout buffering)
_ts="$(date +%H:%M:%S)"
_msg="[$_ts] degrade_one_chunk.sh ENTERED arg1=$1 arg2=$2"
[[ -n "$2" ]] && echo "$_msg" >> "$2" || echo "$_msg" >> /tmp/degrade_debug.log
echo "$_msg"

set -e
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLOUD_DIR/config.sh"

DBG_LOG="${DEGRADE_DEBUG_LOG:-/tmp/degrade_debug.log}"
debug() { echo "[$(date +%H:%M:%S)] $*" >> "$DBG_LOG"; }

chunk_path="${1:?Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>}"
main_log="${2:?Usage: degrade_one_chunk.sh <chunk_path> <main_log_path>}"

debug "degrade_one_chunk: chunk_path=$chunk_path main_log=$main_log"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: sourced config, chunk=$(basename "$chunk_path")" >> "$main_log"

# DEBUG: load MATLAB on compute nodes (module may not exist on login node)
if command -v module &>/dev/null; then
  debug "degrade_one_chunk: loading matlab module"
  if module load matlab/R2024a 2>>"$DBG_LOG"; then
    debug "degrade_one_chunk: module load matlab OK"
  else
    debug "degrade_one_chunk: module load matlab FAILED"
  fi
else
  debug "degrade_one_chunk: module command not found, assuming matlab in PATH"
fi
# DEBUG: verify MATLAB availability
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

# MATLAB writes to main_log via manifest; also capture stdout/stderr to main_log
debug "degrade_one_chunk: invoking MATLAB for $chunk_name"
$MATLAB_CMD "addpath('scripts'); run_from_path_files('$mf_rel')" >> "$main_log" 2>&1
ret=$?
rm -f "$manifest_tmp"
cd "$CLOUD_DIR" || true

# DEBUG: always log MATLAB exit status for troubleshooting
debug "degrade_one_chunk: MATLAB exit=$ret for $chunk_name"
echo "[$(date +%H:%M:%S)] degrade_one_chunk: MATLAB exit=$ret for $chunk_name" >> "$main_log"
[[ $ret -ne 0 ]] && debug "degrade_one_chunk: FAILED exit=$ret"
exit $ret
