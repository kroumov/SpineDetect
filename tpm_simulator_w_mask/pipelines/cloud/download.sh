#!/bin/bash
# download.sh - Wrapper for pipelines/local/download.py
# Outputs CLOUD_VOLUME (volume name) via run_state.json; used for single-volume pipeline.
# On resume after failed download: deletes residual folders from this run, then re-downloads.

set -e
source "$(dirname "$0")/config.sh"
cd "$PROJECT_ROOT" || exit 1

# Activate venv (prefer one with cloudfiles for download)
for venv in "$PROJECT_ROOT/microns_env" "$PROJECT_ROOT/sample_random_volume/microns_env" "$PROJECT_ROOT/tpm_env"; do
  if [[ -f "$venv/bin/activate" ]]; then
    source "$venv/bin/activate"
    if python -c "import cloudfiles" 2>/dev/null; then
      break
    fi
  fi
done

# Enable real-time DEBUG output for download (block progress, errors, tracebacks)
export LOG_LEVEL=DEBUG

DOWNLOAD_DIR="$PROJECT_ROOT/data/download"
mkdir -p "$DOWNLOAD_DIR"

# Snapshot existing folders before download (for resume: detect residual)
EXISTING_BEFORE=""
for d in "$DOWNLOAD_DIR"/microns_*; do
  [[ -d "$d" ]] && EXISTING_BEFORE="$EXISTING_BEFORE $(basename "$d")"
done

python "$DOWNLOAD_PY"

# Detect new volume(s) - should be exactly 1
EXISTING_AFTER=""
for d in "$DOWNLOAD_DIR"/microns_*; do
  [[ -d "$d" ]] && EXISTING_AFTER="$EXISTING_AFTER $(basename "$d")"
done
# New = in AFTER but not in BEFORE (simple diff)
NEW_VOLUME=""
for v in $EXISTING_AFTER; do
  if [[ " $EXISTING_BEFORE " != *" $v "* ]]; then
    NEW_VOLUME="$v"
    break
  fi
done

if [[ -z "$NEW_VOLUME" ]]; then
  echo "WARN: No new microns_* folder detected after download"
  exit 1
fi

# Output for run_pipeline (writes to STATE_FILE via caller)
echo "$NEW_VOLUME"
