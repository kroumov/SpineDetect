#!/bin/bash
# downsample.sh - Wrapper for pipelines/local/downsample.py

set -e
source "$(dirname "$0")/config.sh"
cd "$PROJECT_ROOT" || exit 1

# Activate venv (same logic as download.sh - include tpm_env)
for venv in "$PROJECT_ROOT/microns_env" "$PROJECT_ROOT/sample_random_volume/microns_env" "$PROJECT_ROOT/tpm_env"; do
  if [[ -f "$venv/bin/activate" ]]; then
    source "$venv/bin/activate"
    if python -c "import numpy" 2>/dev/null; then
      break
    fi
  fi
done

python "$DOWNSAMPLE_PY"
