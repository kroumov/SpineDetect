#!/bin/bash
# config.sh - Cloud pipeline configuration. Source this before running steps.
# Override PROJECT_ROOT if tpm_simulator is elsewhere on the server.

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CLOUD_DIR/../.." && pwd)}"

# Data paths (relative to PROJECT_ROOT)
CHUNK_DIR="$PROJECT_ROOT/data/chunk"
DEGRADE_DIR="$PROJECT_ROOT/data/degrade"
OUTPUT_DIR="$PROJECT_ROOT/data/noise"
LOGS_DIR="$CLOUD_DIR/logs"
MODULE_DIR="$PROJECT_ROOT/simulation_module_v4"
NOISE_MODULE_DIR="$PROJECT_ROOT/noise_model"
MANIFEST_DEGRADE="$MODULE_DIR/tmp/degrade_args.txt"
MANIFEST_NOISE="$NOISE_MODULE_DIR/noise_args.txt"

# Python scripts (from local pipeline)
LOCAL_DIR="$PROJECT_ROOT/pipelines/local"
DOWNLOAD_PY="$LOCAL_DIR/download.py"
DOWNSAMPLE_PY="$LOCAL_DIR/downsample.py"
CHUNK_PY="$LOCAL_DIR/chunk.py"
INTERPOLATE_PY="$LOCAL_DIR/interpolate.py"

# MATLAB (run before degrade/noise)
# module load matlab/R2024a
MATLAB_CMD="matlab -batch"

# Noise params (match pipelines/local/noise.py)
NOISE_MU=500
NOISE_SIGMA=50
NOISE_MU0=0
NOISE_SIGMA0=0.1
NOISE_DARKCOUNT=0.01
NOISE_BLEEDP=0.01
NOISE_BLEEDW=0.05

# Run state (for checkpoint/resume)
STATE_FILE="$CLOUD_DIR/run_state.json"

# Parallel jobs for degrade/noise/interpolate (chunk count)
PARALLEL_JOBS=16

# SLURM defaults (for sbatch jobs; edit as needed)
# SBATCH_PARTITION=educluster
# SBATCH_CPUS=8
# SBATCH_MEM=128G
# SBATCH_TIME=24:00:00
