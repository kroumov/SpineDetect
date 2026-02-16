#!/bin/bash -l
#SBATCH --job-name=microns-ds
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-1%25
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ----------------------------
# User-configurable paths
# ----------------------------
PROJECT_DIR="/home/$USER/project"               # contains tasks.tsv, params.json, etc.
TASKS_TSV="${PROJECT_DIR}/tasks.tsv"
SIF="/home/$USER/containers/microns_downsample.sif"

# Secrets (kept private with 700/600 perms)
CAVE_SECRETS_DIR="/home/$USER/.cloudvolume"    # expects secrets under ~/.cloudvolume/secrets/
GCP_SECRETS_DIR="/home/$USER/.secrets"
GCP_KEY="${GCP_SECRETS_DIR}/gcp-uploader-sa.json"

# Where outputs should land on the shared filesystem (optional; depends on your python)
OUT_SHARED="${PROJECT_DIR}/out"

# ----------------------------
# Load container runtime
# ----------------------------
module load singularity

mkdir -p "${PROJECT_DIR}/logs" "${OUT_SHARED}"

# ----------------------------
# Validate inputs early
# ----------------------------
if [[ ! -f "${TASKS_TSV}" ]]; then
  echo "ERROR: tasks file not found: ${TASKS_TSV}" >&2
  exit 2
fi
if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: container SIF not found: ${SIF}" >&2
  exit 2
fi
if [[ ! -f "${GCP_KEY}" ]]; then
  echo "ERROR: GCP key not found: ${GCP_KEY}" >&2
  exit 2
fi

# ----------------------------
# Select task line
#
# Assumption: tasks.tsv is "body only" (no header/comment lines).
# If you have a header, either remove it or adjust indexing below.
# ----------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID}"

line="$(sed -n "${TASK_ID}p" "${TASKS_TSV}" || true)"
if [[ -z "${line}" ]]; then
  echo "ERROR: No line ${TASK_ID} in ${TASKS_TSV}" >&2
  exit 3
fi

# Parse: ix iy iz params_path
# Using read to split on whitespace/tabs
read -r ix iy iz params_path <<< "${line}"

if [[ -z "${ix}" || -z "${iy}" || -z "${iz}" || -z "${params_path}" ]]; then
  echo "ERROR: Malformed line ${TASK_ID}: '${line}'" >&2
  exit 4
fi

if [[ ! -f "${params_path}" ]]; then
  echo "ERROR: params file not found: ${params_path}" >&2
  exit 5
fi

echo "============================================================"
echo "Job:         ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:        $(hostname)"
echo "Tile:        ix=${ix}, iy=${iy}, iz=${iz}"
echo "Params:      ${params_path}"
echo "============================================================"

# ----------------------------
# Fast scratch (node-local)
# ----------------------------
WORKDIR="${TMPDIR:-/tmp}/microns_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${WORKDIR}"

# ----------------------------
# Auth: ADC for Google Cloud
# ----------------------------
export GOOGLE_APPLICATION_CREDENTIALS="${GCP_KEY}"

# Optional: avoid accidental thread oversubscription by native libs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ----------------------------
# Run container
#
# Bind mounts:
# - project dir -> /work
# - ~/.cloudvolume -> /home/$USER/.cloudvolume  (for cave-secret.json auto-discovery)
# - ~/.secrets -> /home/$USER/.secrets (read-only)
# - $WORKDIR -> /tmp/work (fast local scratch inside container)
#
# Env passthrough:
# - GOOGLE_APPLICATION_CREDENTIALS
# ----------------------------
singularity exec \
  --cleanenv \
  --env "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}" \
  --bind "${PROJECT_DIR}:/work" \
  --bind "${CAVE_SECRETS_DIR}:/home/${USER}/.cloudvolume" \
  --bind "${GCP_SECRETS_DIR}:/home/${USER}/.secrets:ro" \
  "${SIF}" \
  python /app/app.py \
    --params "${params_path}" \
    --ix "${ix}" --iy "${iy}" --iz "${iz}" \
    --out-dir "/work/out" \
    --silent

echo "Done. Local scratch: ${WORKDIR}"
