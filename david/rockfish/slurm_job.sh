#!/bin/bash -l
#SBATCH --job-name=microns-ds
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-10%25
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ----------------------------
# User-configurable paths
# ----------------------------
PROJECT_DIR="/scratch4/en580/dkopala1/vessels/"               # contains volume.json, jobs.csv
SIF="/home/$USER/singularity/microns-downsample_latest.sif"

# Secrets (kept private with 700/600 perms)
CAVE_SECRETS_DIR="/home/$USER/.cloudvolume"    # expects secrets under ~/.cloudvolume/secrets/

# ----------------------------
# Load container runtime
# ----------------------------
module load singularity

mkdir -p "${PROJECT_DIR}/logs"

# ----------------------------
# Validate inputs early
# ----------------------------
if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: container SIF not found: ${SIF}" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_DIR}/volume.json" ]]; then
  echo "ERROR: volume.json: ${PROJECT_DIR}/volume.json" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_DIR}/jobs.csv" ]]; then
  echo "ERROR: jobs.csv not found: ${PROJECT_DIR}/jobs.csv" >&2
  exit 2
fi


# ----------------------------
# Select task line
#
# Assumption: tasks.tsv is "body only" (no header/comment lines).
# If you have a header, either remove it or adjust indexing below.
# ----------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID}"

line="$(sed -n "${TASK_ID}p" "${PROJECT_DIR}/jobs.csv" || true)"
if [[ -z "${line}" ]]; then
  echo "ERROR: No line ${TASK_ID} in ${TASKS_TSV}" >&2
  exit 3
fi

# Parse: ix iy iz params_path
# Using read to split on whitespace/tabs
IFS=, read -r ix iy iz <<< "$line"

if [[ -z "${ix}" || -z "${iy}" || -z "${iz}" ]]; then
  echo "ERROR: Malformed line ${TASK_ID}: '${line}'" >&2
  exit 4
fi

echo "============================================================"
echo "Job:         ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:        $(hostname)"
echo "Tile:        ix=${ix}, iy=${iy}, iz=${iz}"
echo "============================================================"

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
  --bind "${PROJECT_DIR}:/work" \
  --bind "${CAVE_SECRETS_DIR}:/root/.cloudvolume" \
  "${SIF}" \
  python /usr/local/app/downsample.py -s /work ${ix} ${iy} ${iz}

echo "Done."
