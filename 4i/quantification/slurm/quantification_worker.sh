#!/usr/bin/env bash
#SBATCH --job-name=myonis_4i
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
# Each field task first runs configured DAPI registration/SSIM QC, then quantifies
# all channels and source contours using those same patch/round translations.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fouri-quantify worker --run-dir "$1" --task-index "${SLURM_ARRAY_TASK_ID:?}"
