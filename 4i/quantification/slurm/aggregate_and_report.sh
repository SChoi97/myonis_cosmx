#!/usr/bin/env bash
#SBATCH --job-name=myonis_4i_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
fouri-quantify finalize --run-dir "$1"
