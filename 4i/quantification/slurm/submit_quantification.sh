#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/dataset.yaml /absolute/path/new_run_directory" >&2
  exit 2
fi
config_path="$1"
run_directory="$2"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fouri-quantify prepare --config "$config_path" --run-dir "$run_directory"
task_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$run_directory/tasks.json")"
job_id="$(sbatch --parsable --array="0-$((task_count-1))" --output="$run_directory/logs/worker-%A_%a.out" "$script_directory/quantification_worker.sh" "$run_directory")"
sbatch --dependency="afterany:${job_id%%;*}" --output="$run_directory/logs/finalize-%j.out" "$script_directory/aggregate_and_report.sh" "$run_directory"
