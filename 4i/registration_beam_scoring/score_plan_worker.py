#!/usr/bin/env python3
"""Score this array worker's assigned wells sequentially within one allocation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from progress import progress
from score_seams import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-plan", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.scoring_plan.read_text())
    if not 0 <= args.task_index < len(plan["worker_wells"]):
        parser.error("Task index outside worker assignment range")
    assigned = plan["worker_wells"][args.task_index]
    print(f"Worker {args.task_index}: {len(assigned)} wells, "
          f"{plan['worker_image_counts'][args.task_index]} patch images", flush=True)
    for well in progress(assigned, desc=f"Worker {args.task_index} wells", unit="well"):
        subprocess.run([sys.executable, "-u", str(HERE / "score_seams.py"),
                        "--cache-manifest", str(args.cache_manifest),
                        "--output-dir", str(args.output_dir), "--well", well,
                        "--scoring-plan", str(args.scoring_plan)], check=True)
    print(f"Worker {args.task_index}: all assigned wells completed", flush=True)


if __name__ == "__main__":
    main()
