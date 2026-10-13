"""Small Slurm adapter around the existing prepare/field/finalize pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def batch_indices(task_count, batch_index, batch_count):
    if batch_count < 1 or not 0 <= batch_index < batch_count:
        raise ValueError("batch-count must be positive and batch-index must be in [0, batch-count)")
    # Keep every field/cohort intact, including all its patches and relationships.
    return list(range(batch_index, task_count, batch_count))


def run_batch(run_dir, batch_index, batch_count, workers):
    from .pipeline import run_prepared_tasks
    run_dir = Path(run_dir)
    tasks = json.loads((run_dir/"tasks.json").read_text())
    manifest = json.loads((run_dir/"run_manifest.json").read_text())
    planned = manifest.get("batch_execution", {}).get("jobs")
    if planned is not None and batch_count != planned:
        raise ValueError(f"This run was split into {planned} batches; use that batch-count for recovery")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and workers > int(allocated):
        raise ValueError(f"Requested {workers} workers but Slurm allocated only {allocated} CPUs")
    indices = batch_indices(len(tasks), batch_index, batch_count)
    print(f"Batch {batch_index+1}/{batch_count}: {len(indices)} fields, up to {workers} workers. Run: {run_dir}", flush=True)
    counts = run_prepared_tasks(run_dir, indices, workers)
    print(f"Batch {batch_index+1}/{batch_count}: {counts}", flush=True)
    return counts


def _script(command, array=False):
    return ("#!/usr/bin/env bash\nset -euo pipefail\n"
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1\n"
            "exec "+shlex.join([str(v) for v in command])
            + (' --batch-index "${SLURM_ARRAY_TASK_ID:?}"' if array else "")+"\n")


def _submit(arguments, script):
    result = subprocess.run(["sbatch", "--parsable", *arguments], input=script,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("sbatch failed: "+(result.stderr.strip() or result.stdout.strip()))
    job_id = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError(f"Unexpected sbatch response; check squeue before resubmitting: {result.stdout.strip()}")
    return job_id


def submit_batch(config_path, jobs, workers, partition="ncpu", memory="32G", time_limit="24:00:00",
                 finalize_memory="64G", finalize_time="04:00:00", run_dir=None,
                 troubleshoot=False, troubleshoot_timing=False, visualisation_fields=None, log_dir=None, finalize_workers=4, require_complete=False):
    from .pipeline import prepare, write_json
    if jobs < 1 or workers < 1 or finalize_workers < 1:
        raise ValueError("jobs, workers and finalize-workers must be positive")
    if shutil.which("sbatch") is None:
        raise ValueError("sbatch is unavailable; submit run_quantification_batch.sh on Nemo")
    launcher = Path(__file__).resolve().parents[2]/"run_quantification.py"
    if not launcher.is_file():
        raise ValueError("Batch submission requires the source checkout's run_quantification.py launcher")
    submission_dir = Path(os.environ.get("SLURM_SUBMIT_DIR") or Path.cwd())
    log_dir = (submission_dir/Path(log_dir).expanduser() if log_dir else submission_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log_pattern = str(log_dir/"myonis_4i_batch-%A_%a.out")
    finalize_log_pattern = str(log_dir/"myonis_4i_finalize-%j.out")
    run_dir = prepare(config_path, run_dir=run_dir, workers=workers,
                      troubleshoot=troubleshoot, troubleshoot_timing=troubleshoot_timing,
                      visualisation_fields=visualisation_fields)
    tasks = json.loads((run_dir/"tasks.json").read_text())
    if not tasks:
        raise ValueError("No fields to submit")
    count = min(jobs, len(tasks))
    plan = dict(jobs=count, requested_jobs=jobs, workers_per_job=workers,
                max_field_workers=min(len(tasks), count*workers), fields_per_job=[len(batch_indices(len(tasks), i, count)) for i in range(count)],
                split="Task indices i, i+jobs, i+2*jobs; whole field/cohort per task",
                partition=partition, memory_per_job=memory, time_per_job=time_limit,
                finalize_memory=finalize_memory, finalize_time=finalize_time, finalize_workers=finalize_workers,
                require_complete=require_complete, log_dir=str(log_dir))
    # Freeze the split before any workers start. Only finalization updates this
    # manifest afterwards; submission receipts have their own single writer.
    manifest = json.loads((run_dir/"run_manifest.json").read_text())
    manifest["batch_execution"] = plan
    manifest["worker_count"] = plan["max_field_workers"]
    write_json(run_dir/"run_manifest.json", manifest)
    receipt = dict(plan, run_dir=str(run_dir), status="submitting",
                   prepare_job_id=os.environ.get("SLURM_JOB_ID"), python=sys.executable,
                   worker_log_pattern=worker_log_pattern, finalize_log_pattern=finalize_log_pattern)
    receipt_path = run_dir/"batch_submission.json"
    write_json(receipt_path, receipt)
    command = [sys.executable, "-u", str(launcher)]
    common = [f"--partition={partition}", "--ntasks=1", "--export=ALL", f"--chdir={run_dir}"]
    array_options = [*common, "--job-name=myonis_4i_batch", f"--cpus-per-task={workers}",
                     f"--mem={memory}", f"--time={time_limit}", f"--array=0-{count-1}%{count}",
                     f"--output={worker_log_pattern}", f"--error={worker_log_pattern}"]
    worker_script = _script([*command, "worker-batch", "--run-dir", run_dir,
                             "--batch-count", count, "--workers", workers], array=True)
    final_script = _script([*command, "finalize", "--run-dir", run_dir, "--workers", finalize_workers,
                            "--require-complete" if require_complete else "--allow-partial"])
    try:
        receipt["worker_array_job_id"] = _submit(array_options, worker_script)
        receipt["worker_logs"] = [str(log_dir/f"myonis_4i_batch-{receipt['worker_array_job_id']}_{i}.out") for i in range(count)]
        write_json(receipt_path, receipt)
        print(f"Submitted worker array {receipt['worker_array_job_id']}: {count} jobs, {workers} workers/job; "
              f"{len(tasks)} fields split as {plan['fields_per_job']}.", flush=True)
        print("Worker Slurm logs:\n  "+"\n  ".join(receipt["worker_logs"]), flush=True)
        receipt["finalize_job_id"] = _submit(
            [*common, "--job-name=myonis_4i_finalize", f"--cpus-per-task={finalize_workers}",
             f"--mem={finalize_memory}", f"--time={finalize_time}",
             f"--dependency=afterany:{receipt['worker_array_job_id']}",
             f"--output={finalize_log_pattern}", f"--error={finalize_log_pattern}"], final_script)
        receipt["finalize_log"] = str(log_dir/f"myonis_4i_finalize-{receipt['finalize_job_id']}.out")
    except Exception as exc:
        receipt.update(status="submission_failed", error=str(exc))
        write_json(receipt_path, receipt)
        raise RuntimeError(f"{exc}. Submission details: {receipt_path}. "
                           "Check the recorded job IDs before submitting again; prepared inputs and checkpoints are retained.") from exc
    receipt["status"] = "submitted"
    write_json(receipt_path, receipt)
    policy = "Require every field checkpoint." if require_complete else "Assemble valid fields and report exclusions."
    print(f"Submitted final assembly {receipt['finalize_job_id']} after all batch jobs finish. {policy}", flush=True)
    print(f"Run: {run_dir}\nJob IDs: {receipt_path}", flush=True)
    print(f"Final assembly Slurm log: {receipt['finalize_log']}", flush=True)
    return receipt
