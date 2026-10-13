"""Read existing field checkpoints with bounded parallelism and explicit exclusions."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import traceback

import pandas as pd

from .inputs.manifests import file_hash
from .outputs.h5ad import read_partition


# Opt-in only, for assembling existing measurements after a reporting/I/O update.
# Measurement, registration, object-assignment and configuration code stay pinned.
FINALIZER_FILES = {"pipeline.py", "cli.py", "slurm.py", "finalization.py",
                   "outputs/h5ad.py", "outputs/aggregate.py", "reporting/pdf.py"}


def check_finalizer_changes(original, current, allow_update):
    changed = sorted(k for k in set(original) | set(current) if original.get(k) != current.get(k))
    if changed and (not allow_update or set(changed) - FINALIZER_FILES):
        raise ValueError("Package code changed after prepare: " + ", ".join(changed) +
                         ". --allow-finalizer-update permits only assembly/reporting changes; "
                         "measurement code must still match the original run.")
    return changed


@contextmanager
def local_checkpoint(path, scratch_dir=None, expected_sha256=None):
    """Stream from shared storage once, hashing the exact bytes opened by HDF5."""
    scratch = scratch_dir or os.environ.get("SLURM_TMPDIR") or None
    with tempfile.TemporaryDirectory(prefix="myonis-4i-read-", dir=scratch) as folder:
        local = Path(folder)/"checkpoint.h5ad"
        digest = hashlib.sha256()
        with Path(path).open("rb") as source, local.open("wb") as target:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                target.write(block)
                digest.update(block)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint does not match run checksum")
        yield local


def failure_record(run_dir, task, error):
    record = dict(task, error=str(error))
    saved = Path(run_dir)/"logs"/(task["task_id"]+".error.json")
    if saved.is_file():
        try:
            previous = json.loads(saved.read_text())
            record.update(worker_error=previous.get("error", ""), traceback=previous.get("traceback", ""))
        except (OSError, ValueError):
            pass
    summary = Path(run_dir)/"registration_qc"/task["task_id"]/"summary.json"
    if summary.is_file():
        try:
            failed = json.loads(summary.read_text()).get("failed", [])
            if failed:
                record["registration_failures_json"] = json.dumps(failed)
        except (OSError, ValueError):
            pass
    return record


def preflight_checkpoints(run_dir, tasks, fingerprint):
    """Find absent/unfinished fields before any expensive HDF5 reads."""
    ready, failures = [], []
    for task in tasks:
        path = Path(run_dir)/"checkpoints"/(task["task_id"]+".h5ad")
        try:
            done = json.loads(path.with_suffix(".done.json").read_text())
            if done["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint does not match run fingerprint")
            if not path.is_file():
                raise FileNotFoundError("Field checkpoint is missing")
            if not done.get("sha256"):
                raise ValueError("Checkpoint completion record has no checksum")
            ready.append(task)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(failure_record(run_dir, task, exc))
    return ready, failures


def _read_one(job):
    run_dir, task, fingerprint, scratch_dir = job
    run_dir = Path(run_dir)
    path = run_dir/"checkpoints"/(task["task_id"]+".h5ad")
    try:
        done = json.loads(path.with_suffix(".done.json").read_text())
        if done["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint does not match run fingerprint")
        with local_checkpoint(path, scratch_dir, done["sha256"]) as local:
            for name, checksum in done.get("registration_files", {}).items():
                if not (run_dir/name).is_file() or file_hash(run_dir/name) != checksum:
                    raise ValueError("Registration/QC artifact missing or changed; rerun this field worker")
            part = read_partition(local)
        if part["field_key"] != task["field_key"]:
            raise ValueError("Checkpoint field identity differs from the prepared task")
        invalid = {name for name, checksum in done.get("visualisation_files", {}).items()
                   if not (run_dir/name).is_file() or file_hash(run_dir/name) != checksum}
        if invalid and len(part["visualisations"]):
            for row_id, view in part["visualisations"].iterrows():
                if any(view.get(key, "") in invalid for key in ["png", "audit", "masks", "metadata"]):
                    part["visualisations"].loc[row_id, ["status", "detail"]] = [
                        "failed", "Visualisation artifact missing or checksum changed; rerun this field worker"]
        return part, done, None
    except Exception as exc:
        # An exhausted/unwritable scratch disk is a job failure, not evidence
        # that an otherwise valid field should be excluded from the archive.
        if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in {
                errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.ENOMEM}):
            raise
        failure = failure_record(run_dir, task, exc)
        failure["checkpoint_traceback"] = traceback.format_exc()
        return None, None, failure


def read_checkpoints(run_dir, tasks, fingerprint, workers=1, scratch_dir=None):
    """Keep task order stable, with at most `workers` pending results/copies."""
    jobs = ((str(run_dir), task, fingerprint, scratch_dir) for task in tasks)
    if workers == 1:
        for job in jobs:
            yield _read_one(job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = deque()
        for _ in range(min(workers, len(tasks))):
            pending.append(pool.submit(_read_one, next(jobs)))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_read_one, job))


def excluded_fields(failures, images):
    columns = ["task_id", "field_key", "plate_name", "repeat", "cell_line", "well_id",
               "well_row", "well_column", "cohort_id", "field_id", "excluded_patches",
               "failed_patch_indices", "reason", "exclusion_scope"]
    metadata = images.drop_duplicates("field_key").set_index("field_key")
    patch_counts = images.groupby("field_key").spatial_key.nunique()
    rows = []
    for failure in failures:
        key = failure["field_key"]
        row = {col: metadata.loc[key, col] for col in columns[2:10] if key in metadata.index and col in metadata}
        reasons, patches = [], []
        for item in json.loads(failure.get("registration_failures_json", "[]")):
            match = re.search(r"__patch_(\d+)", item.get("patch_key", ""))
            patch = str(int(match[1])) if match else "unknown"
            patches.append(patch)
            reasons.append(f"Patch {patch}: {item.get('error', 'Registration failed')}")
        row.update(task_id=failure["task_id"], field_key=key,
                   excluded_patches=int(patch_counts.get(key, 0)), failed_patch_indices=", ".join(patches),
                   reason="; ".join(reasons) or failure.get("worker_error") or failure["error"],
                   exclusion_scope="whole_field")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
