"""Run initialization and reproducibility metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .manifests import read_csv, read_json, sha256_file, write_json
from .layout import FIELD_ID_GRID, FIELD_SHAPE, GRID_SHAPE


LOCAL_TIMEZONE = ZoneInfo("Europe/London")

# These directories contain generated data, reports, caches, or tests rather
# than production pipeline code.  Prune them during traversal so large Zarr
# stores and previous runs are never enumerated while creating code hashes.
CODE_HASH_EXCLUDED_DIRECTORIES = {
    "results",
    "outputs",
    "logs",
    "__pycache__",
    ".ipynb_checkpoints",
    "tests",
}


def timestamp_values() -> dict[str, str]:
    now_utc = datetime.now(timezone.utc)
    return {
        "utc": now_utc.isoformat(),
        "local": now_utc.astimezone(LOCAL_TIMEZONE).isoformat(),
        "timezone": str(LOCAL_TIMEZONE),
    }


def start_task() -> tuple[dict[str, str], float]:
    return timestamp_values(), time.monotonic()


def verify_code_snapshot(run_dir: str | Path) -> None:
    """Refuse to mix production code revisions within one run."""
    run_dir = Path(run_dir)
    expected = read_json(run_dir / "provenance" / "code_sha256.json")
    source_root = Path(__file__).resolve().parents[3]
    mismatches = []
    for relative, expected_hash in expected.items():
        path = source_root / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
        elif sha256_file(path) != expected_hash:
            mismatches.append(f"changed:{relative}")
    if mismatches:
        raise RuntimeError(
            "Production code differs from this run's initialization snapshot: "
            + ", ".join(mismatches[:20])
        )


def command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return (result.stdout or result.stderr).strip()


def production_code_hashes(code_root: Path, source_root: Path) -> dict[str, str]:
    """Hash production Python/shell files without traversing generated data."""
    hashes: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(code_root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in CODE_HASH_EXCLUDED_DIRECTORIES
        )
        directory_path = Path(directory)
        for file_name in sorted(file_names):
            path = directory_path / file_name
            if path.suffix not in {".py", ".sh"} or not path.is_file():
                continue
            hashes[str(path.relative_to(source_root))] = sha256_file(path)
    return hashes


def initialize_run(
    runs_root: Path,
    config_path: Path,
    selection_path: Path,
    selection_workbook_path: Path,
    clip_limits_path: Path,
    source_root: Path,
    *,
    config_snapshot: dict,
    model_load_validation: dict,
) -> Path:
    stamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d_%H%M%S")
    commit = command_output(["git", "rev-parse", "--short", "HEAD"], source_root)
    commit_slug = commit if commit and "unavailable" not in commit else "nogit"
    data_scope = "test" if config_snapshot.get("test_mode") else "production"
    registration_level = config_snapshot.get("registration", {}).get(
        "level", "field_level"
    )
    mode_slug = f"{registration_level}_{data_scope}"
    run_id = f"4i_{mode_slug}_{stamp}_{commit_slug}"
    run_dir = runs_root.expanduser().resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    provenance = run_dir / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "pipeline_config.json").write_text(
        json.dumps(config_snapshot, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(selection_path, provenance / "illumination_method_selection.csv")
    shutil.copy2(
        selection_workbook_path,
        provenance / "illumination_method_choices.xlsx",
    )
    shutil.copy2(clip_limits_path, provenance / clip_limits_path.name)
    override_template = Path(__file__).resolve().parents[1] / "config" / "registration_overrides.csv"
    shutil.copy2(override_template, provenance / "registration_overrides.csv")
    training_snapshot = provenance / "illumination_training"
    training_snapshot.mkdir()
    selection_rows = read_csv(selection_path)
    copied_sampling = set()
    for row in selection_rows:
        training_source = Path(row.get("training_paths_path", ""))
        if training_source.is_file():
            shutil.copy2(
                training_source,
                training_snapshot / f"{row['acquisition_id']}_training_fields.csv",
            )
        sampling_source = Path(row.get("shared_sampling_manifest_path", ""))
        if sampling_source.is_file() and sampling_source not in copied_sampling:
            shutil.copy2(sampling_source, training_snapshot / "shared_sampling_manifest.csv")
            copied_sampling.add(sampling_source)
    diff = command_output(["git", "diff", "--no-ext-diff"], source_root)
    (provenance / "git_diff.patch").write_text(diff, encoding="utf-8")
    (provenance / "pip_freeze.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8"
    )
    conda_executable = os.environ.get("CONDA_EXE", "conda")
    (provenance / "conda_environment.yml").write_text(
        command_output([conda_executable, "env", "export", "--no-builds"]),
        encoding="utf-8",
    )
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = production_code_hashes(code_root, source_root)
    write_json(provenance / "code_sha256.json", code_hashes)
    manifest = {
        "run_id": run_id,
        "status": "initialized",
        "run_mode": mode_slug,
        "data_scope": data_scope,
        "registration_level": registration_level,
        "test_mode": bool(config_snapshot.get("test_mode")),
        "selected_test_wells": (
            config_snapshot.get("test_wells", [])
            if config_snapshot.get("test_mode")
            else []
        ),
        "preflight_model_load_validation": model_load_validation,
        "started": timestamp_values(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {"version": sys.version, "executable": sys.executable},
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"], source_root),
            "status": command_output(["git", "status", "--short"], source_root),
            "diff_path": str(provenance / "git_diff.patch"),
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_SUBMIT_DIR",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
            )
        },
        "snapshots": {
            "config": str(provenance / "pipeline_config.json"),
            "config_sha256": sha256_file(provenance / "pipeline_config.json"),
            "config_template_source": str(config_path),
            "config_template_source_sha256": sha256_file(config_path),
            "selection": str(provenance / "illumination_method_selection.csv"),
            "selection_sha256": sha256_file(
                provenance / "illumination_method_selection.csv"
            ),
            "selection_workbook": str(
                provenance / "illumination_method_choices.xlsx"
            ),
            "selection_workbook_sha256": sha256_file(
                provenance / "illumination_method_choices.xlsx"
            ),
            "clip_limits": str(provenance / clip_limits_path.name),
            "clip_limits_sha256": sha256_file(provenance / clip_limits_path.name),
            "registration_overrides": str(provenance / "registration_overrides.csv"),
            "code_hashes": str(provenance / "code_sha256.json"),
            "pip_freeze": str(provenance / "pip_freeze.txt"),
            "conda_environment": str(provenance / "conda_environment.yml"),
        },
        "image_geometry": {
            "field_layout": "phenix-field1-bottom-left-snake",
            "field_id_grid_top_to_bottom": FIELD_ID_GRID.tolist(),
            "grid_shape": list(GRID_SHAPE),
            "field_shape": list(FIELD_SHAPE),
            "mosaic_shape": [
                GRID_SHAPE[0] * FIELD_SHAPE[0],
                GRID_SHAPE[1] * FIELD_SHAPE[1],
            ],
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def task_environment(
    started: dict[str, str] | None = None,
    monotonic_start: float | None = None,
) -> dict:
    ended = timestamp_values()
    result = {
        "started": started or ended,
        "ended": ended,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if monotonic_start is not None:
        result["duration_seconds"] = max(0.0, time.monotonic() - monotonic_start)
    return result
