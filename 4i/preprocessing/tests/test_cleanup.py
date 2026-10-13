from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from utils.manifests import read_json, write_json
from utils.registry import configured_well_tasks
from utils.config import load_config


def make_run(tmp_path: Path, *, enabled: bool) -> Path:
    run_dir = tmp_path / "run"
    provenance = run_dir / "provenance"
    provenance.mkdir(parents=True)
    config = {
        "data_root": "/data",
        "evaluation_dir": "/evaluation",
        "test_mode": True,
        "storage": {
            "cleanup_corrected_after_registration": enabled,
            "cleanup_registered_after_patching": enabled,
        },
    }
    (provenance / "pipeline_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    write_json(provenance / "code_sha256.json", {})
    write_json(run_dir / "run_manifest.json", {"run_id": "cleanup_test"})
    return run_dir


def write_well_statuses(run_dir: Path, filename: str, status: str) -> None:
    config = load_config(run_dir / "provenance" / "pipeline_config.json")
    for plate, row, column in configured_well_tasks(config):
        path = (
            run_dir
            / "manifests"
            / "wells"
            / f"{plate.name}__r{row:02d}_c{column:02d}"
            / filename
        )
        write_json(path, {"status": status})


def run_cleanup(run_dir: Path, stage: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "cleanup_intermediates.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--stage",
            stage,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_enabled_cleanup_removes_only_corrected_after_registration(tmp_path: Path):
    run_dir = make_run(tmp_path, enabled=True)
    target = run_dir / "corrected"
    target.mkdir()
    (target / "payload.bin").write_bytes(b"data")
    retained = run_dir / "quantitative_patches"
    retained.mkdir()
    (retained / "keep.bin").write_bytes(b"keep")
    write_well_statuses(run_dir, "well_status.json", "complete")

    result = run_cleanup(run_dir, "corrected")

    assert result.returncode == 0, result.stderr
    assert not target.exists()
    assert (retained / "keep.bin").is_file()
    manifest = read_json(run_dir / "manifests" / "cleanup" / "corrected.json")
    assert manifest["status"] == "complete"
    assert manifest["removed"] is True
    assert manifest["validated_downstream_well_count"] == 4


def test_cleanup_refuses_incomplete_downstream_and_preserves_target(tmp_path: Path):
    run_dir = make_run(tmp_path, enabled=True)
    target = run_dir / "registered_crops"
    target.mkdir()
    (target / "payload.bin").write_bytes(b"data")

    result = run_cleanup(run_dir, "registered_crops")

    assert result.returncode != 0
    assert target.is_dir()
    assert (target / "payload.bin").is_file()


def test_disabled_cleanup_is_recorded_and_preserves_target(tmp_path: Path):
    run_dir = make_run(tmp_path, enabled=False)
    target = run_dir / "corrected"
    target.mkdir()
    (target / "payload.bin").write_bytes(b"data")

    result = run_cleanup(run_dir, "corrected")

    assert result.returncode == 0, result.stderr
    assert (target / "payload.bin").is_file()
    manifest = read_json(run_dir / "manifests" / "cleanup" / "corrected.json")
    assert manifest["status"] == "skipped_disabled"
    assert manifest["removed"] is False
