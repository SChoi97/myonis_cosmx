#!/usr/bin/env python3
"""Safely remove stage intermediates after all downstream consumers succeed."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from utils.config import load_config
from utils.manifests import read_json, write_json
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.registry import configured_well_tasks


TARGETS = {
    "corrected": {
        "directory": "corrected",
        "setting": "cleanup_corrected_after_registration",
        "prerequisite_status": "well_status.json",
        "allowed_statuses": {"complete", "quarantined"},
        "consumer": "registration_and_crop",
    },
    "registered_crops": {
        "directory": "registered_crops",
        "setting": "cleanup_registered_after_patching",
        "prerequisite_status": "patch_status.json",
        "allowed_statuses": {"complete", "skipped_quarantined_well"},
        "consumer": "patch_generation",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(TARGETS), required=True)
    return parser


def validated_target(run_dir: Path, directory_name: str) -> Path:
    """Resolve only one known direct child of a genuine pipeline run."""
    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Not a pipeline run directory: {run_dir}")
    if not (run_dir / "provenance" / "pipeline_config.json").is_file():
        raise FileNotFoundError(f"Run lacks its configuration snapshot: {run_dir}")
    target = (run_dir / directory_name).resolve()
    if target.parent != run_dir or target.name != directory_name:
        raise RuntimeError(f"Refusing unsafe cleanup target: {target}")
    if target.is_symlink():
        raise RuntimeError(f"Refusing to recursively remove symlink: {target}")
    return target


def validate_downstream_completion(run_dir: Path, config, specification: dict) -> int:
    """Require one successful downstream status for every configured well."""
    missing = []
    invalid = []
    tasks = configured_well_tasks(config)
    for plate, row, column in tasks:
        status_path = (
            run_dir
            / "manifests"
            / "wells"
            / f"{plate.name}__r{row:02d}_c{column:02d}"
            / specification["prerequisite_status"]
        )
        if not status_path.is_file():
            missing.append(str(status_path))
            continue
        status = read_json(status_path).get("status")
        if status not in specification["allowed_statuses"]:
            invalid.append(f"{status_path}:{status}")
    if missing or invalid:
        raise RuntimeError(
            f"Cannot clean {specification['directory']}; downstream "
            f"{specification['consumer']} is incomplete. "
            f"missing={missing[:10]}, invalid={invalid[:10]}"
        )
    return len(tasks)


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    verify_code_snapshot(run_dir)
    started, monotonic_start = start_task()
    config = load_config(run_dir / "provenance" / "pipeline_config.json")
    specification = TARGETS[args.stage]
    target = validated_target(run_dir, specification["directory"])
    enabled = bool(getattr(config.storage, specification["setting"]))
    result = {
        "stage": "intermediate_cleanup",
        "cleanup_target": args.stage,
        "target_path": str(target),
        "configuration_setting": f"storage.{specification['setting']}",
        "enabled": enabled,
        "downstream_consumer": specification["consumer"],
    }

    if not enabled:
        result.update(
            {
                "status": "skipped_disabled",
                "target_existed_before": target.exists(),
                "removed": False,
                "validated_downstream_well_count": 0,
            }
        )
    else:
        validated_count = validate_downstream_completion(
            run_dir, config, specification
        )
        existed = target.exists()
        if existed:
            if not target.is_dir():
                raise RuntimeError(f"Cleanup target is not a directory: {target}")
            shutil.rmtree(target)
        if target.exists():
            raise RuntimeError(f"Cleanup target still exists after removal: {target}")
        result.update(
            {
                "status": "complete",
                "target_existed_before": existed,
                "removed": existed,
                "validated_downstream_well_count": validated_count,
            }
        )

    result["environment"] = task_environment(started, monotonic_start)
    manifest_path = run_dir / "manifests" / "cleanup" / f"{args.stage}.json"
    write_json(manifest_path, result)
    print(
        f"{args.stage}: {result['status']} "
        f"(removed={result['removed']}, target={target})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
