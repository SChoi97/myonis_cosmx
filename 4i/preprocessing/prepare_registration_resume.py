#!/usr/bin/env python3
"""Validate registration recovery and record submission-script-only changes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

from utils.manifests import atomic_write_text, read_json, sha256_file, write_json
from utils.registry import ACQUISITIONS, PLATES


RESUME_SUBMISSION_SCRIPTS = {
    "4i/preprocessing/submit_preprocessing.sh",
    "4i/preprocessing/submit_preprocessing_1080.sh",
    "4i/preprocessing/submit_preprocessing_1080_global_registration.sh",
}


def prepare_resume(
    run_dir: Path,
    run_mode: str,
    patch_size: int,
    *,
    check_only: bool = False,
) -> tuple[str, str]:
    run_dir = run_dir.expanduser().resolve()
    provenance = run_dir / "provenance"
    manifest = read_json(run_dir / "run_manifest.json")
    config = read_json(provenance / "pipeline_config.json")
    registration_level, scope = run_mode.rsplit("_", 1)
    expected_test_mode = scope == "test"
    if (
        manifest.get("run_mode") != run_mode
        or config.get("registration", {}).get("level") != registration_level
        or bool(config.get("test_mode")) != expected_test_mode
        or config.get("patch_size") != patch_size
    ):
        raise ValueError(
            f"Resume run must match RUN_MODE={run_mode} and PATCH_SIZE={patch_size}: "
            f"{run_dir}. Its saved configuration will not be changed."
        )

    # Reuse the run's saved inputs, without regenerating the selection or config.
    for name, hash_key in (
        ("pipeline_config.json", "config_sha256"),
        ("illumination_method_selection.csv", "selection_sha256"),
        ("clip_limits.xlsx", "clip_limits_sha256"),
    ):
        expected_hash = manifest.get("snapshots", {}).get(hash_key)
        if not expected_hash or sha256_file(provenance / name) != expected_hash:
            raise RuntimeError(f"Run input differs from its snapshot: {name}")
    if not (provenance / "registration_overrides.csv").is_file():
        raise FileNotFoundError(provenance / "registration_overrides.csv")

    source_root = Path(__file__).resolve().parents[2]
    code_path = provenance / "code_sha256.json"
    expected_code = read_json(code_path)
    updated_code = dict(expected_code)
    changes = {}
    mismatches = []
    for relative, expected_hash in expected_code.items():
        source = source_root / relative
        if not source.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        current_hash = sha256_file(source)
        if current_hash == expected_hash:
            continue
        if relative not in RESUME_SUBMISSION_SCRIPTS:
            mismatches.append(f"changed:{relative}")
            continue
        changes[relative] = {"before": expected_hash, "after": current_hash}
        updated_code[relative] = current_hash
    if mismatches:
        raise RuntimeError(
            "Processing code differs from the run snapshot; registration recovery "
            "allows changes only to submission controllers: "
            + ", ".join(mismatches[:20])
        )

    # Registration workers need these success markers to verify cached wells,
    # and corrected stores to rebuild the wells interrupted by the full disk.
    for acquisition in ACQUISITIONS:
        for plate in PLATES:
            folder = run_dir / "corrected" / acquisition.acquisition_id
            success = read_json(folder / f"{plate.name}_SUCCESS.json")
            if (
                success.get("status") != "complete"
                or not success.get("task_hash")
                or not (folder / f"{plate.name}.ome.zarr").is_dir()
            ):
                raise RuntimeError(
                    "Registration recovery requires completed corrected inputs: "
                    f"{folder}/{plate.name}. Restore missing inputs before resuming."
                )

    graph = manifest.get("slurm_job_graph", {})
    prerequisite_jobs = []
    for stage in ("real_data_smoke_test", "illumination"):
        job = str(graph.get(stage, "")).split(";", 1)[0]
        if not job.isdigit():
            raise ValueError(f"Run lacks its original {stage} job ID.")
        prerequisite_jobs.append(job)

    if not check_only:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        history = provenance / "registration_resume" / f"{stamp}_{uuid4().hex[:8]}"
        # Save the old hashes and job graph before changing either active record.
        atomic_write_text(history / "code_sha256_before.json", code_path.read_text())
        atomic_write_text(
            history / "run_manifest_before.json",
            (run_dir / "run_manifest.json").read_text(),
        )
        write_json(
            history / "resume_plan.json",
            {
                "run_dir": str(run_dir),
                "run_mode": run_mode,
                "resume_from": "registration_and_crop",
                "submission_script_changes": changes,
                "resume_helper_sha256": sha256_file(Path(__file__)),
                "previous_slurm_job_graph": graph,
            },
        )
        for relative in sorted(changes):
            atomic_write_text(
                history / Path(relative).name,
                (source_root / relative).read_text(),
            )
        if changes:
            write_json(code_path, updated_code)
        print(f"Registration recovery history: {history}", file=sys.stderr)
    print(
        "Reusing completed illumination correction. Registration workers use "
        "--resume to reuse completed wells and replace incomplete crop stores.",
        file=sys.stderr,
    )
    return tuple(prerequisite_jobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--run-mode",
        choices=(
            "field_level_test",
            "field_level_production",
            "patch_level_test",
            "patch_level_production",
        ),
        required=True,
    )
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    jobs = prepare_resume(
        args.run_dir, args.run_mode, args.patch_size, check_only=args.check_only
    )
    # stdout is consumed by the submission scripts; diagnostics go to stderr.
    print(" ".join(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
