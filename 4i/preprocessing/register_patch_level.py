#!/usr/bin/env python3
"""Patch-level registration branch: independently align every 1080x1080 field."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from utils.config import load_config
from utils.correction import selected_model_hash
from utils.manifests import (
    approved_selections,
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.patch_level import (
    PATCH_LEVEL_COHORTS,
    analyze_patch_level_dapi,
    cohort_for_measurement,
    region_patch_plan,
)
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.qc_visualization import (
    save_field_qc_heatmaps,
    save_patch_level_region_heatmaps,
)
from utils.registration import apply_translation
from utils.registry import ACQUISITIONS, configured_well_tasks, dapi_for_measurement
from utils.storage import (
    corrected_store_path,
    create_cropped_store,
    cropped_store_path,
    read_corrected_well,
    write_patch_level_crop,
)


def patch_override_for(
    rows: list[dict[str, str]],
    plate_name: str,
    row: int,
    column: int,
    measurement: int,
    field_id: int,
) -> dict[str, str] | None:
    """Allow a blank field_id override to apply to all fields, or target one field."""
    matches = []
    for item in rows:
        try:
            identity_matches = (
                item.get("plate_name") == plate_name
                and int(item.get("row", -1)) == row
                and int(item.get("column", -1)) == column
                and int(item.get("measurement", -1)) == measurement
            )
        except (TypeError, ValueError):
            continue
        raw_field = str(item.get("field_id", "")).strip()
        field_matches = not raw_field or int(raw_field) == field_id
        if identity_matches and field_matches:
            matches.append(item)
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate patch registration overrides for {plate_name} "
            f"r{row:02d}_c{column:02d}, measurement {measurement}, field {field_id}."
        )
    return matches[0] if matches else None


def task_input_hash(run_dir: Path, plate_name: str, row: int, column: int) -> str:
    corrected_hashes = {}
    for acquisition in ACQUISITIONS:
        success_path = (
            run_dir
            / "corrected"
            / acquisition.acquisition_id
            / f"{plate_name}_SUCCESS.json"
        )
        result = read_json(success_path)
        if result.get("status") != "complete" or not result.get("task_hash"):
            raise RuntimeError(f"Incomplete correction prerequisite: {success_path}.")
        corrected_hashes[acquisition.acquisition_id] = result["task_hash"]
    return sha256_json(
        {
            "stage": "patch_level_registration_and_crop",
            "plate_name": plate_name,
            "row": row,
            "column": column,
            "config_sha256": sha256_file(
                run_dir / "provenance" / "pipeline_config.json"
            ),
            "selection_sha256": sha256_file(
                run_dir / "provenance" / "illumination_method_selection.csv"
            ),
            "overrides_sha256": sha256_file(
                run_dir / "provenance" / "registration_overrides.csv"
            ),
            "corrected_task_hashes": corrected_hashes,
        }
    )


def run(args: argparse.Namespace, run_dir: Path, config) -> int:
    if config.registration.level != "patch_level":
        raise ValueError("register_patch_level.run requires registration.level=patch_level.")
    tasks = configured_well_tasks(config)
    if args.task_index is None or not 0 <= args.task_index < len(tasks):
        raise ValueError(f"--task-index must be in 0..{len(tasks) - 1}.")
    verify_code_snapshot(run_dir)
    task_started, monotonic_start = start_task()
    selections = approved_selections(
        run_dir / "provenance" / "illumination_method_selection.csv"
    )
    plate, row, column = tasks[args.task_index]
    well_slug = f"{plate.name}__r{row:02d}_c{column:02d}"
    manifest_dir = run_dir / "manifests" / "wells" / well_slug
    status_path = manifest_dir / "well_status.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    input_hash = task_input_hash(run_dir, plate.name, row, column)
    output_store = cropped_store_path(run_dir, plate.name, row, column)
    if args.resume and status_path.is_file():
        previous = read_json(status_path)
        reusable = (
            previous.get("task_hash") == input_hash
            and previous.get("execution_mode") == "production"
            and (previous.get("whole_well_quarantine") or output_store.is_dir())
        )
        if reusable:
            print(f"{well_slug}: resumed verified {previous.get('status')} patch-level result")
            return 0

    dapi_fields = {
        measurement: read_corrected_well(
            corrected_store_path(
                run_dir,
                dapi_for_measurement(measurement).acquisition_id,
                plate.name,
            ),
            row,
            column,
        )
        for measurement in range(1, 13)
    }
    overrides = read_csv(run_dir / "provenance" / "registration_overrides.csv")
    shifts, field_qc_rows, regions = analyze_patch_level_dapi(
        dapi_fields,
        config.registration,
        config.patch_size,
        override_lookup=lambda measurement, field_id: patch_override_for(
            overrides, plate.name, row, column, measurement, field_id
        ),
    )
    del dapi_fields

    shift_lookup = {
        (str(item["cohort_id"]), int(item["field_id"]), int(item["measurement"])): item
        for item in shifts
    }
    successful_regions = [item for item in regions if item["status"] == "complete"]
    failed_regions = [item for item in regions if item["status"] != "complete"]
    quarantine_rows = []
    quarantined_patch_ids = set()
    for region in failed_regions:
        possible_patches = region_patch_plan(region, config.patch_size)
        if possible_patches:
            for patch in possible_patches:
                quarantined_patch_ids.add(patch["patch_id"])
                quarantine_rows.append(
                    {
                        "plate_name": plate.name,
                        "row": row,
                        "column": column,
                        "registration_level": "patch_level",
                        "cohort_id": region["cohort_id"],
                        "field_id": region["field_id"],
                        "scope": "patch",
                        "patch_id": patch["patch_id"],
                        "failed_field_ids": region["field_id"],
                        "reason": region["failure_reasons"],
                    }
                )
        else:
            quarantine_rows.append(
                {
                    "plate_name": plate.name,
                    "row": row,
                    "column": column,
                    "registration_level": "patch_level",
                    "cohort_id": region["cohort_id"],
                    "field_id": region["field_id"],
                    "scope": "field_cohort",
                    "patch_id": "",
                    "failed_field_ids": region["field_id"],
                    "reason": region["failure_reasons"],
                }
            )

    write_csv(manifest_dir / "registration_shifts.csv", shifts)
    write_csv(manifest_dir / "field_qc.csv", field_qc_rows)
    write_csv(manifest_dir / "patch_level_regions.csv", regions)
    write_csv(manifest_dir / "quarantine.csv", quarantine_rows)
    save_field_qc_heatmaps(
        manifest_dir / "figures" / "field_qc_heatmaps.png",
        field_qc_rows,
        title=f"{plate.name} r{row:02d}_c{column:02d} patch-level registration QC",
    )
    save_patch_level_region_heatmaps(
        manifest_dir / "figures" / "patch_level_regions.png",
        regions,
        title=f"{plate.name} r{row:02d}_c{column:02d} cohort-specific field crops",
    )

    patch_count_total = sum(int(item["patch_count_total"]) for item in regions)
    patch_count_retained = sum(
        int(item["patch_count_retained"]) for item in regions
    )
    acquisitions_per_cohort = {
        cohort.cohort_id: sum(
            acquisition.measurement in cohort.measurements
            for acquisition in ACQUISITIONS
        )
        for cohort in PATCH_LEVEL_COHORTS
    }
    expected_files = sum(
        int(region["patch_count_retained"])
        * acquisitions_per_cohort[str(region["cohort_id"])]
        for region in regions
    )
    failed_field_ids = sorted({int(item["field_id"]) for item in failed_regions})
    whole_well_quarantine = not successful_regions
    overlap_values = [float(item["common_overlap_fraction"]) for item in regions]
    crop_values = [float(item["crop_area_fraction"]) for item in regions]
    status = {
        "stage": "registration_and_crop",
        "registration_level": "patch_level",
        "registration_cohorts": [
            {
                "cohort_id": item.cohort_id,
                "measurements": list(item.measurements),
                "reference_measurement": item.reference_measurement,
            }
            for item in PATCH_LEVEL_COHORTS
        ],
        "task_index": args.task_index,
        "test_mode": config.test_mode,
        "plate_name": plate.name,
        "row": row,
        "column": column,
        "status": "quarantined" if whole_well_quarantine else "complete",
        "whole_well_quarantine": whole_well_quarantine,
        "failed_field_ids": failed_field_ids,
        "failed_field_cohort_count": len(failed_regions),
        "retained_field_cohort_count": len(successful_regions),
        "quarantined_patch_ids": sorted(quarantined_patch_ids),
        "common_bounds": None,
        "common_overlap_fraction": float(np.mean(overlap_values)),
        "minimum_common_overlap_fraction": float(np.min(overlap_values)),
        "crop_bounds": None,
        "crop_area_fraction": float(np.mean(crop_values)),
        "patch_size": config.patch_size,
        "patch_count_total": patch_count_total,
        "patch_count_retained": patch_count_retained,
        "expected_quantitative_patch_files": expected_files,
        "patch_level_regions": regions,
        "task_hash": input_hash,
        "execution_mode": "validation_only" if args.validate_only else "production",
        "environment": task_environment(task_started, monotonic_start),
    }
    if args.validate_only or whole_well_quarantine:
        if whole_well_quarantine and not args.validate_only and output_store.exists():
            shutil.rmtree(output_store)
        write_json(status_path, status)
        print(f"{well_slug}: {status['status']} patch-level result")
        return 0

    if output_store.exists():
        shutil.rmtree(output_store)
    group = create_cropped_store(
        output_store,
        {
            "plate_name": plate.name,
            "row": row,
            "column": column,
            "registration_level": "patch_level",
            "registration_cohorts": [
                {
                    "cohort_id": item.cohort_id,
                    "measurements": list(item.measurements),
                    "reference_measurement": item.reference_measurement,
                }
                for item in PATCH_LEVEL_COHORTS
            ],
        },
    )
    regions_by_cohort = {
        cohort.cohort_id: [
            item
            for item in successful_regions
            if item["cohort_id"] == cohort.cohort_id
        ]
        for cohort in PATCH_LEVEL_COHORTS
    }
    for acquisition in ACQUISITIONS:
        selection = selections[acquisition.acquisition_id]
        cohort = cohort_for_measurement(acquisition.measurement)
        fields = read_corrected_well(
            corrected_store_path(run_dir, acquisition.acquisition_id, plate.name),
            row,
            column,
        )
        for region in regions_by_cohort[cohort.cohort_id]:
            field_id = int(region["field_id"])
            shift = shift_lookup[
                (cohort.cohort_id, field_id, acquisition.measurement)
            ]
            image = fields[field_id - 1]
            registered = (
                image
                if acquisition.measurement == cohort.reference_measurement
                else apply_translation(
                    image, float(shift["shift_y"]), float(shift["shift_x"])
                )
            )
            y0, y1, x0, x1 = map(int, region["crop_bounds"])
            write_patch_level_crop(
                group,
                cohort.cohort_id,
                field_id,
                acquisition.acquisition_id,
                registered[y0:y1, x0:x1],
                metadata={
                    **acquisition.as_row(),
                    "registration_level": "patch_level",
                    "cohort_id": cohort.cohort_id,
                    "cohort_reference_measurement": cohort.reference_measurement,
                    "field_id": field_id,
                    "field_crop_bounds": list(region["crop_bounds"]),
                    "shift_y": float(shift["shift_y"]),
                    "shift_x": float(shift["shift_x"]),
                    "correction_method": selection["chosen_method"],
                    "correction_model_sha256": selected_model_hash(selection),
                },
                chunk_shape=(config.storage.chunk_y, config.storage.chunk_x),
                compression_level=config.storage.compression_level,
            )
    status["crop_store"] = str(output_store)
    status["environment"] = task_environment(task_started, monotonic_start)
    write_json(status_path, status)
    print(
        f"{well_slug}: patch-level complete; retained "
        f"{len(successful_regions)}/{len(regions)} field cohorts and "
        f"{patch_count_retained} patch geometries"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    return run(args, run_dir, load_config(run_dir / "provenance" / "pipeline_config.json"))


if __name__ == "__main__":
    raise SystemExit(main())
