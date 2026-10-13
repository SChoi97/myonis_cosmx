#!/usr/bin/env python3
"""Materialize cohort-aligned 512 patches from independently registered fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.config import load_config
from utils.correction import selected_model_hash
from utils.layout import FIELD_ID_GRID
from utils.manifests import (
    approved_selections,
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.patch_level import cohort_for_measurement, region_patch_plan
from utils.patches import patch_level_filename
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.registry import ACQUISITIONS, biological_coordinates, configured_well_tasks
from utils.storage import (
    cropped_store_path,
    open_cropped_store,
    read_patch_level_crop_group,
    write_float_ome_tiff,
)


def run(args: argparse.Namespace, run_dir: Path, config) -> int:
    if config.registration.level != "patch_level":
        raise ValueError(
            "generate_patch_level_patches.run requires registration.level=patch_level."
        )
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
    stage_status_path = manifest_dir / "patch_status.json"
    well_status = read_json(manifest_dir / "well_status.json")
    task_hash = sha256_json(
        {
            "stage": "patch_level_patch_generation",
            "well_status_sha256": sha256_file(manifest_dir / "well_status.json"),
            "config_sha256": sha256_file(
                run_dir / "provenance" / "pipeline_config.json"
            ),
            "selection_sha256": sha256_file(
                run_dir / "provenance" / "illumination_method_selection.csv"
            ),
        }
    )
    if args.resume and stage_status_path.is_file():
        previous = read_json(stage_status_path)
        reusable = (
            previous.get("task_hash") == task_hash
            and previous.get("execution_mode") == "production"
        )
        if reusable and previous.get("status") == "skipped_quarantined_well":
            print(f"{well_slug}: resumed quarantined-well patch decision")
            return 0
        if reusable and previous.get("status") == "complete":
            prior_rows = read_csv(manifest_dir / "patches.csv")
            if len(prior_rows) == int(previous.get("written_patch_count", -1)) and all(
                Path(item["quantitative_path"]).is_file() for item in prior_rows
            ):
                print(f"{well_slug}: resumed verified patch-level outputs")
                return 0

    replaced_outputs = []
    if not args.validate_only:
        quantitative_root = (run_dir / "quantitative_patches").resolve()
        for prior in read_csv(manifest_dir / "patches.csv"):
            prior_path = Path(prior.get("quantitative_path", "")).resolve()
            if prior_path.is_file() and prior_path.is_relative_to(quantitative_root):
                prior_path.unlink()
                replaced_outputs.append(
                    {
                        "stage": "patch_generation",
                        "plate_name": plate.name,
                        "row": row,
                        "column": column,
                        "path": str(prior_path),
                        "reason": "task_hash_changed_or_incomplete_previous_run",
                    }
                )
        write_csv(
            run_dir / "manifests" / "replaced_outputs" / f"{well_slug}__patches.csv",
            replaced_outputs,
        )
    if well_status["whole_well_quarantine"]:
        write_json(
            stage_status_path,
            {
                "stage": "patch_generation",
                "registration_level": "patch_level",
                "status": "skipped_quarantined_well",
                "plate_name": plate.name,
                "row": row,
                "column": column,
                "written_patch_count": 0,
                "replaced_previous_output_count": len(replaced_outputs),
                "task_hash": task_hash,
                "execution_mode": (
                    "validation_only" if args.validate_only else "production"
                ),
                "environment": task_environment(task_started, monotonic_start),
            },
        )
        print(f"{well_slug}: skipped quarantined well")
        return 0

    regions = [
        item
        for item in well_status["patch_level_regions"]
        if item["status"] == "complete"
    ]
    regions_by_cohort = {}
    for region in regions:
        regions_by_cohort.setdefault(str(region["cohort_id"]), []).append(region)
    shifts = {
        (str(item["cohort_id"]), int(item["field_id"]), int(item["measurement"])): item
        for item in read_csv(manifest_dir / "registration_shifts.csv")
    }
    biological = biological_coordinates(plate, row, column)
    crop_store = cropped_store_path(run_dir, plate.name, row, column)
    crop_group = open_cropped_store(crop_store, mode="r")
    manifest_rows = []
    per_acquisition_counts = {}
    for acquisition in ACQUISITIONS:
        selection = selections[acquisition.acquisition_id]
        cohort = cohort_for_measurement(acquisition.measurement)
        source_index_path = (
            run_dir
            / "manifests"
            / "illumination"
            / "source_files"
            / f"{acquisition.acquisition_id}__{plate.name}.csv"
        )
        source_index = {
            int(item["field_id"]): item
            for item in read_csv(source_index_path)
            if int(item["row"]) == row and int(item["column"]) == column
        }
        if set(source_index) != set(range(1, 37)):
            raise RuntimeError(
                f"Incomplete source index for {acquisition.acquisition_id} {well_slug}."
            )
        acquisition_count = 0
        for region in regions_by_cohort.get(cohort.cohort_id, []):
            field_id = int(region["field_id"])
            image = read_patch_level_crop_group(
                crop_group,
                cohort.cohort_id,
                field_id,
                acquisition.acquisition_id,
            )
            shift = shifts[(cohort.cohort_id, field_id, acquisition.measurement)]
            grid_position = np.argwhere(FIELD_ID_GRID == field_id)[0]
            for patch in region_patch_plan(region, config.patch_size):
                y0 = int(patch["local_y"])
                x0 = int(patch["local_x"])
                image_patch = image[
                    y0 : y0 + config.patch_size,
                    x0 : x0 + config.patch_size,
                ]
                if image_patch.shape != (config.patch_size, config.patch_size):
                    raise RuntimeError(
                        f"Unexpected patch shape {image_patch.shape} for {well_slug}, "
                        f"field {field_id}, {cohort.cohort_id}."
                    )
                filename = patch_level_filename(
                    plate=plate,
                    row=row,
                    column=column,
                    acquisition=acquisition,
                    patch=patch,
                )
                output = (
                    run_dir
                    / "quantitative_patches"
                    / acquisition.acquisition_id
                    / filename
                )
                if not args.validate_only:
                    write_float_ome_tiff(
                        output,
                        image_patch,
                        metadata={
                            "acquisition_id": acquisition.acquisition_id,
                            "plate_name": plate.name,
                            "well": f"r{row:02d}_c{column:02d}",
                            "registration_level": "patch_level",
                            "cohort_id": cohort.cohort_id,
                            "cohort_reference_measurement": cohort.reference_measurement,
                            "field_id": field_id,
                            "patch_id": patch["patch_id"],
                            "reference_y": patch["reference_y"],
                            "reference_x": patch["reference_x"],
                        },
                        compression=config.storage.tiff_compression,
                    )
                source = source_index[field_id]
                manifest_rows.append(
                    {
                        "acquisition_id": acquisition.acquisition_id,
                        "measurement": acquisition.measurement,
                        "source_channel": acquisition.source_channel,
                        "fluorophore": acquisition.fluorophore,
                        "marker": acquisition.marker,
                        "plate_name": plate.name,
                        "repeat": biological["repeat"],
                        "cell_line_row": biological["cell_line_row"],
                        "well_within_repeat": biological["well_within_repeat"],
                        "row": row,
                        "column": column,
                        "registration_level": "patch_level",
                        "cohort_id": cohort.cohort_id,
                        "cohort_reference_measurement": cohort.reference_measurement,
                        "field_id": field_id,
                        "field_grid_row": int(grid_position[0]) + 1,
                        "field_grid_column": int(grid_position[1]) + 1,
                        "patch_id": patch["patch_id"],
                        "patch_index": patch["patch_index"],
                        "patch_row": patch["patch_row"],
                        "patch_column": patch["patch_column"],
                        "field_reference_y": patch["field_reference_y"],
                        "field_reference_x": patch["field_reference_x"],
                        "reference_y": patch["reference_y"],
                        "reference_x": patch["reference_x"],
                        "patch_size": config.patch_size,
                        "source_field_ids": str(field_id),
                        "source_image_filenames": source["source_filename"],
                        "source_image_paths": source["source_path"],
                        "source_index_path": str(source_index_path),
                        "correction_method": selection["chosen_method"],
                        "correction_model_sha256": selected_model_hash(selection),
                        "registration_shift_y": shift["shift_y"],
                        "registration_shift_x": shift["shift_x"],
                        "quantitative_path": str(output),
                    }
                )
                acquisition_count += 1
        per_acquisition_counts[acquisition.acquisition_id] = acquisition_count

    expected_count = int(well_status["expected_quantitative_patch_files"])
    if len(manifest_rows) != expected_count:
        raise RuntimeError(
            f"Patch-level manifest count mismatch for {well_slug}: "
            f"expected {expected_count}, generated {len(manifest_rows)}."
        )
    write_csv(manifest_dir / "patches.csv", manifest_rows)
    retained_patch_ids = sorted(
        {
            patch["patch_id"]
            for region in regions
            for patch in region_patch_plan(region, config.patch_size)
        }
    )
    write_json(
        stage_status_path,
        {
            "stage": "patch_generation",
            "registration_level": "patch_level",
            "status": "validated" if args.validate_only else "complete",
            "plate_name": plate.name,
            "test_mode": config.test_mode,
            "row": row,
            "column": column,
            "retained_patch_ids": retained_patch_ids,
            "quarantined_patch_ids": well_status["quarantined_patch_ids"],
            "per_acquisition_patch_counts": per_acquisition_counts,
            "written_patch_count": 0 if args.validate_only else len(manifest_rows),
            "replaced_previous_output_count": len(replaced_outputs),
            "task_hash": task_hash,
            "execution_mode": "validation_only" if args.validate_only else "production",
            "environment": task_environment(task_started, monotonic_start),
        },
    )
    print(
        f"{well_slug}: {len(retained_patch_ids)} retained field/cohort patch geometries; "
        f"{len(manifest_rows)} acquisition patches"
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
