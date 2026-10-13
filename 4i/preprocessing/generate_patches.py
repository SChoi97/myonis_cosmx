#!/usr/bin/env python3
"""Stage 3: materialize aligned non-overlapping quantitative OME-TIFF patches."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.correction import selected_model_hash
from utils.config import load_config
from utils.layout import field_box
from utils.manifests import (
    approved_selections,
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.patches import patch_filename, patch_plan, rectangles_intersect
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.registry import ACQUISITIONS, biological_coordinates, configured_well_tasks
from utils.storage import cropped_store_path, read_crop, write_float_ome_tiff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def source_field_ids(bounds, shift_y: float, shift_x: float) -> list[int]:
    y0, y1, x0, x1 = bounds
    source_bounds = (
        y0 - shift_y,
        y1 - shift_y,
        x0 - shift_x,
        x1 - shift_x,
    )
    return [
        field_id
        for field_id in range(1, 37)
        if rectangles_intersect(source_bounds, field_box(field_id))
    ]


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = load_config(run_dir / "provenance" / "pipeline_config.json")
    tasks = configured_well_tasks(config)
    if args.count_tasks:
        print(len(tasks))
        return 0
    if args.task_index is None or not 0 <= args.task_index < len(tasks):
        raise ValueError(f"--task-index must be in 0..{len(tasks) - 1}.")
    if config.registration.level == "patch_level":
        from generate_patch_level_patches import run as run_patch_level

        return run_patch_level(args, run_dir, config)
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
            "stage": "patch_generation",
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
                Path(row["quantitative_path"]).is_file() for row in prior_rows
            ):
                print(f"{well_slug}: resumed verified patch outputs")
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
                "status": "skipped_quarantined_well",
                "plate_name": plate.name,
                "row": row,
                "column": column,
                "written_patch_count": 0,
                "replaced_previous_output_count": len(replaced_outputs),
                "task_hash": task_hash,
                "execution_mode": "validation_only" if args.validate_only else "production",
                "environment": task_environment(task_started, monotonic_start),
            },
        )
        print(f"{well_slug}: skipped quarantined well")
        return 0
    crop_bounds = tuple(well_status["crop_bounds"])
    plan = patch_plan(crop_bounds, config.patch_size)
    quarantined = set(well_status["quarantined_patch_ids"])
    retained = [patch for patch in plan if patch["patch_id"] not in quarantined]
    shift_rows = read_csv(manifest_dir / "registration_shifts.csv")
    shifts = {int(item["measurement"]): item for item in shift_rows}
    biological = biological_coordinates(plate, row, column)
    crop_store = cropped_store_path(run_dir, plate.name, row, column)
    manifest_rows = []
    for acquisition in ACQUISITIONS:
        selection = selections[acquisition.acquisition_id]
        image = read_crop(crop_store, acquisition.acquisition_id)
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
        shift_y = float(shifts[acquisition.measurement]["shift_y"])
        shift_x = float(shifts[acquisition.measurement]["shift_x"])
        for patch in retained:
            y0 = patch["local_y"]
            x0 = patch["local_x"]
            image_patch = image[
                y0 : y0 + config.patch_size,
                x0 : x0 + config.patch_size,
            ]
            if image_patch.shape != (config.patch_size, config.patch_size):
                raise RuntimeError(
                    f"Unexpected patch shape {image_patch.shape} for {well_slug}."
                )
            filename = patch_filename(
                plate=plate,
                row=row,
                column=column,
                acquisition=acquisition,
                patch=patch,
            )
            output = run_dir / "quantitative_patches" / acquisition.acquisition_id / filename
            if not args.validate_only:
                write_float_ome_tiff(
                    output,
                    image_patch,
                    metadata={
                        "acquisition_id": acquisition.acquisition_id,
                        "plate_name": plate.name,
                        "well": f"r{row:02d}_c{column:02d}",
                        "patch_id": patch["patch_id"],
                        "reference_y": patch["reference_y"],
                        "reference_x": patch["reference_x"],
                    },
                    compression=config.storage.tiff_compression,
                )
            patch_source_fields = source_field_ids(
                patch["bounds"], shift_y, shift_x
            )
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
                    "patch_id": patch["patch_id"],
                    "patch_index": patch["patch_index"],
                    "patch_row": patch["patch_row"],
                    "patch_column": patch["patch_column"],
                    "reference_y": patch["reference_y"],
                    "reference_x": patch["reference_x"],
                    "patch_size": config.patch_size,
                    "source_field_ids": ";".join(
                        map(str, patch_source_fields)
                    ),
                    "source_image_filenames": ";".join(
                        source_index[field_id]["source_filename"]
                        for field_id in patch_source_fields
                    ),
                    "source_image_paths": ";".join(
                        source_index[field_id]["source_path"]
                        for field_id in patch_source_fields
                    ),
                    "source_index_path": str(source_index_path),
                    "correction_method": selection["chosen_method"],
                    "correction_model_sha256": selected_model_hash(selection),
                    "registration_shift_y": shift_y,
                    "registration_shift_x": shift_x,
                    "quantitative_path": str(output),
                }
            )
    write_csv(manifest_dir / "patches.csv", manifest_rows)
    write_json(
        stage_status_path,
        {
            "stage": "patch_generation",
            "status": "validated" if args.validate_only else "complete",
            "plate_name": plate.name,
            "test_mode": config.test_mode,
            "row": row,
            "column": column,
            "retained_patch_ids": [patch["patch_id"] for patch in retained],
            "quarantined_patch_ids": sorted(quarantined),
            "per_acquisition_patch_count": len(retained),
            "written_patch_count": 0 if args.validate_only else len(manifest_rows),
            "replaced_previous_output_count": len(replaced_outputs),
            "task_hash": task_hash,
            "execution_mode": "validation_only" if args.validate_only else "production",
            "environment": task_environment(task_started, monotonic_start),
        },
    )
    print(f"{well_slug}: {len(retained)} retained patches x {len(ACQUISITIONS)} acquisitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
