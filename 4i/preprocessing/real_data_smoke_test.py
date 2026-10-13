#!/usr/bin/env python3
"""Run one end-to-end in-memory corrected DAPI registration per plate."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from register_and_crop import register_measurement
from utils.config import load_config
from utils.correction import (
    apply_model,
    load_model,
    model_load_mode,
    selected_model_hash,
)
from utils.layout import (
    FIELD_SHAPE,
    index_channel_directory,
    parse_field_path,
    paths_for_well,
    tile_fields,
)
from utils.manifests import (
    approved_selections,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.provenance import (
    start_task,
    task_environment,
    verify_code_snapshot,
)
from utils.patch_level import PATCH_LEVEL_COHORTS, analyze_patch_level_dapi
from utils.qc_visualization import (
    save_crop_outline,
    save_field_qc_heatmaps,
    save_registration_overlay,
    save_patch_level_region_heatmaps,
)
from utils.registration import (
    apply_translation,
    bounds_area,
    centered_patch_crop,
    intersect_bounds,
    valid_bounds_for_shift,
    validate_registered_fields,
)
from utils.registry import ACQUISITIONS, PLATES, resolve_data_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--task-index", type=int)
    result.add_argument("--count-tasks", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def load_corrected_fields(paths, method, model, batch_size: int) -> np.ndarray:
    corrected = []
    for start in range(0, len(paths), batch_size):
        images = [np.asarray(iio.imread(path)) for path in paths[start : start + batch_size]]
        if any(image.shape != FIELD_SHAPE for image in images):
            raise ValueError(f"Smoke-test inputs must all be {FIELD_SHAPE}.")
        corrected.append(apply_model(np.stack(images), method, model))
    result = np.concatenate(corrected).astype(np.float32, copy=False)
    if result.shape != (36, *FIELD_SHAPE) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Invalid corrected smoke stack {result.shape}.")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.count_tasks:
        print(len(PLATES))
        return 0
    if args.task_index is None or not 0 <= args.task_index < len(PLATES):
        raise ValueError(f"--task-index must be in 0..{len(PLATES) - 1}.")
    run_dir = args.run_dir.expanduser().resolve()
    verify_code_snapshot(run_dir)
    task_started, monotonic_start = start_task()
    config = load_config(run_dir / "provenance" / "pipeline_config.json")
    selections = approved_selections(
        run_dir / "provenance" / "illumination_method_selection.csv"
    )
    plate = PLATES[args.task_index]
    row, column = config.smoke_row, config.smoke_column
    data_root = resolve_data_root(config.data_root)
    output_dir = run_dir / "manifests" / "smoke" / plate.name
    status_path = output_dir / "smoke_status.json"

    indexed_by_acquisition = {}
    source_rows = []
    for acquisition in ACQUISITIONS:
        directory = acquisition.plate_directory(plate, data_root)
        indexed = index_channel_directory(directory)
        paths = paths_for_well(indexed, row, column)
        for path in paths:
            parsed = parse_field_path(path)
            if parsed["channel"] != acquisition.source_channel:
                raise ValueError(f"Channel mismatch in {path}.")
            stat = path.stat()
            source_rows.append(
                {
                    "acquisition_id": acquisition.acquisition_id,
                    "plate_name": plate.name,
                    "row": row,
                    "column": column,
                    "field_id": parsed["field"],
                    "source_path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        indexed_by_acquisition[acquisition.acquisition_id] = paths
    task_hash = sha256_json(
        {
            "stage": "real_data_smoke_test",
            "plate": plate.name,
            "row": row,
            "column": column,
            "config_sha256": sha256_file(
                run_dir / "provenance" / "pipeline_config.json"
            ),
            "selection_sha256": sha256_file(
                run_dir / "provenance" / "illumination_method_selection.csv"
            ),
            "sources": source_rows,
        }
    )
    if args.resume and status_path.is_file():
        previous = read_json(status_path)
        if previous.get("task_hash") == task_hash and previous.get("status") == "pass":
            print(f"{plate.name}: resumed verified real-data smoke test")
            return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "source_files.csv", source_rows)

    dapi_fields = {}
    correction_rows = []
    for number, acquisition in enumerate(ACQUISITIONS, start=1):
        selection = selections[acquisition.acquisition_id]
        method, model = load_model(selection)
        fields = load_corrected_fields(
            indexed_by_acquisition[acquisition.acquisition_id],
            method,
            model,
            config.correction_batch_size,
        )
        correction_rows.append(
            {
                "acquisition_id": acquisition.acquisition_id,
                "measurement": acquisition.measurement,
                "source_channel": acquisition.source_channel,
                "marker": acquisition.marker,
                "method": method,
                "model_sha256": selected_model_hash(selection),
                "model_load_mode": model_load_mode(model),
                "minimum": float(np.min(fields)),
                "maximum": float(np.max(fields)),
                "finite_fraction": float(np.mean(np.isfinite(fields))),
            }
        )
        if acquisition.is_dapi:
            dapi_fields[acquisition.measurement] = fields
        else:
            del fields
        del model
        gc.collect()
        print(f"{plate.name}: corrected smoke acquisition {number}/{len(ACQUISITIONS)}")
    write_csv(output_dir / "correction_validation.csv", correction_rows)

    if config.registration.level == "patch_level":
        shift_rows, field_rows, regions = analyze_patch_level_dapi(
            dapi_fields,
            config.registration,
            config.patch_size,
        )
        write_csv(output_dir / "registration_shifts.csv", shift_rows)
        write_csv(output_dir / "field_qc.csv", field_rows)
        write_csv(output_dir / "patch_level_regions.csv", regions)
        save_field_qc_heatmaps(
            output_dir / "figures" / "field_qc_heatmaps.png",
            field_rows,
            title=f"Smoke test {plate.name} r{row:02d}_c{column:02d} patch-level QC",
        )
        save_patch_level_region_heatmaps(
            output_dir / "figures" / "patch_level_regions.png",
            regions,
            title=f"Smoke test {plate.name} patch-level cohort crops",
        )
        retained_by_cohort = {
            cohort.cohort_id: sum(
                int(item["patch_count_retained"])
                for item in regions
                if item["cohort_id"] == cohort.cohort_id
            )
            for cohort in PATCH_LEVEL_COHORTS
        }
        passed = all(value > 0 for value in retained_by_cohort.values())
        failed_regions = [item for item in regions if item["status"] != "complete"]
        status = {
            "stage": "real_data_smoke_test",
            "registration_level": "patch_level",
            "status": "pass" if passed else "fail",
            "task_hash": task_hash,
            "plate_name": plate.name,
            "row": row,
            "column": column,
            "corrected_acquisition_count": len(correction_rows),
            "registered_dapi_field_count": len(shift_rows),
            "field_cohort_count": len(regions),
            "retained_field_cohort_count": len(regions) - len(failed_regions),
            "failed_field_cohort_count": len(failed_regions),
            "retained_patch_geometries_by_cohort": retained_by_cohort,
            "environment": task_environment(task_started, monotonic_start),
        }
        write_json(status_path, status)
        if not passed:
            raise RuntimeError(
                f"{plate.name} patch-level real-data smoke test failed; see {status_path}."
            )
        print(f"{plate.name}: patch-level real-data smoke test passed")
        return 0

    fixed_fields = dapi_fields[config.registration.reference_measurement]
    fixed_mosaic = tile_fields(fixed_fields)
    measurement_shifts = {1: (0.0, 0.0)}
    shift_rows = [
        {
            "measurement": 1,
            "shift_y": 0.0,
            "shift_x": 0.0,
            "shift_magnitude": 0.0,
            "overlap_fraction": 1.0,
            "post_registration_correlation": 1.0,
            "global_qc_status": "pass",
            "shift_source": "reference",
        }
    ]
    field_rows = [
        {"measurement": 1, **item}
        for item in validate_registered_fields(
            fixed_mosaic,
            fixed_mosaic,
            upsample_factor=config.registration.upsample_factor,
            max_residual_px=config.registration.max_field_residual_px,
            min_correlation=config.registration.min_field_correlation,
            min_normalized_std=config.registration.min_field_normalized_std,
        )
    ]
    for measurement in range(2, 13):
        result, local_rows, _, moving_mosaic = register_measurement(
            fixed_fields,
            dapi_fields[measurement],
            config.registration,
            None,
        )
        shift_rows.append({"measurement": measurement, **result})
        field_rows.extend(
            {"measurement": measurement, **item} for item in local_rows
        )
        measurement_shifts[measurement] = (result["shift_y"], result["shift_x"])
        registered = apply_translation(
            moving_mosaic, result["shift_y"], result["shift_x"]
        )
        save_registration_overlay(
            output_dir / "figures" / f"measurement_{measurement:02d}.png",
            fixed_mosaic,
            moving_mosaic,
            registered,
            title=f"Smoke test {plate.name} r{row:02d}_c{column:02d}",
            subtitle=(
                f"measurement {measurement} | y={result['shift_y']:+.2f}, "
                f"x={result['shift_x']:+.2f} | {result['shift_source']} | "
                f"QC={result['global_qc_status']}"
            ),
            downsample=config.registration.display_downsample,
        )
    write_csv(output_dir / "registration_shifts.csv", shift_rows)
    write_csv(output_dir / "field_qc.csv", field_rows)
    save_field_qc_heatmaps(
        output_dir / "figures" / "field_qc_heatmaps.png",
        field_rows,
        title=f"Smoke test {plate.name} r{row:02d}_c{column:02d}",
    )
    common_bounds = intersect_bounds(
        valid_bounds_for_shift(fixed_mosaic.shape, *measurement_shifts[m])
        for m in range(1, 13)
    )
    crop_bounds = centered_patch_crop(common_bounds, config.patch_size)
    overlap = bounds_area(common_bounds) / float(fixed_mosaic.size)
    crop_has_patch = bounds_area(crop_bounds) >= config.patch_size**2
    save_crop_outline(
        output_dir / "figures" / "common_overlap_and_crop.png",
        fixed_mosaic,
        common_bounds,
        crop_bounds,
        title=f"Smoke test {plate.name} common overlap and crop",
        downsample=config.registration.display_downsample,
    )
    failed_measurements = [
        int(item["measurement"])
        for item in shift_rows
        if item.get("global_qc_status") != "pass"
    ]
    passed = (
        not failed_measurements
        and overlap >= config.registration.min_common_overlap_fraction
        and crop_has_patch
    )
    status = {
        "stage": "real_data_smoke_test",
        "status": "pass" if passed else "fail",
        "task_hash": task_hash,
        "plate_name": plate.name,
        "row": row,
        "column": column,
        "corrected_acquisition_count": len(correction_rows),
        "registered_dapi_count": len(shift_rows),
        "failed_measurements": failed_measurements,
        "common_bounds": common_bounds,
        "common_overlap_fraction": overlap,
        "crop_bounds": crop_bounds,
        "crop_has_patch": crop_has_patch,
        "environment": task_environment(task_started, monotonic_start),
    }
    write_json(status_path, status)
    if not passed:
        raise RuntimeError(
            f"{plate.name} real-data smoke test failed; see {status_path}."
        )
    print(f"{plate.name}: real-data smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
