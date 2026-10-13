#!/usr/bin/env python3
"""Stage 2: register each well with DAPI, quarantine local failures, and crop."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from utils.config import load_config
from utils.correction import selected_model_hash
from utils.layout import tile_fields
from utils.manifests import (
    approved_selections,
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.patches import patch_plan, quarantined_patch_ids
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.qc_visualization import (
    save_crop_outline,
    save_field_qc_heatmaps,
    save_registration_overlay,
)
from utils.registration import (
    apply_translation,
    bounds_area,
    centered_patch_crop,
    estimate_translation,
    field_candidates,
    global_qc,
    intersect_bounds,
    robust_consensus_shift,
    valid_bounds_for_shift,
    validate_registered_fields,
)
from utils.registry import (
    ACQUISITIONS,
    acquisitions_for_measurement,
    configured_well_tasks,
    dapi_for_measurement,
)
from utils.storage import (
    corrected_store_path,
    create_cropped_store,
    cropped_store_path,
    read_corrected_well,
    write_crop,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def override_for(
    rows: list[dict[str, str]], plate_name: str, row: int, column: int, measurement: int
) -> dict[str, str] | None:
    matches = [
        item
        for item in rows
        if item.get("plate_name") == plate_name
        and int(item.get("row", -1)) == row
        and int(item.get("column", -1)) == column
        and int(item.get("measurement", -1)) == measurement
        and not str(item.get("field_id", "")).strip()
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate registration overrides for {plate_name} r{row:02d}_c{column:02d} "
            f"measurement {measurement}."
        )
    return matches[0] if matches else None


def register_measurement(
    fixed_fields: np.ndarray,
    moving_fields: np.ndarray,
    settings,
    override: dict[str, str] | None,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    fixed_mosaic = tile_fields(fixed_fields)
    moving_mosaic = tile_fields(moving_fields)
    result = estimate_translation(
        fixed_mosaic,
        moving_mosaic,
        downsample=settings.downsample,
        upsample_factor=settings.upsample_factor,
    )
    result["shift_source"] = "whole_well_phase_correlation"
    checked = global_qc(
        fixed_mosaic,
        moving_mosaic,
        result,
        max_shift_px=settings.max_global_shift_px,
        min_overlap_fraction=settings.min_common_overlap_fraction,
        min_correlation=settings.min_global_correlation,
    )
    initial_diagnostics = {
        f"initial_{key}": checked.get(key)
        for key in (
            "shift_y",
            "shift_x",
            "shift_magnitude",
            "phase_error",
            "phase_difference",
            "overlap_fraction",
            "post_registration_correlation",
            "global_qc_status",
            "global_qc_reasons",
        )
    }
    candidates = []
    if checked["global_qc_status"] == "fail":
        candidates = field_candidates(
            fixed_fields,
            moving_fields,
            upsample_factor=settings.upsample_factor,
            min_normalized_std=settings.min_field_normalized_std,
        )
        consensus = robust_consensus_shift(
            candidates,
            tolerance_px=settings.consensus_tolerance_px,
            minimum_fields=settings.consensus_min_fields,
        )
        if consensus is not None:
            consensus["shift_source"] = "robust_field_consensus"
            checked = global_qc(
                fixed_mosaic,
                moving_mosaic,
                consensus,
                max_shift_px=settings.max_global_shift_px,
                min_overlap_fraction=settings.min_common_overlap_fraction,
                min_correlation=settings.min_global_correlation,
            )
    if override:
        action = override.get("action", "").strip().lower()
        if action == "exclude":
            checked["global_qc_status"] = "fail"
            checked["global_qc_reasons"] = "manual_exclusion"
            checked["override_action"] = action
        elif action == "replace":
            replacement = {
                "shift_y": float(override["shift_y"]),
                "shift_x": float(override["shift_x"]),
                "shift_magnitude": float(
                    np.hypot(float(override["shift_y"]), float(override["shift_x"]))
                ),
                "phase_error": float("nan"),
                "phase_difference": float("nan"),
                "fft_shape": None,
                "shift_source": "manual_replacement",
            }
            checked = global_qc(
                fixed_mosaic,
                moving_mosaic,
                replacement,
                max_shift_px=settings.max_global_shift_px,
                min_overlap_fraction=settings.min_common_overlap_fraction,
                min_correlation=settings.min_global_correlation,
            )
            checked["override_action"] = action
        elif action == "accept":
            checked["global_qc_status"] = "pass"
            checked["global_qc_reasons"] = ""
            checked["override_action"] = action
        elif action:
            raise ValueError(f"Unknown override action {action!r}.")
    field_rows = validate_registered_fields(
        fixed_mosaic,
        checked["registered"],
        upsample_factor=settings.upsample_factor,
        max_residual_px=settings.max_field_residual_px,
        min_correlation=settings.min_field_correlation,
        min_normalized_std=settings.min_field_normalized_std,
    )
    candidate_by_field = {int(item["field_id"]): item for item in candidates}
    for field_row in field_rows:
        candidate = candidate_by_field.get(int(field_row["field_id"]))
        field_row.update(
            {
                "fallback_candidate_informative": (
                    candidate.get("informative") if candidate else None
                ),
                "fallback_candidate_shift_y": (
                    candidate.get("shift_y") if candidate else None
                ),
                "fallback_candidate_shift_x": (
                    candidate.get("shift_x") if candidate else None
                ),
                "fallback_candidate_magnitude": (
                    candidate.get("shift_magnitude") if candidate else None
                ),
            }
        )
    checked.update(initial_diagnostics)
    checked.pop("registered")
    return checked, field_rows, fixed_mosaic, moving_mosaic


def task_input_hash(
    run_dir: Path, plate_name: str, row: int, column: int
) -> str:
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
            "stage": "registration_and_crop",
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
        from register_patch_level import run as run_patch_level

        return run_patch_level(args, run_dir, config)
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
            and (
                previous.get("whole_well_quarantine")
                or output_store.is_dir()
            )
        )
        if reusable:
            print(f"{well_slug}: resumed verified {previous.get('status')} result")
            return 0
    overrides_path = run_dir / "provenance" / "registration_overrides.csv"
    override_rows = read_csv(overrides_path)
    fixed_acquisition = dapi_for_measurement(config.registration.reference_measurement)
    fixed_fields = read_corrected_well(
        corrected_store_path(run_dir, fixed_acquisition.acquisition_id, plate.name),
        row,
        column,
    )
    fixed_mosaic = tile_fields(fixed_fields)
    shifts = []
    field_qc_rows = []
    measurement_shifts: dict[int, tuple[float, float]] = {1: (0.0, 0.0)}
    reference_field_rows = validate_registered_fields(
        fixed_mosaic,
        fixed_mosaic,
        upsample_factor=config.registration.upsample_factor,
        max_residual_px=config.registration.max_field_residual_px,
        min_correlation=config.registration.min_field_correlation,
        min_normalized_std=config.registration.min_field_normalized_std,
    )
    for field_row in reference_field_rows:
        field_qc_rows.append({"measurement": 1, **field_row})
    shifts.append(
        {
            "measurement": 1,
            "shift_y": 0.0,
            "shift_x": 0.0,
            "shift_magnitude": 0.0,
            "phase_error": 0.0,
            "phase_difference": 0.0,
            "overlap_fraction": 1.0,
            "post_registration_correlation": 1.0,
            "global_qc_status": "pass",
            "global_qc_reasons": "",
            "shift_source": "reference",
        }
    )
    global_failure = False
    for measurement in range(2, 13):
        acquisition = dapi_for_measurement(measurement)
        moving_fields = read_corrected_well(
            corrected_store_path(run_dir, acquisition.acquisition_id, plate.name),
            row,
            column,
        )
        result, field_rows, _, moving_mosaic = register_measurement(
            fixed_fields,
            moving_fields,
            config.registration,
            override_for(override_rows, plate.name, row, column, measurement),
        )
        result_row = {"measurement": measurement, **result}
        shifts.append(result_row)
        measurement_shifts[measurement] = (result["shift_y"], result["shift_x"])
        registered = apply_translation(moving_mosaic, result["shift_y"], result["shift_x"])
        for field_row in field_rows:
            field_qc_rows.append({"measurement": measurement, **field_row})
        global_failure = global_failure or result["global_qc_status"] != "pass"
        save_registration_overlay(
            manifest_dir / "figures" / f"measurement_{measurement:02d}_registration.png",
            fixed_mosaic,
            moving_mosaic,
            registered,
            title=f"{plate.name} r{row:02d}_c{column:02d} | measurement {measurement}",
            subtitle=(
                f"shift y={result['shift_y']:+.2f}, x={result['shift_x']:+.2f} px | "
                f"source={result['shift_source']} | QC={result['global_qc_status']}"
            ),
            downsample=config.registration.display_downsample,
        )
    save_field_qc_heatmaps(
        manifest_dir / "figures" / "field_qc_heatmaps.png",
        field_qc_rows,
        title=f"{plate.name} r{row:02d}_c{column:02d} field QC",
    )
    common_bounds = intersect_bounds(
        valid_bounds_for_shift(fixed_mosaic.shape, *measurement_shifts[m])
        for m in range(1, 13)
    )
    common_overlap_fraction = bounds_area(common_bounds) / float(fixed_mosaic.size)
    crop_bounds = centered_patch_crop(common_bounds, config.patch_size)
    crop_has_patch = bounds_area(crop_bounds) >= config.patch_size**2
    crop_area_fraction = bounds_area(crop_bounds) / float(fixed_mosaic.size)
    whole_well_quarantine = (
        global_failure
        or common_overlap_fraction < config.registration.min_common_overlap_fraction
        or not crop_has_patch
    )
    failed_fields = {
        int(item["field_id"]) for item in field_qc_rows if item["status"] == "fail"
    }
    plan = patch_plan(crop_bounds, config.patch_size) if crop_has_patch else []
    quarantined_ids = quarantined_patch_ids(plan, failed_fields)
    quarantine_rows = [
        {
            "plate_name": plate.name,
            "row": row,
            "column": column,
            "scope": "whole_well" if whole_well_quarantine else "patch",
            "patch_id": patch["patch_id"],
            "failed_field_ids": ";".join(map(str, sorted(failed_fields))),
            "reason": (
                "global_registration_or_overlap_failure"
                if whole_well_quarantine
                else "intersects_failed_field"
            ),
        }
        for patch in plan
        if whole_well_quarantine or patch["patch_id"] in quarantined_ids
    ]
    write_csv(manifest_dir / "registration_shifts.csv", shifts)
    write_csv(manifest_dir / "field_qc.csv", field_qc_rows)
    write_csv(manifest_dir / "quarantine.csv", quarantine_rows)
    save_crop_outline(
        manifest_dir / "figures" / "common_overlap_and_crop.png",
        fixed_mosaic,
        common_bounds,
        crop_bounds,
        title=f"{plate.name} r{row:02d}_c{column:02d} overlap and crop",
        downsample=config.registration.display_downsample,
    )
    status = {
        "stage": "registration_and_crop",
        "registration_level": "field_level",
        "task_index": args.task_index,
        "test_mode": config.test_mode,
        "plate_name": plate.name,
        "row": row,
        "column": column,
        "status": "quarantined" if whole_well_quarantine else "complete",
        "whole_well_quarantine": whole_well_quarantine,
        "failed_field_ids": sorted(failed_fields),
        "quarantined_patch_ids": sorted(quarantined_ids),
        "common_bounds": common_bounds,
        "common_overlap_fraction": common_overlap_fraction,
        "crop_bounds": crop_bounds,
        "crop_area_fraction": crop_area_fraction,
        "patch_size": config.patch_size,
        "patch_count_total": len(plan),
        "patch_count_retained": 0 if whole_well_quarantine else len(plan) - len(quarantined_ids),
        "expected_quantitative_patch_files": (
            0
            if whole_well_quarantine
            else (len(plan) - len(quarantined_ids)) * len(ACQUISITIONS)
        ),
        "task_hash": input_hash,
        "execution_mode": "validation_only" if args.validate_only else "production",
        "environment": task_environment(task_started, monotonic_start),
    }
    if args.validate_only or whole_well_quarantine:
        if whole_well_quarantine and not args.validate_only and output_store.exists():
            shutil.rmtree(output_store)
        write_json(status_path, status)
        print(f"{well_slug}: {status['status']}")
        return 0
    if output_store.exists():
        shutil.rmtree(output_store)
    group = create_cropped_store(
        output_store,
        {
            "plate_name": plate.name,
            "row": row,
            "column": column,
            "reference_measurement": 1,
            "registration_level": "field_level",
            "common_bounds": list(common_bounds),
            "crop_bounds": list(crop_bounds),
        },
    )
    y0, y1, x0, x1 = crop_bounds
    for acquisition in ACQUISITIONS:
        selection = selections[acquisition.acquisition_id]
        fields = read_corrected_well(
            corrected_store_path(run_dir, acquisition.acquisition_id, plate.name),
            row,
            column,
        )
        mosaic = tile_fields(fields)
        shift_y, shift_x = measurement_shifts[acquisition.measurement]
        registered = mosaic if acquisition.measurement == 1 else apply_translation(
            mosaic, shift_y, shift_x
        )
        write_crop(
            group,
            acquisition.acquisition_id,
            registered[y0:y1, x0:x1],
            metadata={
                **acquisition.as_row(),
                "shift_y": shift_y,
                "shift_x": shift_x,
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
        f"{well_slug}: complete; retained {status['patch_count_retained']}/"
        f"{status['patch_count_total']} patches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
