#!/usr/bin/env python3
"""Stage 1: apply one approved illumination model to one acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import imageio.v3 as iio
import numpy as np

from utils.config import load_config
from utils.correction import (
    apply_model,
    load_model,
    model_load_mode,
    selected_model_hash,
)
from utils.layout import (
    FIELD_SHAPE,
    assert_expected_wells,
    complete_wells,
    index_channel_directory,
    parse_field_path,
    paths_for_well,
)
from utils.manifests import (
    approved_selections,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.registry import ACQUISITIONS, PLATES, resolve_data_root, wells_for_plate
from utils.storage import corrected_store_path, create_corrected_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def load_stack(paths: list[Path]) -> np.ndarray:
    images = [np.asarray(iio.imread(path)) for path in paths]
    shapes = {image.shape for image in images}
    if len(shapes) != 1 or any(image.ndim != 2 for image in images):
        raise ValueError(f"Inconsistent non-2D images: {shapes}.")
    return np.stack(images)


def correct_plate(
    run_dir: Path,
    acquisition,
    plate,
    source_directory: Path,
    method: str,
    model,
    selection: dict[str, str],
    config,
    selected_wells: tuple[tuple[int, int], ...],
    *,
    validate_only: bool,
    resume: bool,
) -> dict:
    indexed = index_channel_directory(source_directory)
    available_wells = complete_wells(indexed)
    if not config.test_mode:
        assert_expected_wells(available_wells)
    wells = tuple(selected_wells)
    unavailable = sorted(set(wells) - set(available_wells))
    if unavailable:
        raise ValueError(
            f"Configured wells are absent or incomplete in {source_directory}: {unavailable}"
        )
    selected_coordinates = set(wells)
    source_records = []
    for (row, column, field), path in sorted(indexed.items()):
        if (row, column) not in selected_coordinates:
            continue
        stat = path.stat()
        parsed = parse_field_path(path)
        if parsed["channel"] != acquisition.source_channel:
            raise ValueError(
                f"{path} encodes channel {parsed['channel']}, but "
                f"{acquisition.acquisition_id} requires channel "
                f"{acquisition.source_channel}."
            )
        source_records.append(
            {
                "plate_name": plate.name,
                "row": row,
                "column": column,
                "field_id": field,
                "timepoint": parsed["timepoint"],
                "encoded_channel": parsed["channel"],
                "source_filename": path.name,
                "source_path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    source_sizes = [int(item["size_bytes"]) for item in source_records]
    source_mtimes = [int(item["mtime_ns"]) / 1e9 for item in source_records]
    output_path = corrected_store_path(run_dir, acquisition.acquisition_id, plate.name)
    success_path = output_path.parent / f"{plate.name}_SUCCESS.json"
    source_index_path = (
        run_dir
        / "manifests"
        / "illumination"
        / "source_files"
        / f"{acquisition.acquisition_id}__{plate.name}.csv"
    )
    write_csv(source_index_path, source_records)
    task_hash = sha256_json(
        {
            "stage": "illumination_correction",
            "acquisition_id": acquisition.acquisition_id,
            "plate_name": plate.name,
            "method": method,
            "model_sha256": selected_model_hash(selection),
            "config_sha256": sha256_file(
                run_dir / "provenance" / "pipeline_config.json"
            ),
            "source_files": source_records,
            "selected_wells": [list(item) for item in wells],
        }
    )
    if resume and success_path.is_file() and output_path.is_dir():
        previous = read_json(success_path)
        if previous.get("task_hash") == task_hash:
            return {
                **previous,
                "status": "resumed",
                "resumed_from_status": previous.get("status"),
            }
    if validate_only:
        first = np.asarray(iio.imread(paths_for_well(indexed, *wells[0])[0]))
        if first.shape != FIELD_SHAPE:
            raise ValueError(
                f"Expected {FIELD_SHAPE} fields; received {first.shape}."
            )
        return {
            "plate_name": plate.name,
            "status": "validated",
            "source_directory": str(source_directory),
            "well_count": len(wells),
            "available_source_well_count": len(available_wells),
            "field_shape": list(first.shape),
            "source_index_path": str(source_index_path),
            "task_hash": task_hash,
        }
    if output_path.exists():
        shutil.rmtree(output_path)
    first_path = paths_for_well(indexed, *wells[0])[0]
    first = np.asarray(iio.imread(first_path))
    if first.shape != FIELD_SHAPE:
        raise ValueError(f"Expected {FIELD_SHAPE} fields; received {first.shape}.")
    group, output = create_corrected_store(
        output_path,
        wells=wells,
        field_shape=tuple(first.shape),
        acquisition_metadata={
            **acquisition.as_row(),
            "plate_name": plate.name,
            "source_directory": str(source_directory),
            "correction_method": method,
            "model_sha256": selected_model_hash(selection),
            "task_hash": task_hash,
        },
        chunk_shape=(config.storage.chunk_y, config.storage.chunk_x),
        compression_level=config.storage.compression_level,
    )
    minimum = float("inf")
    maximum = float("-inf")
    for well_index, (row, column) in enumerate(wells):
        paths = paths_for_well(indexed, row, column)
        for start in range(0, 36, config.correction_batch_size):
            batch_paths = paths[start : start + config.correction_batch_size]
            corrected = apply_model(load_stack(batch_paths), method, model)
            output[well_index, start : start + len(batch_paths)] = corrected
            minimum = min(minimum, float(np.min(corrected)))
            maximum = max(maximum, float(np.max(corrected)))
    group.attrs["finite_validation"] = True
    group.attrs["value_minimum"] = minimum
    group.attrs["value_maximum"] = maximum
    result = {
        "plate_name": plate.name,
        "status": "complete",
        "store": str(output_path),
        "source_directory": str(source_directory),
        "well_count": len(wells),
        "available_source_well_count": len(available_wells),
        "field_count": len(wells) * 36,
        "source_total_bytes": int(sum(source_sizes)),
        "source_mtime_min": min(source_mtimes),
        "source_mtime_max": max(source_mtimes),
        "field_shape": list(first.shape),
        "source_index_path": str(source_index_path),
        "task_hash": task_hash,
        "minimum": minimum,
        "maximum": maximum,
    }
    write_json(success_path, result)
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.count_tasks:
        print(len(ACQUISITIONS))
        return 0
    if args.task_index is None or not 0 <= args.task_index < len(ACQUISITIONS):
        raise ValueError(f"--task-index must be in 0..{len(ACQUISITIONS) - 1}.")
    run_dir = args.run_dir.expanduser().resolve()
    verify_code_snapshot(run_dir)
    task_started, monotonic_start = start_task()
    config = load_config(run_dir / "provenance" / "pipeline_config.json")
    selections = approved_selections(
        run_dir / "provenance" / "illumination_method_selection.csv"
    )
    acquisition = ACQUISITIONS[args.task_index]
    selection = selections[acquisition.acquisition_id]
    method, model = load_model(selection)
    load_mode = model_load_mode(model)
    data_root = resolve_data_root(config.data_root)
    results = []
    for plate in PLATES:
        results.append(
            correct_plate(
                run_dir,
                acquisition,
                plate,
                acquisition.plate_directory(plate, data_root),
                method,
                model,
                selection,
                config,
                wells_for_plate(config, plate),
                validate_only=args.validate_only,
                resume=args.resume,
            )
        )
    manifest = {
        "stage": "illumination_correction",
        "task_index": args.task_index,
        "acquisition": acquisition.as_row(),
        "method": method,
        "model_load_mode": load_mode,
        "test_mode": config.test_mode,
        "selection": selection,
        "environment": task_environment(task_started, monotonic_start),
        "plates": results,
        "status": "validated" if args.validate_only else "complete",
    }
    manifest_path = (
        run_dir
        / "manifests"
        / "illumination"
        / f"{acquisition.acquisition_id}.json"
    )
    write_json(manifest_path, manifest)
    print(f"{acquisition.acquisition_id}: {manifest['status']} ({method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
