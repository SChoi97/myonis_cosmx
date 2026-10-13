#!/usr/bin/env python3
"""Validate and record an immutable field-level patch run for seam experiments.

The patches stay in the existing preprocessing run. Scoring results are stored
separately, so adding or changing scores never repeats illumination correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE.parent / "preprocessing/results/runs/4i_production_20260823_200037_f6ecb24"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def workbook_rows(path: Path, sheet_name: str) -> list[dict]:
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book[sheet_name]
    iterator = sheet.values
    headers = next(iterator)
    return [dict(zip(headers, row)) for row in iterator if row[0]]


def validate_current_inputs(settings_path: Path, review_path: Path, clip_path: Path) -> dict[str, str]:
    settings = read_csv(settings_path)
    if len(settings) != 29 or len({row["acquisition_id"] for row in settings}) != 29:
        raise ValueError("marker_settings.csv must contain 29 unique supplied entries.")
    reviewed = {
        row["acquisition_id"]: row["chosen_method"]
        for sheet in ("Marker Selections", "DAPI Selections")
        for row in workbook_rows(review_path, sheet)
    }
    clipping = {row["acquisition_id"]: row for row in workbook_rows(clip_path, "clip_limits")}
    if len(reviewed) != 40 or len(clipping) != 40:
        raise ValueError("Approved method and clipping workbooks must contain all 40 acquisitions.")
    if any(value not in ("BaSiC", "Harmony-like") for value in reviewed.values()):
        raise ValueError("Every acquisition needs an approved IC choice.")
    for row in settings:
        key = row["acquisition_id"]
        if reviewed.get(key) != row["chosen_method"]:
            raise ValueError(f"Current IC workbook disagrees with supplied table: {key}.")
        if key not in clipping:
            raise ValueError(f"Current clipping workbook lacks {key}.")
        for column in ("min_16bit", "max_16bit"):
            if float(clipping[key][column]) != float(row[column]):
                raise ValueError(f"Current clipping workbook disagrees with supplied table: {key} {column}.")
    return reviewed


def validate(run_dir: Path, settings_path: Path, current_review: Path, current_clip: Path) -> dict:
    reviewed = validate_current_inputs(settings_path, current_review, current_clip)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Patch run is not complete: {manifest.get('status')!r}")
    config_path = run_dir / "provenance/pipeline_config.json"
    config = json.loads(config_path.read_text())
    if config.get("registration", {}).get("level", "field_level") != "field_level":
        raise ValueError("The cached run must use field-level registration.")
    if config.get("registration", {}).get("reference_measurement") != 1:
        raise ValueError("Expected measurement 1 as the registration reference.")
    selection_path = run_dir / "provenance/illumination_method_selection.csv"
    clip_path = run_dir / "provenance/clip_limits.xlsx"
    selection = {row["acquisition_id"]: row for row in read_csv(selection_path)}
    clipping = {row["acquisition_id"]: row for row in workbook_rows(clip_path, "clip_limits")}
    current_clipping = {row["acquisition_id"]: row for row in workbook_rows(current_clip, "clip_limits")}
    for key, stored in clipping.items():
        for column in ("min_16bit", "max_16bit"):
            if float(stored[column]) != float(current_clipping[key][column]):
                raise ValueError(f"Cached clip differs from current workbook: {key} {column}.")
    settings = read_csv(settings_path)
    if len(settings) != 29 or len({row["acquisition_id"] for row in settings}) != 29:
        raise ValueError("marker_settings.csv must contain 29 unique supplied entries.")
    for row in settings:
        key = row["acquisition_id"]
        if key not in selection or key not in clipping:
            raise ValueError(f"Missing acquisition {key} in cached run.")
        if selection[key]["chosen_method"] != row["chosen_method"]:
            raise ValueError(f"IC method mismatch for {key}.")
        for requested, stored in (("min_16bit", "min_16bit"), ("max_16bit", "max_16bit")):
            if float(row[requested]) != float(clipping[key][stored]):
                raise ValueError(f"Display clip mismatch for {key}: {requested}.")
    if len(selection) != 40 or len(clipping) != 40:
        raise ValueError("Expected all 40 acquisitions in the cached production run.")
    # The user chose to inherit DAPI 2–12 from the existing approved workbook.
    current_dapi = {key: value for key, value in reviewed.items() if key.endswith("_dapi")}
    for key, method in reviewed.items():
        if selection.get(key, {}).get("chosen_method") != method:
            raise ValueError(f"Cached IC choice differs from approved workbook: {key}.")
    for measurement in range(2, 13):
        key = f"measurement_{measurement:02d}_dapi"
        if current_dapi.get(key) != selection[key]["chosen_method"]:
            raise ValueError(f"Cached DAPI choice differs from approved workbook: {key}.")
    wells_root = run_dir / "manifests/wells"
    wells = sorted(path for path in wells_root.iterdir() if path.is_dir())
    if len(wells) != 120:
        raise ValueError(f"Expected 120 production wells; found {len(wells)}.")
    completed = quarantined = patch_files = 0
    for well in wells:
        registration = json.loads((well / "well_status.json").read_text())
        if registration.get("registration_level", "field_level") != "field_level":
            raise ValueError(f"Wrong registration level in {well}.")
        patches = json.loads((well / "patch_status.json").read_text())
        if patches["status"] == "complete":
            completed += 1
            patch_files += int(patches["written_patch_count"])
            if not (well / "patches.csv").is_file():
                raise FileNotFoundError(well / "patches.csv")
        elif patches["status"] == "skipped_quarantined_well":
            quarantined += 1
        else:
            raise ValueError(f"Incomplete patch stage in {well}: {patches['status']}.")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_run": str(run_dir.resolve()),
        "patch_run_id": run_dir.name,
        "registration_level": "field_level",
        "reference_measurement": 1,
        "acquisition_count": 40,
        "well_count": len(wells),
        "completed_wells": completed,
        "quarantined_wells": quarantined,
        "quantitative_patch_file_count_from_manifests": patch_files,
        "marker_settings": str(settings_path.resolve()),
        "marker_settings_sha256": digest(settings_path),
        "run_manifest_sha256": digest(manifest_path),
        "pipeline_config_sha256": digest(config_path),
        "selection_sha256": digest(selection_path),
        "clip_limits_sha256": digest(clip_path),
        "dapi_choices_2_to_12": {key: current_dapi[key] for key in sorted(current_dapi) if key != "measurement_01_dapi"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--settings", type=Path, default=HERE / "marker_settings.csv")
    parser.add_argument(
        "--approved-review",
        type=Path,
        default=HERE.parent / "preprocessing/outputs/illumination_method_selection/illumination_method_choices.xlsx",
    )
    parser.add_argument(
        "--current-clipping",
        type=Path,
        default=HERE.parent / "preprocessing/config/clip_limits.xlsx",
    )
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results/cache_manifest.json")
    args = parser.parse_args()
    if args.validate_inputs_only:
        validate_current_inputs(args.settings, args.approved_review, args.current_clipping)
        print("Current IC and clipping workbooks match all supplied marker settings.")
        return
    result = validate(args.run_dir, args.settings, args.approved_review, args.current_clipping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Cached {result['completed_wells']} wells and {result['quantitative_patch_file_count_from_manifests']} quantitative patch files: {args.output}")


if __name__ == "__main__":
    main()
