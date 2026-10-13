"""Strict metadata parsing for Phenix 4i canonical filenames."""
from __future__ import annotations

import re
from pathlib import Path

SPATIAL_COLUMNS = ["plate_name", "well_id", "repeat", "cohort_id", "field_id", "patch_index"]
SAMPLE_COLUMNS = ["dataset_id", "plate_name", "plate_id", "repeat", "well_id", "well_row",
                  "well_column", "cell_line", "condition", "sample_id"]
PATTERNS = {
    "plate_name": r"^plate-(.*?)__", "repeat": r"__rep-(\d+)__",
    "well_row": r"__well-r(\d+)c", "well_column": r"__well-r\d+c(\d+)__",
    "field_id": r"__field-f(\d+)__", "cohort_id": r"__cohort-(.*?)__",
    "measurement": r"__measurement-(\d+)__", "source_channel": r"__source-ch(\d+)__",
    "fluorophore": r"__fluor-(.*?)__", "marker": r"__marker-(.*?)__",
    "patch_index": r"__patch_(\d+)__", "patch_row": r"__pr(\d+)-pc",
    "patch_column": r"__pr\d+-pc(\d+)__", "field_reference_y": r"__fieldrefy-(\d+)-",
    "field_reference_x": r"fieldrefx-(\d+)__", "reference_y": r"__refy-(\d+)-",
    "reference_x": r"-refx-(\d+)$",
}


def image_stem(path):
    name = Path(path).name
    for suffix in [".ome.tiff", ".ome.tif", ".tiff", ".tif", ".png", ".txt"]:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def parse_filename(path, dataset_id, layout):
    name = image_stem(path)
    result = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, name)
        if not match:
            raise ValueError(f"Cannot extract {key} from {name}")
        value = match.group(1)
        result[key] = value if key in {"plate_name", "cohort_id", "fluorophore", "marker"} else int(value)
    result["cohort_id"] = result["cohort_id"].replace("-", "_")
    repeated = re.search(r"__(m\d+_\d+)__field_(\d+)__", name)
    if not repeated or repeated.group(1) != result["cohort_id"] or int(repeated.group(2)) != result["field_id"]:
        raise ValueError(f"Conflicting repeated cohort/field tokens: {name}")
    cohort = re.fullmatch(r"m(\d+)_(\d+)", result["cohort_id"])
    if not cohort or not int(cohort.group(1)) <= result["measurement"] <= int(cohort.group(2)):
        raise ValueError(f"Measurement outside declared cohort: {name}")
    plate = layout["plates"].get(result["plate_name"])
    if plate is None or result["repeat"] not in plate["repeats"]:
        raise ValueError(f"Unknown plate or inconsistent repeat: {name}")
    rows = {int(k): v for k, v in layout["rows"].items()}
    if result["well_row"] not in rows:
        raise ValueError(f"Unmapped physical well row {result['well_row']}: {name}")
    if min(result[k] for k in ["well_column", "field_id", "patch_index", "patch_row", "patch_column"]) < 1:
        raise ValueError(f"Well/field/patch indices must be one-based: {name}")
    result.update(rows[result["well_row"]])
    result["plate_id"] = int(plate["plate_id"])
    result["dataset_id"] = dataset_id
    result["well_id"] = f"r{result['well_row']:02d}c{result['well_column']:02d}"
    result["patch_id"] = f"{result['cohort_id']}__field_{result['field_id']:02d}__patch_{result['patch_index']:04d}"
    result["sample_id"] = f"{dataset_id}|{result['plate_name']}|{result['well_id']}|rep{result['repeat']:02d}"
    result["spatial_key"] = "|".join(str(result[k]) for k in ["dataset_id", *SPATIAL_COLUMNS])
    result["field_key"] = result["spatial_key"].rsplit("|", 1)[0]
    result["image_name"] = name
    return result
