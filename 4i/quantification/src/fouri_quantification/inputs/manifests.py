"""Build an explicit image/asset index before any feature work starts."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pandas as pd

from .metadata import parse_filename, SPATIAL_COLUMNS
from ..diagnostics import progress


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase(path, root, subtree="quantitative_patches"):
    path = Path(path)
    if subtree in path.parts:
        candidate = Path(root).joinpath(*path.parts[path.parts.index(subtree):])
        return str(candidate.resolve())
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def selected(meta, selection):
    for option, key in [("plates", "plate_name"), ("wells", "well_id"), ("fields", "field_id"),
                        ("cohorts", "cohort_id"), ("repeats", "repeat")]:
        values = selection.get(option, "all")
        if values != "all" and values is not None:
            if not isinstance(values, list):
                raise ValueError(f"selection.{option} must be a list or 'all'")
            if str(meta[key]) not in {str(v).replace("-", "_") if option == "cohorts" else str(v) for v in values}:
                return False
    return True


def build_index(config):
    prep = config["preprocessing"]
    raw = pd.read_csv(prep["image_manifest"], dtype=str, keep_default_na=False)
    required = {"quantitative_path", "acquisition_id", "patch_size"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Image manifest lacks {required - set(raw.columns)}")
    issues, images = [], []
    for row in progress(raw.to_dict("records"), len(raw), "Index images", "image"):
        meta = parse_filename(row["quantitative_path"], config["dataset_id"], config["plate_layout_data"])
        if not selected(meta, config["selection"]):
            continue
        suffix = "dapi" if meta["marker"] == "dapi" else f"{meta['fluorophore']}_{meta['marker']}"
        expected_acquisition = f"measurement_{meta['measurement']:02d}_{suffix}"
        if row["acquisition_id"] != expected_acquisition:
            raise ValueError(f"Filename/manifest conflict for acquisition_id: {row['quantitative_path']}")
        checks = {"row": "well_row", "column": "well_column", "plate_name": "plate_name", "repeat": "repeat",
                  "field_id": "field_id", "cohort_id": "cohort_id", "patch_index": "patch_index",
                  "measurement": "measurement", "source_channel": "source_channel",
                  "field_reference_x": "field_reference_x", "field_reference_y": "field_reference_y",
                  "reference_x": "reference_x", "reference_y": "reference_y"}
        for manifest_key, parsed_key in checks.items():
            if row.get(manifest_key, "") != "":
                source_value = row[manifest_key].replace("-", "_") if parsed_key == "cohort_id" else row[manifest_key]
                expected = meta[parsed_key]
                if isinstance(expected, int):
                    source_value = int(source_value)
                if source_value != expected:
                    raise ValueError(f"Filename/manifest conflict for {manifest_key}: {row['quantitative_path']}")
        meta.update(acquisition_id=row["acquisition_id"], patch_size=int(row["patch_size"]),
                    original_path=row["quantitative_path"], resolved_path=rebase(row["quantitative_path"], prep["root"]),
                    legacy_cell_line_row=row.get("cell_line_row", ""),
                    correction_method=row.get("correction_method", ""))
        images.append(meta)
    if not images:
        raise ValueError("No images match the dataset selection")
    images = pd.DataFrame(images)
    # Limit independent field/cohort units globally after plate/well/field filters.
    # Hash-based ordering is reproducible regardless of manifest row order.
    selection = config["selection"]
    eligible = sorted(images.field_key.unique())
    limit = selection.get("max_fields")
    ranked = eligible
    if selection.get("field_sampling", "random") == "random":
        seed = selection.get("field_seed", 0)
        ranked = sorted(eligible, key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    chosen = ranked[:limit] if limit is not None else ranked
    images = images[images.field_key.isin(chosen)].copy()
    images["field_selection_rank"] = images.field_key.map({key: i+1 for i, key in enumerate(chosen)})
    field_selection = dict(eligible_fields=len(eligible), selected_fields=len(chosen), requested_max_fields=limit,
                           sampling=selection.get("field_sampling", "random"), seed=selection.get("field_seed", 0))
    if limit is not None and limit > len(eligible):
        issues.append(dict(severity="info", source="selection", status="fewer_fields_available",
                           detail=f"Requested up to {limit}; using all {len(eligible)} eligible field-cohorts"))
    if images.duplicated(["spatial_key", "acquisition_id"]).any():
        raise ValueError("Duplicate acquisition/spatial keys in image manifest")
    # All channels inside a spatial key must share the exact output geometry.
    for key, group in images.groupby("spatial_key", sort=False):
        for col in ["repeat", "patch_size", "field_reference_x", "field_reference_y", "reference_x", "reference_y"]:
            if group[col].nunique() != 1:
                raise ValueError(f"Inconsistent {col} across acquisitions for {key}")
    panel = config["panel"]
    markers = []
    requested = config["selection"].get("markers", "all")
    declared = panel.get("markers", {})
    if requested != "all" and not set(requested).issubset(declared):
        raise ValueError(f"Unknown requested marker keys: {set(requested) - set(declared)}")
    for key, spec in declared.items():
        if not spec.get("enabled", True) or (requested != "all" and key not in requested):
            continue
        # All DAPI rounds remain image inputs for registration/persistence QC,
        # but only the segmentation reference is a quantified DAPI marker.
        if spec["acquisition_id"].endswith("_dapi"):
            continue
        spec = dict(spec, key=key)
        if "__" in key:
            raise ValueError("Marker keys cannot contain '__'")
        match = re.match(r"measurement_(\d+)_", spec["acquisition_id"])
        if not match:
            raise ValueError(f"Invalid acquisition_id for {key}")
        round_number = int(match.group(1))
        inside_cohort = any(int(c.split('_')[0][1:]) <= round_number <= int(c.split('_')[1]) for c in images.cohort_id.unique())
        if not inside_cohort:
            issues.append(dict(severity="info", source=key, status="outside_selection", detail=spec["acquisition_id"]))
            continue
        spec.setdefault("nuclear_compartments", ["whole_nucleus"])
        spec.setdefault("myotube_compartments", ["whole_myotube", "cytoplasm"])
        spec.setdefault("texture", config["texture"]["enabled"])
        spec.setdefault("structure_parent", "nucleus")
        spec.setdefault("structure_overlap_threshold", 0.8)
        if spec["structure_parent"] not in {"nucleus", "nucleus_then_myotube", "myotube"}:
            raise ValueError(f"Unknown structure_parent for {key}")
        if not 0 <= spec["structure_overlap_threshold"] <= 1:
            raise ValueError(f"Invalid structure overlap threshold for {key}")
        allowed_n = {"whole_nucleus", "nuclear_rim", "nuclear_interior", "local_perinuclear_cytoplasm"}
        allowed_m = {"whole_myotube", "cytoplasm", "assigned_nuclei"}
        if set(spec["nuclear_compartments"]) - allowed_n or set(spec["myotube_compartments"]) - allowed_m:
            raise ValueError(f"Unknown compartments for {key}")
        markers.append(spec)
    if panel.get("include_dapi", True):
        for acq in [config["segmentations"]["nuclei"]["acquisition_id"]]:
            if not acq.endswith("_dapi"):
                raise ValueError("include_dapi requires a DAPI nucleus reference acquisition")
            markers.append(dict(key="dapi_r" + acq.split("_")[1], acquisition_id=acq,
                                nuclear_compartments=["whole_nucleus"], myotube_compartments=["whole_myotube", "assigned_nuclei"],
                                texture=config["texture"]["enabled"]))
    if len({m["key"] for m in markers}) != len(markers) or len({m["acquisition_id"] for m in markers}) != len(markers):
        raise ValueError("Marker keys and acquisition IDs must be unique")
    if not markers:
        raise ValueError("No marker acquisitions are selected for measurement")
    requested_qc = config.get("visualisation", {}).get("markers", "all")
    qc_keys = {m["key"] for m in markers}
    if config["nuclear_aberration"]["enabled"]:
        qc_keys.add("nuclear_aberration")
    if requested_qc != "all" and set(requested_qc) - qc_keys:
        raise ValueError("visualisation.markers contains unselected or unknown marker keys")
    # Quantization limits are fixed for an acquisition, independent of individual objects.
    display = Path(prep["display_manifest"]) if prep.get("display_manifest") else None
    ranges = {}
    if display and display.is_file():
        clipping = pd.read_csv(display)
        for acq, group in clipping.groupby("acquisition_id"):
            pairs = group[["min_16bit", "max_16bit"]].drop_duplicates()
            if len(pairs) != 1:
                raise ValueError(f"Inconsistent display/texture limits for acquisition {acq}")
            ranges[acq] = pairs.iloc[0].astype(float).tolist()
    for marker in markers:
        marker["texture_range"] = marker.get("texture_range", ranges.get(marker["acquisition_id"], config["texture"]["fallback_range"]))
        low, high = marker["texture_range"]
        if not float("-inf") < low < high < float("inf"):
            raise ValueError(f"Invalid texture range for {marker['key']}")
    for spatial, group in images.groupby("spatial_key", sort=False):
        present = set(group.acquisition_id)
        for marker in markers:
            if marker["acquisition_id"] not in present:
                issues.append(dict(severity="error", source=marker["key"], status="missing_acquisition", detail=spatial))
    # Reject cross-cohort reuse of reference masks before any work is dispatched.
    ref_acqs = {name: config["segmentations"][name]["acquisition_id"] for name in ["nuclei", "myotubes"]}
    for field, group in images.groupby("field_key", sort=False):
        for name, acq in ref_acqs.items():
            if acq not in set(group.acquisition_id):
                raise ValueError(f"{field}: reference {name} acquisition {acq} is absent. Select its cohort; cross-cohort registration is not implicit.")
    images = images.sort_values([*SPATIAL_COLUMNS, "acquisition_id"]).reset_index(drop=True)
    source_specs = {name: config["segmentations"][name] for name in ["nuclei", "myotubes"]}
    for name, spec in config["segmentations"].get("structures", {}).items():
        source_specs["structures." + name] = spec
    aberr = config["nuclear_aberration"]
    if aberr["enabled"]:
        for i, directory in enumerate(aberr["prediction_directories"]):
            source_specs[f"aberration.{i}"] = dict(directory=directory, pattern=aberr.get("pattern", "*.png"), acquisition_id=aberr["acquisition_id"])
    explicit = {}
    if config.get("asset_map"):
        for row in pd.read_csv(config["asset_map"], keep_default_na=False).to_dict("records"):
            for key in ["source_key", "path", "image_name"]:
                if not row.get(key):
                    raise ValueError("asset_map requires source_key,path,image_name")
            explicit.setdefault(row["source_key"], []).append(row)
    assets = []
    for source_key, spec in source_specs.items():
        targets = images[images.acquisition_id == spec.get("acquisition_id")]
        if targets.empty:
            if spec.get("directory"):
                issues.append(dict(severity="info", source=source_key, status="outside_selection", detail=str(spec.get("acquisition_id"))))
            continue
        directory = spec.get("directory")
        paths = explicit.get(source_key)
        if paths is None:
            paths = [dict(path=str(p), image_name=p.name) for p in sorted(Path(directory).glob(spec.get("pattern", "*.txt")))] if directory and Path(directory).is_dir() else []
        by_key = {}
        target_by_spatial = {row["spatial_key"]: row for row in targets.to_dict("records")}
        for item in paths:
            actual = Path(item["path"])
            if not actual.is_absolute():
                actual = Path(prep["root"]) / actual
            meta = parse_filename(item["image_name"], config["dataset_id"], config["plate_layout_data"])
            if meta["spatial_key"] not in target_by_spatial:
                continue
            target = target_by_spatial[meta["spatial_key"]]
            for col in ["measurement", "source_channel", "marker", "fluorophore", "field_reference_x", "field_reference_y", "reference_x", "reference_y"]:
                if meta[col] != target[col]:
                    raise ValueError(f"{source_key}: incompatible {col} in {item['path']}")
            if meta["spatial_key"] in by_key:
                raise ValueError(f"Duplicate {source_key} assets for {meta['spatial_key']}")
            by_key[meta["spatial_key"]] = str(actual.resolve())
        for target in targets.to_dict("records"):
            asset = by_key.get(target["spatial_key"], "")
            status = "present" if asset and Path(asset).is_file() else ("not_configured" if not directory and source_key not in explicit else "missing")
            assets.append(dict(source_key=source_key, spatial_key=target["spatial_key"], field_key=target["field_key"],
                               acquisition_id=spec.get("acquisition_id", ""), resolved_path=asset, status=status,
                               patch_index=int(target["patch_index"]), image_name=target["image_name"],
                               segmentation_id=spec.get("segmentation_id", source_key)))
            if status != "present":
                issues.append(dict(severity="error" if source_key in ref_acqs else "warning", source=source_key,
                                   status=status, detail=target["spatial_key"]))
    assets = pd.DataFrame(assets)
    if config.get("sample_metadata"):
        extra = pd.read_csv(config["sample_metadata"], keep_default_na=False)
        keys = ["plate_name", "well_id"]
        if extra.duplicated(keys).any():
            raise ValueError("Duplicate sample metadata keys")
        existing = (set(extra.columns) & set(images.columns)) - set(keys)
        for col in existing:
            checked = images[keys + [col]].merge(extra[keys + [col]], on=keys, suffixes=("_derived", "_provided"))
            if (checked[col + "_derived"].astype(str) != checked[col + "_provided"].astype(str)).any():
                raise ValueError(f"Sample metadata conflicts with filename-derived {col}")
        images = images.merge(extra.drop(columns=list(existing)), on=keys, how="left", validate="many_to_one")
    from .nucleus_qc import load_nucleus_qc
    nucleus_qc = load_nucleus_qc(config, images, assets, file_hash)
    if config.get("nucleus_qc", {}).get("directory"):
        covered = set(nucleus_qc["patches"].get("spatial_key", []))
        missing = len(set(images.spatial_key)-covered)
        if missing:
            issues.append(dict(severity="warning", source="nucleus_qc", status="qc_not_assessed",
                               detail=f"{missing} selected reference patches have no joinable QC export; nuclei will retain unknown persistence"))
        qc_patches = nucleus_qc["patches"]
        if len(qc_patches) and qc_patches.qc_patch_status.ne("complete").any():
            issues.append(dict(severity="warning", source="nucleus_qc", status="qc_incomplete",
                               detail="Some reference patches have incomplete/failed QC; their object persistence is unknown"))
    hash_inputs = config["execution"]["hash_inputs"]
    from ..registration import input_rows
    registration_inputs = pd.DataFrame(input_rows(config, images), columns=[
        "field_key", "spatial_key", "round", "patch_index", "path", "image_name"])
    fingerprints = {}
    paths = set(images.resolved_path) | {p for p in assets.resolved_path if p}
    paths.update(nucleus_qc["files"])
    paths.update(registration_inputs.path)
    paths.update(config.get("registration_qc", {}).get("source_files", {}))
    if config["myhc_filter"].get("enabled"):
        paths.add(config["myhc_filter"]["metadata_csv"])
    for path in progress(sorted(paths), len(paths), "Check input files", "file"):
        p = Path(path)
        if p.is_file():
            stat = p.stat()
            fingerprints[path] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                      sha256=file_hash(p) if hash_inputs else "")
        else:
            fingerprints[path] = dict(missing=True)
    images["available"] = images.resolved_path.map(lambda p: not fingerprints[p].get("missing", False))
    for path in registration_inputs.path:
        if fingerprints[path].get("missing"):
            issues.append(dict(severity="error", source="registration_qc", status="missing_dapi_png", detail=path))
    for row in images[~images.available].to_dict("records"):
        issues.append(dict(severity="error", source=row["acquisition_id"], status="missing_image", detail=row["resolved_path"]))
    for marker in markers:
        source = marker.get("structure_source")
        if source and "structures." + source not in source_specs:
            raise ValueError(f"{marker['key']} refers to undeclared structure source {source}")
        if source and source_specs["structures."+source]["acquisition_id"] != marker["acquisition_id"]:
            raise ValueError(f"{marker['key']} structure source acquisition differs from its intensity acquisition")
    return dict(images=images, assets=assets, markers=markers, fingerprints=fingerprints, field_selection=field_selection,
                nucleus_qc=nucleus_qc, registration_inputs=registration_inputs,
                issues=pd.DataFrame(issues, columns=["severity", "source", "status", "detail"]))
