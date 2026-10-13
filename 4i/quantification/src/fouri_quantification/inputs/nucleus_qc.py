"""Attach reference-nucleus QC without filtering or changing object identities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import parse_filename

STATUS = {"passed", "failed", "not_evaluable", "not_assessed"}
NUMERIC = ["qc_reference_round", "qc_reference_id", "qc_score", "qc_mean_change_score", "qc_max_change_score",
           "qc_valid_area_fraction", "qc_threshold", "qc_n_comparison_rounds", "qc_n_rounds_passing", "qc_required_other_rounds"]


def _flag(value, name, missing=False):
    if value == "" and missing:
        return np.nan
    if value not in {"0", "1"}:
        raise ValueError(f"Nucleus QC {name} must be 0, 1"+(" or blank" if missing else ""))
    return float(value)


def load_nucleus_qc(config, images, assets, file_hash):
    """Read once at prepare; pass only matching rows to each field worker."""
    empty = pd.DataFrame(columns=["object_uid", "field_key"])
    result = dict(objects=empty, rounds=empty.copy(), patches=pd.DataFrame(), files=[], manifest={})
    settings = config.get("nucleus_qc", {})
    if not settings.get("directory"):
        return result
    root = Path(settings["directory"])
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported nucleus QC exchange schema")
    expected_acq = config["segmentations"]["nuclei"]["acquisition_id"]
    ref = int(manifest["settings"]["reference_round"])
    registered_rounds = manifest["settings"].get("rounds")
    filter_rounds = manifest["settings"].get("filter_rounds", registered_rounds)
    if expected_acq != f"measurement_{ref:02d}_dapi":
        raise ValueError(f"Nucleus QC reference round {ref} differs from quantification {expected_acq}; use the same reference segmentation")
    tables = {}
    paths = [str(manifest_path)]
    for name in ("nucleus_qc.csv", "round_qc.csv", "patch_qc.csv"):
        path = root/name
        if manifest.get("files", {}).get(name) != file_hash(path):
            raise ValueError(f"Nucleus QC checksum mismatch: {path}; regenerate a complete export")
        tables[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
        paths.append(str(path))
    objects, rounds, patches = (tables[name] for name in ("nucleus_qc.csv", "round_qc.csv", "patch_qc.csv"))
    object_keys = ["reference_image_name", "reference_source_line_number"]
    for frame, keys in [(objects, object_keys), (rounds, object_keys+["round"]), (patches, ["patch_key"])]:
        if not set(keys).issubset(frame.columns) or frame.duplicated(keys).any():
            raise ValueError("Missing or duplicate nucleus QC identity keys")
    targets = images[images.acquisition_id == expected_acq].set_index("spatial_key")
    target_assets = assets[assets.source_key == "nuclei"].set_index("spatial_key")
    rows, patch_rows, lookup = [], [], {}
    object_groups = {name: group for name, group in objects.groupby("reference_image_name", sort=False)}
    seen_spatial = set()
    for patch in patches.to_dict("records"):
        name = patch["reference_image_name"]
        if not name:
            continue  # Old results or unavailable reference inputs carry no joinable identity.
        meta = parse_filename(name, config["dataset_id"], config["plate_layout_data"])
        spatial = meta["spatial_key"]
        if spatial not in targets.index:
            continue
        if spatial in seen_spatial:
            raise ValueError("Duplicate nucleus QC reference patch")
        seen_spatial.add(spatial)
        target, asset = targets.loc[spatial], target_assets.loc[spatial]
        if name != target.image_name or int(patch["reference_round"]) != ref:
            raise ValueError("Nucleus QC reference image/coordinates differ from quantification")
        if asset.status != "present":
            continue  # Existing required-contour preflight reports this separately.
        if patch["reference_contour_sha256"] != file_hash(asset.resolved_path):
            raise ValueError(f"Nucleus QC contour checksum differs for {name}; use identical, unfiltered reference contours")
        source_lines = [i for i, line in enumerate(Path(asset.resolved_path).read_text().splitlines(), 1) if line.strip()]
        group = object_groups.get(name, objects.iloc[:0])
        lines = pd.to_numeric(group.reference_source_line_number, errors="raise").tolist()
        if (len(source_lines) != int(patch["reference_count"]) or sorted(lines) != source_lines):
            raise ValueError(f"Nucleus QC table does not cover every original reference line for {name}")
        if patch["height"] and (int(patch["height"]) != target.patch_size or int(patch["width"]) != target.patch_size):
            raise ValueError("Nucleus QC reference raster grid differs from quantification")
        patch_rows.append(dict(patch, field_key=meta["field_key"], spatial_key=spatial))
        reference_ids = {line: i+1 for i, line in enumerate(source_lines)}
        for row in group.to_dict("records"):
            line = int(row["reference_source_line_number"])
            if (row["reference_contour_sha256"] != patch["reference_contour_sha256"] or
                    int(row["reference_round"]) != ref or int(row["reference_id"]) != reference_ids[line] or
                    row["patch_key"] != patch["patch_key"]):
                raise ValueError("Nucleus QC object identity disagrees with its patch")
            status = row["qc_status"]
            if status not in STATUS:
                raise ValueError(f"Unknown nucleus QC status: {status}")
            edge = _flag(row["is_edge"], "is_edge", missing=True)
            persistent = _flag(row["is_persistent"], "is_persistent", missing=True)
            passed = bool(_flag(row["qc_pass"], "qc_pass"))
            if ((status == "passed") != passed or
                    (status in {"passed", "failed"} and (edge != 0 or persistent != float(passed))) or
                    (status in {"not_evaluable", "not_assessed"} and not np.isnan(persistent))):
                raise ValueError("Nucleus QC flags are inconsistent; unevaluable persistence must be blank")
            uid = f"{spatial}|{asset.segmentation_id}|nuclei|row{line}"
            imported = {k: v for k, v in row.items() if k.startswith("qc_")}
            imported.update(object_uid=uid, field_key=meta["field_key"], is_persistent=persistent,
                            qc_is_edge=edge, qc_pass=passed, qc_reference_round=ref,
                            qc_reference_id=int(row["reference_id"]), qc_reference_contour_sha256=patch["reference_contour_sha256"])
            rows.append(imported)
            first, last = map(int, meta["cohort_id"].removeprefix("m").split("_"))
            lookup[(name, str(line))] = (uid, meta["field_key"], row, range(first, last+1))
    if len(objects):
        orphan = objects[objects.reference_image_name.isin(targets.image_name) & ~objects.reference_image_name.isin([p["reference_image_name"] for p in patch_rows])]
        if len(orphan):
            raise ValueError("Nucleus QC object rows lack a matching reference patch record")
    imported_rounds = []
    for row in rounds.to_dict("records"):
        matched = lookup.get((row["reference_image_name"], row["reference_source_line_number"]))
        if matched:
            uid, field, parent, cohort_rounds = matched
            if (row["reference_contour_sha256"] != parent["reference_contour_sha256"] or int(row["reference_round"]) != ref or
                    row["reference_id"] != parent["reference_id"] or row["patch_key"] != parent["patch_key"]):
                raise ValueError("Nucleus QC round evidence has a mismatched source identity")
            if int(row["round"]) not in cohort_rounds or int(row["is_reference"]) != int(int(row["round"]) == ref):
                raise ValueError("Nucleus QC round evidence has an invalid round/reference flag")
            for flag in ["passes_round", "is_edge_round"]:
                _flag(row[flag], flag, missing=True)
            if "is_filter_round" in row:
                scored = bool(_flag(row["is_filter_round"], "is_filter_round"))
                r = int(row["round"])
                if (r == ref and scored) or (filter_rounds is not None and scored != (r != ref and r in filter_rounds)):
                    raise ValueError("Nucleus QC scored-round flag disagrees with the run settings")
                if r != ref and not scored and (row["score"] != "" or row["passes_round"] != ""):
                    raise ValueError("Excluded nucleus QC rounds must have blank scores and decisions")
            imported_rounds.append(dict(row, object_uid=uid, field_key=field))
    obj = pd.DataFrame(rows) if rows else empty.copy()
    rnd = pd.DataFrame(imported_rounds) if imported_rounds else empty.copy()
    for col in NUMERIC:
        if col in obj:
            obj[col] = pd.to_numeric(obj[col], errors="raise")
    for col in ["round", "is_reference", "is_filter_round", "score", "passes_round", "matched_contour_index", "is_edge_round"]:
        if col in rnd:
            rnd[col] = pd.to_numeric(rnd[col], errors="raise")
    assessed = set(obj.loc[obj.qc_status != "not_assessed", "object_uid"]) if len(obj) else set()
    counts = rnd.groupby("object_uid").size() if len(rnd) else pd.Series(dtype=int)
    scored_counts = rnd.groupby("object_uid").is_filter_round.sum() if len(rnd) and "is_filter_round" in rnd else None
    observed_rounds = rnd.groupby("object_uid")["round"].agg(set) if len(rnd) else pd.Series(dtype=object)
    if len(rnd) and rnd.duplicated(["object_uid", "round"]).any():
        raise ValueError("Duplicate nucleus QC round evidence")
    for row in obj.to_dict("records"):
        completed = row["object_uid"] in assessed
        expected = (len(registered_rounds) if registered_rounds is not None else int(row["qc_n_comparison_rounds"])+1) if completed else 0
        if int(counts.get(row["object_uid"], 0)) != expected:
            raise ValueError("Nucleus QC per-round evidence is incomplete")
        if completed and registered_rounds is not None and observed_rounds.get(row["object_uid"], set()) != set(registered_rounds):
            raise ValueError("Nucleus QC per-round evidence differs from the registered rounds")
        if completed and scored_counts is not None and int(scored_counts.get(row["object_uid"], 0)) != int(row["qc_n_comparison_rounds"]):
            raise ValueError("Nucleus QC scored-round count differs from the object summary")
    if settings.get("require_all", False):
        if set(targets.index) != seen_spatial or any(p["qc_patch_status"] != "complete" for p in patch_rows) or (len(obj) and obj.qc_status.eq("not_assessed").any()):
            raise ValueError("nucleus_qc.require_all requires completed QC for every selected reference patch")
    return dict(objects=obj, rounds=rnd, patches=pd.DataFrame(patch_rows), files=paths, manifest=manifest)


def attach_nucleus_qc(nuclei, rows, configured=False):
    """Flags only. No mutation of polygon, UID, source indexing or inclusion."""
    by_uid = {row["object_uid"]: row for row in (rows or [])}
    for obj in nuclei:
        obj.update(is_edge_geometry=obj["is_edge"], is_persistent=np.nan, qc_is_edge=np.nan,
                   qc_pass=False, qc_status="not_assessed", qc_exclusion_reason="not_in_qc_table" if configured else "not_configured")
        row = by_uid.get(obj["object_uid"])
        if row is not None:
            obj.update({k: v for k, v in row.items() if k not in {"object_uid", "field_key"}})
            if np.isfinite(row["qc_is_edge"]):
                obj["is_edge"] = bool(row["qc_is_edge"])
