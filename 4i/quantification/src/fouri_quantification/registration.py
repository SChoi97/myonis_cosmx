"""Run the original patch QC first, then reuse its DAPI translations everywhere.

Transforms map source patch coordinates to the reference DAPI patch. They are
never estimated from marker channels and never applied to input files on disk.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .qc import run_4i_nucleus_segmentation_qc as qc
from .qc.myonis_4i_qc_export import sha256, source_snapshot, export_myonis_4i


def configure(config, yaml_dir):
    from .config import resolve_path
    options = config.setdefault("registration_qc", {})
    defaults = dict(enabled=False, fast_mode=True, metric="ssim", threshold=0.35,
                    aggregation="max", smoothing_sigma_px=1.0, min_valid_fraction=0.90,
                    local_window=11, texture_floor=0.01, comparison_gain_limits=[0.01, 100.0],
                    compare_metrics=False, registration_config=None, tools_dir=None)
    for key, value in defaults.items():
        options.setdefault(key, value)
    for key in ("enabled", "fast_mode", "compare_metrics"):
        if type(options[key]) is not bool:
            raise ValueError(f"registration_qc.{key} must be true or false")
    if options["metric"] not in {"pixel_max", "pixel_mean", "relative", "relative_calibrated", "ssim", "zncc_signal", "gradient"}:
        raise ValueError("Unknown registration_qc.metric")
    if options["aggregation"] not in {"mean", "max"}:
        raise ValueError("registration_qc.aggregation must be mean or max")
    if options["metric"].startswith("pixel_") and options["aggregation"] != options["metric"].removeprefix("pixel_"):
        raise ValueError("registration_qc pixel_mean/pixel_max must match aggregation")
    for key, low, high in [("threshold", 0, 1), ("min_valid_fraction", 0, 1)]:
        value = float(options[key])
        if not math.isfinite(value) or not low <= value <= high or (key == "min_valid_fraction" and value == 0):
            raise ValueError(f"Invalid registration_qc.{key}")
    for key in ["smoothing_sigma_px", "texture_floor"]:
        value = float(options[key])
        if not math.isfinite(value) or value < 0 or (key == "texture_floor" and value == 0):
            raise ValueError(f"Invalid registration_qc.{key}")
    window = options["local_window"]
    if type(window) is not int or window < 3 or window % 2 != 1:
        raise ValueError("registration_qc.local_window must be an odd integer >= 3")
    gains = options["comparison_gain_limits"]
    if len(gains) != 2 or not all(math.isfinite(float(v)) for v in gains) or not 0 < gains[0] <= 1 <= gains[1]:
        raise ValueError("Invalid registration_qc.comparison_gain_limits")
    if not options["enabled"]:
        return
    if config.get("nucleus_qc", {}).get("directory"):
        raise ValueError("Use either registration_qc.enabled or an external nucleus_qc.directory, not both")
    match = re.fullmatch(r"measurement_(\d+)_dapi", config["segmentations"]["nuclei"]["acquisition_id"])
    if not match:
        raise ValueError("Integrated registration requires a DAPI nucleus reference acquisition")
    options["reference_round"] = int(match[1])
    options["tools_dir"] = resolve_path(options["tools_dir"], yaml_dir) or str(qc.default_tools_dir())
    options["registration_config"] = resolve_path(options["registration_config"], config["preprocessing"]["root"])
    args = argparse.Namespace(tools_dir=Path(options["tools_dir"]), run_dir=Path(config["preprocessing"]["root"]),
                              registration_config=Path(options["registration_config"]) if options["registration_config"] else None)
    qc.prepare_registration_settings(args)
    options["registration_config"] = str(args.registration_config)
    options["registration_settings"] = args.registration_settings
    # Include transitive preprocessing helpers, not just the registration entry point.
    root = args.tools_dir/"myonis_cosmx/4i/preprocessing/utils"
    options["source_files"] = {str(p): sha256(p) for p in sorted(root.glob("*.py"))}
    options["source_files"][str(args.registration_config)] = args.registration_config_sha256


def input_rows(config, images):
    """Pin exactly the selected patches' DAPI PNGs; no second field sampling."""
    if not config.get("registration_qc", {}).get("enabled"):
        return []
    rows = []
    ref = config["registration_qc"]["reference_round"]
    for spatial, group in images.groupby("spatial_key", sort=True):
        cohort = group.iloc[0].cohort_id
        first, last = map(int, cohort.removeprefix("m").split("_"))
        expected = set(range(first, last+1))
        dapi = group[group.marker == "dapi"]
        if ref not in expected or set(dapi.measurement) != expected:
            raise ValueError(f"Registration QC requires every DAPI round in {cohort} for {spatial}")
        for entry in dapi.to_dict("records"):
            path = Path(config["preprocessing"]["root"])/"png_8bit"/entry["acquisition_id"]/(entry["image_name"]+".png")
            rows.append(dict(field_key=entry["field_key"], spatial_key=spatial, round=int(entry["measurement"]),
                             patch_index=entry["patch_index"], path=str(path), image_name=entry["image_name"]))
    return rows


def verify_sources(config):
    for path, checksum in config.get("registration_qc", {}).get("source_files", {}).items():
        if not Path(path).is_file() or sha256(path) != checksum:
            raise ValueError(f"Preprocessing registration code/config changed after prepare: {path}")


def _args(config, output):
    options = config["registration_qc"]
    # Start from the standalone parser's defaults to preserve every scoring and
    # rendering option; the dataset always determines its reference and selection.
    args = qc.parse_args(["--datapath", config["preprocessing"]["root"], "--output-dir", str(output),
                          "--tools-dir", options["tools_dir"], "--registration-metric", options["metric"],
                          "--registration-aggregation", options["aggregation"]])
    args.reference_round = options["reference_round"]
    args.fast_mode = options["fast_mode"]
    args.quiet = True
    args.registration_compare_metrics = options["compare_metrics"]
    args.registration_smoothing_sigma = options["smoothing_sigma_px"]
    args.registration_min_valid_fraction = options["min_valid_fraction"]
    args.registration_local_window = options["local_window"]
    args.registration_texture_floor = options["texture_floor"]
    args.registration_comparison_gain_limits = options["comparison_gain_limits"]
    for key in ["change", "ssim", "zncc", "gradient", "active"]:
        setattr(args, f"registration_{key}_threshold", options["threshold"])
    args.registration_config = Path(options["registration_config"])
    args.registration_config_sha256 = options["source_files"][options["registration_config"]]
    args.registration_settings = options["registration_settings"]
    args.registration_source_hashes = {Path(p).name: digest for p, digest in options["source_files"].items()}
    return args


def run_field(config, task, output, timer=None):
    """QC all selected patches in this field before any feature calculation."""
    from .inputs.nucleus_qc import load_nucleus_qc
    from .diagnostics import StageTimer
    timer = timer or StageTimer()
    output = Path(output)
    imported_config = copy.deepcopy(config)
    imported_config["nucleus_qc"] = dict(directory=str(output/"myonis_4i"), require_all=True)

    def import_flags():
        return load_nucleus_qc(imported_config, pd.DataFrame(task["images"]), pd.DataFrame(task["assets"]), sha256)

    cache_path = output/"complete.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if (cached["fingerprint"] == task["fingerprint"] and cached["field_key"] == task["field_key"] and
                    all((output/p).is_file() and sha256(output/p) == digest for p, digest in cached["files"].items())):
                return pd.DataFrame(json.loads((output/"transforms.json").read_text())), import_flags()
        except (ValueError, KeyError, OSError):
            pass  # Recompute invalid/missing QC artifacts, never silently reuse them.
    args = _args(config, output)
    qc.initialize_worker(args.tools_dir)
    ref = args.reference_round
    by_patch = {}
    for row in task["registration_inputs"]:
        by_patch.setdefault(row["spatial_key"], {})[row["round"]] = row
    contours = {a["spatial_key"]: a["resolved_path"] for a in task["assets"] if a["source_key"] == "nuclei" and a["status"] == "present"}
    args.rounds = sorted(next(iter(by_patch.values())))
    folders = ["assignments"]
    if not args.fast_mode:
        folders += ["plots", "registration_overlays", "registration_maps", "registration_scores"]
    for name in folders:
        (output/name).mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), settings=settings,
                   selected_plots=len(by_patch), selected_patch_keys=[], completed=[], failed=[], reference_sources={},
                   discovery=dict(incomplete_patches=[]))
    transforms = []
    for spatial, entries in sorted(by_patch.items()):
        paths = {r: (Path(row["path"]), Path(contours[spatial]) if r == ref else None) for r, row in entries.items()}
        key = qc.registered_patch_key(entries[ref]["path"])
        stem = "__".join(key)
        summary["selected_patch_keys"].append(stem)
        summary["reference_sources"][stem] = source_snapshot(*paths[ref])
        try:
            completed = qc.process_patch(key, paths, args, timer=timer)
            summary["completed"].append(completed)
            record = json.loads((output/completed["assignments"]).read_text())
            for r, row in sorted(entries.items()):
                reg = record["rounds"][str(r)]["registration"]
                h, w = record["shape"]
                reg = reg or dict(shift_y=0.0, shift_x=0.0, valid_bounds=[0, h, 0, w], global_qc_status="pass",
                                  global_qc_reasons="", post_registration_correlation=1.0, overlap_fraction=1.0,
                                  shift_source="reference_identity")
                y0, y1, x0, x1 = reg["valid_bounds"]
                transforms.append(dict(field_key=task["field_key"], spatial_key=spatial, patch_index=row["patch_index"],
                    round=r, reference_round=ref, shift_y=float(reg["shift_y"]), shift_x=float(reg["shift_x"]),
                    height=h, width=w, valid_y0=y0, valid_y1=y1, valid_x0=x0, valid_x1=x1,
                    status=reg["global_qc_status"], detail=reg["global_qc_reasons"],
                    correlation=reg["post_registration_correlation"], overlap_fraction=reg["overlap_fraction"],
                    source_dapi=row["path"], reference_dapi=entries[ref]["path"],
                    source_sha256=sha256(row["path"]), reference_sha256=sha256(entries[ref]["path"]),
                    direction="source_to_reference", interpolation="linear", border="missing_nan",
                    matrix_json=json.dumps([[1, 0, reg["shift_x"]], [0, 1, reg["shift_y"]], [0, 0, 1]])))
        except Exception as exc:
            summary["failed"].append(dict(patch_key=stem, error=f"{type(exc).__name__}: {exc}"))
    summary["status"] = "complete_with_errors" if summary["failed"] else "complete"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    qc.write_json(output/"summary.json", summary)
    qc.write_statistics(output, summary)
    export_myonis_4i(output, summary)
    frame = pd.DataFrame(transforms)
    frame.to_csv(output/"transforms.csv", index=False)
    qc.write_json(output/"transforms.json", transforms)
    if summary["failed"]:
        raise ValueError(f"Registration QC failed for {len(summary['failed'])} patches; field was not quantified. See {output/'summary.json'}")
    imported = import_flags()
    qc.write_json(cache_path, dict(fingerprint=task["fingerprint"], field_key=task["field_key"],
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file() and p != cache_path}))
    return frame, imported


def export_run(run_dir, parts, config, status):
    """Combine checked field outputs without recomputing any registration."""
    if not config.get("registration_qc", {}).get("enabled"):
        return
    root = Path(run_dir)/"registration_qc"
    root.mkdir(exist_ok=True)
    frames = [p["registration"] for p in parts if "registration" in p]
    (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(root/"transforms.csv", index=False)
    args = _args(config, root)
    args.rounds = sorted({int(r) for frame in frames for r in frame["round"]}) or [args.reference_round]
    summary = dict(status=status, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   completed=[], failed=[], reference_sources={}, discovery=dict(incomplete_patches=[]))
    for part in parts:
        if "registration" not in part:
            continue
        from .config import stable_hash
        task_id = stable_hash(part["field_key"])[:20]
        field = json.loads((root/task_id/"summary.json").read_text())
        for row in field["completed"]:
            row = copy.deepcopy(row)
            for key in ["assignments", "plot", "registration_overlay", "registration_map", "registration_score_plot"]:
                if row.get(key):
                    row[key] = str(Path(task_id)/row[key])
            for comparison in row.get("metric_comparisons", {}).values():
                for key in ["assignments", "plot", "overlay", "map", "scores"]:
                    if comparison.get(key):
                        comparison[key] = str(Path(task_id)/comparison[key])
            summary["completed"].append(row)
        summary["reference_sources"].update(field["reference_sources"])
    summary["selected_plots"] = len(summary["completed"])
    summary["selected_patch_keys"] = [row["patch_key"] for row in summary["completed"]]
    qc.write_json(root/"summary.json", summary)
    export_myonis_4i(root, summary)


def transform_lookup(rows):
    return {(row["spatial_key"], int(row["round"])): row for row in rows}


def source_transform(lookup, spatial, acquisition):
    if not lookup:
        return None
    return lookup[(spatial, int(acquisition.split("_")[1]))]


def translate_image(image, transform):
    """Linear, float32 translation; invalid borders are missing, never zeros.

    scipy's settings match preprocessing.apply_translation exactly. The reference
    identity is returned without interpolation, including its original borders.
    """
    from scipy.ndimage import shift
    image = np.asarray(image, dtype=np.float32)
    if transform is None:
        return image
    dy, dx = float(transform["shift_y"]), float(transform["shift_x"])
    if not dy and not dx:
        return image
    registered = shift(image, (dy, dx), order=1, mode="constant", cval=np.nan, prefilter=False)
    valid = np.zeros(image.shape, dtype=bool)
    valid[int(transform["valid_y0"]):int(transform["valid_y1"]),
          int(transform["valid_x0"]):int(transform["valid_x1"])] = True
    registered[~valid] = np.nan
    return registered


def translate_objects(objects, lookup, edge_threshold=0):
    from .compatibility._geometry import compute_bbox, is_edge_polygon
    for obj in objects:
        tr = source_transform(lookup, obj["spatial_key"], obj["segmentation_acquisition_id"])
        if tr is None:
            continue
        dx, dy = tr["shift_x"], tr["shift_y"]
        obj.update(registration_shift_x=dx, registration_shift_y=dy,
                   registration_reference_round=tr["reference_round"],
                   source_centroid_local_x=obj["centroid_local_x"], source_centroid_local_y=obj["centroid_local_y"],
                   is_edge_source=obj["is_edge"])
        obj["local_polygon"] = obj["local_polygon"] + np.array([dx, dy], dtype=np.float32)
        obj["polygon"] = obj["local_polygon"] + obj["offset"]
        obj["bbox"] = compute_bbox(obj["polygon"])
        for key in ["centroid_local_x", "centroid_field_x"]:
            obj[key] += dx
        for key in ["centroid_local_y", "centroid_field_y"]:
            obj[key] += dy
        obj["is_edge"] = bool(is_edge_polygon(obj["local_polygon"], obj["patch_size"], edge_threshold))


def registered_predictions(predictions, patch_records, acquisition, lookup):
    """Shift native float sigmoid grids without requantizing probabilities."""
    from scipy.ndimage import shift
    by_patch = {obj.get("patch_idx", obj.get("patch_index")): obj for obj in patch_records}
    for patch, prediction in list(predictions.items()):
        obj = by_patch.get(patch)
        if obj is None:
            continue
        tr = source_transform(lookup, obj["spatial_key"], acquisition)
        if tr is not None and (tr["shift_y"] or tr["shift_x"]):
            scale = np.asarray(prediction.shape)/float(obj["patch_size"])
            predictions[patch] = shift(prediction, (tr["shift_y"]*scale[0], tr["shift_x"]*scale[1]),
                                       order=1, mode="constant", cval=np.nan, prefilter=False)
    return predictions
