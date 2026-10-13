"""Resolve user configuration without depending on the shell working directory."""
from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import yaml


def load_yaml(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_path(value, base):
    if value is None or value == "":
        return None
    p = Path(value).expanduser()
    return str((p if p.is_absolute() else Path(base) / p).resolve())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_config(path, troubleshoot=False, troubleshoot_timing=False, visualisation_fields=None):
    path = Path(path).resolve()
    config = copy.deepcopy(load_yaml(path))
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported configuration schema_version")
    if not config.get("dataset_id"):
        raise ValueError("dataset_id is required and must remain stable across reruns")
    prep = config.setdefault("preprocessing", {})
    if not prep.get("root"):
        raise ValueError("preprocessing.root is required")
    prep["root"] = resolve_path(prep["root"], path.parent)
    for key, default in [("image_manifest", "manifests/aggregate/patch_manifest.csv"),
                         ("display_manifest", "manifests/aggregate/clipping_manifest.csv")]:
        prep[key] = resolve_path(prep.get(key, default), prep["root"])
    config["output_root"] = resolve_path(config.get("output_root", "../results"), path.parent)
    marker_path = resolve_path(config.get("marker_config", "markers.yaml"), path.parent)
    config["marker_config"] = marker_path
    config["panel"] = load_yaml(marker_path)
    layout_path = config.get("plate_layout")
    layout = (load_yaml(resolve_path(layout_path, path.parent)) if layout_path else
              yaml.safe_load(files("fouri_quantification").joinpath("data/plate_layout.yaml").read_text()))
    config["plate_layout_data"] = layout
    config["asset_map"] = resolve_path(config.get("asset_map"), path.parent)
    config["sample_metadata"] = resolve_path(config.get("sample_metadata"), path.parent)
    nucleus_qc = config.setdefault("nucleus_qc", {})
    nucleus_qc["directory"] = resolve_path(nucleus_qc.get("directory"), prep["root"])
    nucleus_qc.setdefault("require_all", False)
    if type(nucleus_qc["require_all"]) is not bool:
        raise ValueError("nucleus_qc.require_all must be true or false")
    if nucleus_qc["require_all"] and not nucleus_qc["directory"] and not config.get("registration_qc", {}).get("enabled"):
        raise ValueError("nucleus_qc.require_all requires a directory")
    seg = config.setdefault("segmentations", {})
    if seg.get("format", "yolov8_seg") != "yolov8_seg":
        raise ValueError("Only yolov8_seg contours are supported")
    for name in ["nuclei", "myotubes"]:
        if name not in seg or not seg[name].get("directory") or not seg[name].get("acquisition_id"):
            raise ValueError(f"segmentations.{name} requires directory and acquisition_id")
    for name, source in list((n, seg[n]) for n in ["nuclei", "myotubes"]) + list(seg.get("structures", {}).items()):
        source["directory"] = resolve_path(source.get("directory"), prep["root"])
        source.setdefault("segmentation_id", name + "_v1")
        source.setdefault("pattern", seg.get("pattern", "*.txt"))
    filt = config.setdefault("myhc_filter", {})
    filt.setdefault("enabled", False)
    filt["metadata_csv"] = resolve_path(filt.get("metadata_csv"), prep["root"])
    filt.setdefault("intensity_threshold", 15)
    if filt["enabled"] and not filt["metadata_csv"]:
        raise ValueError("Enabled MyHC filtering requires its original intensity metadata CSV")
    aberr = config.setdefault("nuclear_aberration", {})
    aberr.setdefault("enabled", False)
    aberr["prediction_directories"] = [resolve_path(p, prep["root"]) for p in aberr.get("prediction_directories", [])]
    if aberr.get("ensemble", "mean") != "mean":
        raise ValueError("Compatibility mode supports the existing mean ensemble only")
    if aberr["enabled"] and (not aberr["prediction_directories"] or not aberr.get("acquisition_id")):
        raise ValueError("Enabled aberration requires prediction_directories and source acquisition_id")
    geometry = config.setdefault("geometry", {})
    if geometry.get("merge_patch_objects", False):
        raise ValueError("Patch merging is not supported: source object identities are preserved")
    geometry.setdefault("pixel_size_um", None)
    if geometry["pixel_size_um"] is not None and float(geometry["pixel_size_um"]) <= 0:
        raise ValueError("pixel_size_um must be positive")
    geometry.setdefault("rim_width_px", 2)
    geometry.setdefault("perinuclear_width_px", 4)
    geometry.setdefault("edge_threshold", 0)
    measurement = config.setdefault("measurement", {})
    measurement.setdefault("overlap_policy", "include")
    if measurement["overlap_policy"] not in ("include", "exclude"):
        raise ValueError("measurement.overlap_policy must be include or exclude")
    assignment = config.setdefault("nucleus_myotube_assignment", {})
    assignment.setdefault("overlap_threshold", 0.8)
    if not 0 <= float(assignment["overlap_threshold"]) <= 1:
        raise ValueError("overlap_threshold must be in [0, 1]")
    execution = config.setdefault("execution", {})
    execution.setdefault("workers", 1)
    execution.setdefault("allow_partial", False)
    execution.setdefault("hash_inputs", True)
    execution["troubleshoot"] = bool(troubleshoot or execution.get("troubleshoot", False))
    execution["troubleshoot_timing"] = bool(troubleshoot_timing or execution.get("troubleshoot_timing", False) or execution["troubleshoot"])
    if int(execution["workers"]) < 1:
        raise ValueError("workers must be at least 1")
    selection = config.setdefault("selection", {})
    selection.setdefault("max_fields", None)
    selection.setdefault("field_sampling", "random")
    selection.setdefault("field_seed", 0)
    limit = selection["max_fields"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("selection.max_fields must be a positive integer or null (all)")
    if selection["field_sampling"] not in {"random", "first"}:
        raise ValueError("selection.field_sampling must be random or first")
    if type(selection["field_seed"]) is not int or selection["field_seed"] < 0:
        raise ValueError("selection.field_seed must be a nonnegative integer")
    vis = config.setdefault("visualisation", {})
    vis.setdefault("enabled", False)
    vis.setdefault("max_fields", 100)
    vis.setdefault("field_seed", 0)
    vis.setdefault("markers", "all")
    vis.setdefault("max_patches_per_field", 1)
    vis.setdefault("dpi", 120)
    vis.setdefault("max_labels_per_panel", 40)
    vis.setdefault("export_masks", False)
    vis.setdefault("export_audits", False)
    if visualisation_fields is not None:
        vis["max_fields"] = visualisation_fields
        vis["enabled"] = visualisation_fields > 0
    if execution["troubleshoot"]:
        vis.update(enabled=True, markers="all", max_patches_per_field=None, export_masks=True, export_audits=True)
    if vis["export_masks"]:
        vis["export_audits"] = True  # TSV maps packed mask keys to object IDs.
    limit = vis["max_fields"]
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("visualisation.max_fields must be a nonnegative integer or null (all)")
    if type(vis["field_seed"]) is not int or vis["field_seed"] < 0:
        raise ValueError("visualisation.field_seed must be a nonnegative integer")
    if vis["markers"] != "all" and (not isinstance(vis["markers"], list) or not vis["markers"]):
        raise ValueError("visualisation.markers must be all or a nonempty list of marker keys")
    limit = vis["max_patches_per_field"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("visualisation.max_patches_per_field must be a positive integer or null")
    if type(vis["dpi"]) is not int or not 60 <= vis["dpi"] <= 300:
        raise ValueError("visualisation.dpi must be an integer between 60 and 300")
    if type(vis["max_labels_per_panel"]) is not int or vis["max_labels_per_panel"] < 0:
        raise ValueError("visualisation.max_labels_per_panel must be a nonnegative integer")
    outputs = config.setdefault("outputs", {})
    outputs.setdefault("pdf", True)
    outputs.setdefault("qc_overlays", 0)
    outputs.setdefault("object_h5ad", False)
    outputs.setdefault("measurements_h5ad", True)
    texture = config.setdefault("texture", {})
    texture.setdefault("enabled", True)
    texture.setdefault("levels", 32)
    texture.setdefault("distances_px", [1, 3])
    texture.setdefault("min_pixels", 16)
    texture.setdefault("min_pairs", 16)
    texture.setdefault("fallback_range", [0, 65535])
    if not 2 <= int(texture["levels"]) <= 256:
        raise ValueError("texture.levels must be between 2 and 256")
    if any(int(d) != d or int(d) < 1 for d in texture["distances_px"]):
        raise ValueError("Texture distances must be positive integer pixels")
    from .registration import configure
    configure(config, path.parent)
    config["config_path"] = str(path)
    return config
