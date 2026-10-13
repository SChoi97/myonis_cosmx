"""Optional overlap exclusion on the native sigmoid grid; legacy code is untouched."""
from __future__ import annotations

import numpy as np

from ..compatibility._aberration import (
    _load_nuclear_aberration_predictions_by_patch,
    _scale_local_polygon_to_prediction,
    _rasterize_polygon_indices,
    _compute_object_nuclear_aberration_stats,
)
from ..objects.overlap import measurement_masks


def prediction_masks(objects, shape, patch_size, policy):
    masks = []
    for obj in objects:
        polygon = _scale_local_polygon_to_prediction(obj["local_polygon"], shape, patch_size)
        rr, cc = _rasterize_polygon_indices(polygon, shape)
        mask = np.zeros(shape, dtype=bool)
        mask[rr, cc] = True
        masks.append(mask)
    return measurement_masks(masks, shape, policy)[0]


def exclusive_stats(objects, predictions, patch_size):
    stats = {key: np.full(len(objects), np.nan, dtype=np.float32) for key in ("total", "mean", "area")}
    for patch in sorted({int(o["patch_idx"]) for o in objects}):
        prediction = predictions.get(patch)
        if prediction is None:
            continue
        indices = [i for i, obj in enumerate(objects) if int(obj["patch_idx"]) == patch]
        masks = prediction_masks([objects[i] for i in indices], prediction.shape, patch_size, "exclude")
        for i, mask in zip(indices, masks):
            area = int(mask.sum())
            stats["area"][i] = area
            # Retain the legacy empty-area convention: zero sum, undefined mean.
            total = float(prediction[mask].sum(dtype=np.float64))
            stats["total"][i] = total
            if area:
                stats["mean"][i] = total/area
    return stats


def add_exclusive_aberration_features(na, ma, nuclei, myotubes, assignments, paths, patch_size, field_key,
                                     predictions=None, provenance=None, policy="exclude"):
    if predictions is None:
        predictions, provenance = _load_nuclear_aberration_predictions_by_patch(paths, patch_size, field_key)
    na.uns["nuclear_aberration_ensemble"] = dict(provenance, overlap_policy=policy)
    ma.uns["nuclear_aberration_ensemble"] = na.uns["nuclear_aberration_ensemble"]
    stats = exclusive_stats if policy == "exclude" else _compute_object_nuclear_aberration_stats
    ns = stats(nuclei, predictions, patch_size)
    ms = stats(myotubes, predictions, patch_size)
    na.obs["nuclear_aberration_mean_sigmoid"] = ns["mean"]
    na.obs["nuclear_aberration_total_sigmoid"] = ns["total"]
    na.obs["nuclear_aberration_measurement_area_px2"] = ns["area"]
    ma.obs["nuclear_aberration_measurement_area_px2"] = ms["area"]
    totals = np.full(len(myotubes), np.nan, dtype=np.float32)
    nuclear_ratios = totals.copy()
    myotube_ratios = totals.copy()
    for j in range(len(myotubes)):
        residents = [i for i, parent in assignments.items() if parent == j]
        if not residents:
            totals[j] = 0
            continue
        values, areas = ns["total"][residents], ns["area"][residents]
        if not (np.isfinite(values).all() and np.isfinite(areas).all()):
            continue  # Preserve the complete-assigned-nuclei missing-data rule.
        total, area = values.sum(dtype=np.float64), areas.sum(dtype=np.float64)
        totals[j] = total
        if area > 0:
            nuclear_ratios[j] = total/area
            myotube_ratios[j] = ms["total"][j]/area
    ma.obs["nuclear_aberration_myotube_total_sigmoid"] = ms["total"]
    ma.obs["nuclear_aberration_myotube_mean_sigmoid"] = ms["mean"]
    ma.obs["nuclear_aberration_assigned_nuclei_total_sigmoid"] = totals
    ma.obs["nuclear_aberration_assigned_nuclei_total_per_nuclear_area"] = nuclear_ratios
    ma.obs["nuclear_aberration_myotube_total_per_nuclear_area"] = myotube_ratios
