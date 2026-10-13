from __future__ import annotations

import numpy as np

from .intensity import intensity

SUMMARY_KEYS = ["count", "area_mean_px2", "area_median_px2", "area_max_px2", "area_std_px2",
                "mean_object_intensity", "std_object_intensity", "mean_structure_pixel_intensity",
                "integrated_structure_intensity", "occupied_area_px2", "coverage_fraction"]


def structure_summary(items, image, parent_mask, available=True):
    """items contain raster indices clipped to this parent, plus original polygon area."""
    if not available:
        return {key: np.nan for key in SUMMARY_KEYS}
    result = {key: np.nan for key in SUMMARY_KEYS}
    result["count"] = len(items)
    union = np.zeros(parent_mask.shape, dtype=bool)
    areas, means = [], []
    for item in items:
        union.ravel()[item["pixels"]] = True
        areas.append(item["area_px2"])
        means.append(item["mean_in_parent"])
    union &= parent_mask
    result["occupied_area_px2"] = int(union.sum())
    result["coverage_fraction"] = float(union.sum()/parent_mask.sum()) if parent_mask.any() else np.nan
    if items:
        result.update(area_mean_px2=float(np.mean(areas)), area_median_px2=float(np.median(areas)),
                      area_max_px2=float(np.max(areas)), area_std_px2=float(np.std(areas)))
        if np.isfinite(means).all():
            result.update(mean_object_intensity=float(np.mean(means)), std_object_intensity=float(np.std(means)))
    if image is not None:
        if union.any():
            values = intensity(image, union)
            result["mean_structure_pixel_intensity"] = values["mean"]
            result["integrated_structure_intensity"] = values["integrated"]
        else:
            result["integrated_structure_intensity"] = 0.0
    return result
