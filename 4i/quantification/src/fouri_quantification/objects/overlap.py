"""Measurement masks are derived from, and never replace, source object masks."""
from __future__ import annotations

import numpy as np

from ..compatibility._aberration import _rasterize_polygon_indices
from .compartments import union_masks


def _measure(mask, shared, policy):
    area = int(mask.sum())
    overlap = int(np.count_nonzero(mask & shared))
    roi = mask & ~shared if policy == "exclude" else mask
    return roi, dict(overlap_area_px2=overlap,
                     overlap_fraction=overlap/area if area else np.nan,
                     measurement_area_px2=int(roi.sum()),
                     measurement_empty=not bool(roi.any()))


def measurement_masks(masks, shape, policy="include"):
    """Compare peers on one patch/grid; exclude every pixel owned by 2+ peers.

    Call separately for nuclei, myotubes and each structure segmentation source.
    Parent/child containment is not an overlap conflict.
    """
    if policy not in {"include", "exclude"}:
        raise ValueError("measurement.overlap_policy must be include or exclude")
    occupancy = np.zeros(shape, dtype=np.uint32)
    for mask in masks:
        occupancy += mask
    shared = occupancy > 1
    measured, diagnostics = [], []
    for mask in masks:
        roi, diagnostic = _measure(mask, shared, policy)
        measured.append(roi)
        diagnostics.append(diagnostic)
    return measured, diagnostics


def iter_object_measurements(objects, shape, policy):
    """Rasterize puncta once; retain compact indices, not N full patch masks.

    The occupancy grid is shared; only the current object's dense masks are
    materialized. This matters for fields with thousands of small structures.
    """
    if policy not in {"include", "exclude"}:
        raise ValueError("measurement.overlap_policy must be include or exclude")
    occupancy = np.zeros(shape, dtype=np.uint32)
    indices = []
    for obj in objects:
        rr, cc = _rasterize_polygon_indices(obj["local_polygon"], shape)
        pixels = np.ravel_multi_index((rr, cc), shape)
        occupancy.ravel()[pixels] += 1
        indices.append(pixels)
    shared = occupancy > 1
    for pixels in indices:
        original = np.zeros(shape, dtype=bool)
        original.ravel()[pixels] = True
        measured, diagnostics = _measure(original, shared, policy)
        yield original, measured, diagnostics


def restrict_compartments(nuclear, myotube, nmasks, mmasks, assignments, shape):
    """Trim existing compartments; do not create artificial rims or new cytoplasm.

    Rims, interiors and perinuclear territories have already been derived from
    original masks. Removed nuclear pixels must never become cytoplasmic pixels.
    """
    for i, compartments in enumerate(nuclear):
        for name in ("whole_nucleus", "nuclear_rim", "nuclear_interior"):
            compartments[name] = compartments[name] & nmasks[i]
        parent = assignments.get(i)
        if parent is not None:
            compartments["local_perinuclear_cytoplasm"] &= mmasks[parent]
    for j, compartments in enumerate(myotube):
        for name in compartments:
            compartments[name] = compartments[name] & mmasks[j]
        residents = union_masks([nmasks[i] for i, p in assignments.items() if p == j], shape)
        compartments["assigned_nuclei"] &= residents
