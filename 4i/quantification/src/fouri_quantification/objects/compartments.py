from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import disk


def union_masks(masks, shape):
    out = np.zeros(shape, dtype=bool)
    for mask in masks:
        out |= mask
    return out


def build_compartments(nuclear_masks, myotube_masks, assignments, geometry, shape):
    """Perinuclear territories are disjoint within a resident myotube."""
    all_nuclei = union_masks(nuclear_masks, shape)
    myotubes = []
    for j, mask in enumerate(myotube_masks):
        residents = union_masks([nuclear_masks[i] for i, p in assignments.items() if p == j], shape)
        myotubes.append(dict(whole_myotube=mask, cytoplasm=mask & ~all_nuclei, assigned_nuclei=mask & residents))
    rings = [np.zeros(shape, dtype=bool) for _ in nuclear_masks]
    radius = float(geometry.get("perinuclear_width_px", 4))
    if geometry.get("perinuclear_width_um") is not None:
        if not geometry.get("pixel_size_um"):
            raise ValueError("perinuclear_width_um requires pixel_size_um")
        radius = float(geometry["perinuclear_width_um"]) / geometry["pixel_size_um"]
    for j, myotube in enumerate(myotubes):
        owners = [i for i, p in assignments.items() if p == j]
        best = np.full(shape, np.inf)
        owner = np.full(shape, -1, dtype=np.int32)
        for i in owners:
            if not nuclear_masks[i].any():
                continue
            distance = distance_transform_edt(~nuclear_masks[i])
            take = (distance < best) & (distance <= radius) & myotube["cytoplasm"]
            best[take], owner[take] = distance[take], i
        for i in owners:
            rings[i] = owner == i
    rim = float(geometry.get("rim_width_px", 2))
    if geometry.get("rim_width_um") is not None:
        if not geometry.get("pixel_size_um"):
            raise ValueError("rim_width_um requires pixel_size_um")
        rim = float(geometry["rim_width_um"]) / geometry["pixel_size_um"]
    if rim < 1 or radius < 0:
        raise ValueError("Rim width must be at least one pixel and perinuclear radius nonnegative")
    nuclei = []
    for mask, ring in zip(nuclear_masks, rings):
        inside = binary_erosion(mask, structure=disk(int(round(rim))), border_value=0)
        nuclei.append(dict(whole_nucleus=mask, nuclear_interior=inside, nuclear_rim=mask & ~inside,
                           local_perinuclear_cytoplasm=ring))
    return nuclei, myotubes
