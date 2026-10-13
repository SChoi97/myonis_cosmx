"""Retain legacy decisions and add audit evidence without changing those decisions."""
from __future__ import annotations

from shapely.geometry import Polygon

from ..compatibility._geometry import assign_nuclei_to_myotubes


def assign(children, parents, threshold=0.8, relationship="nucleus_to_myotube"):
    mapping = assign_nuclei_to_myotubes(children, parents, threshold)
    candidates, rows = [], []
    parent_geometries = [Polygon(p["polygon"]) for p in parents]
    for i, child in enumerate(children):
        geom = Polygon(child["polygon"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        overlaps = []
        if geom.area > 0:
            for j, parent in enumerate(parents):
                x0, y0, x1, y1 = child["bbox"]
                a0, b0, a1, b1 = parent["bbox"]
                if x0 > a1 or x1 < a0 or y0 > b1 or y1 < b0:
                    continue
                try:
                    overlap = float(geom.intersection(parent_geometries[j]).area / geom.area)
                except Exception:
                    continue  # Same invalid-parent behavior as the original routine.
                if overlap > 0:
                    overlaps.append((overlap, j))
                    candidates.append(dict(child_uid=child["object_uid"], parent_uid=parent["object_uid"],
                                           relationship=relationship, overlap_fraction=overlap,
                                           selected=bool(mapping.get(i) == j), qualifies=bool(overlap > threshold)))
        overlaps.sort(key=lambda pair: -pair[0])  # Stable order for equal overlaps.
        best = overlaps[0][0] if overlaps else 0.0
        second = overlaps[1][0] if len(overlaps) > 1 else 0.0
        parent = mapping.get(i)
        rows.append(dict(child_uid=child["object_uid"], parent_uid=parents[parent]["object_uid"] if parent is not None else "",
                         relationship=relationship, status="assigned" if parent is not None else "unassigned",
                         overlap_fraction=best, second_overlap_fraction=second, threshold=float(threshold),
                         competing_parent=bool(second > threshold), overlap_tie=bool(len(overlaps) > 1 and best == second)))
    return mapping, rows, candidates
