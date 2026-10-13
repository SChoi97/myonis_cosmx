from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from ..compatibility._geometry import compute_bbox, is_edge_polygon
from ..compatibility._aberration import _rasterize_polygon_indices


def load_objects(asset, image, source_key, edge_threshold=0):
    """Preserve YOLO row order; offsets come from the manifest, never grid guesses."""
    if asset is None or asset["status"] != "present":
        return []
    size = int(image["patch_size"])
    offset = np.array([image["field_reference_x"], image["field_reference_y"]], dtype=np.float32)
    objects = []
    for line_no, line in enumerate(Path(asset["resolved_path"]).read_text().splitlines(), 1):
        values = line.split()
        if not values:
            continue
        try:
            class_id = int(values[0])
            coords = np.asarray([float(v) for v in values[1:]], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO row at {asset['resolved_path']}:{line_no}") from exc
        if len(coords) < 6 or len(coords) % 2 or not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any():
            raise ValueError(f"Invalid normalized polygon at {asset['resolved_path']}:{line_no}")
        local = coords.reshape(-1, 2) * float(size)
        polygon = local + offset
        uid = f"{image['spatial_key']}|{asset['segmentation_id']}|{source_key}|row{line_no}"
        meta_keys = ["dataset_id", "plate_name", "plate_id", "repeat", "well_id", "well_row", "well_column",
                     "cell_line", "condition", "sample_id", "cohort_id", "field_id", "field_key", "patch_id",
                     "patch_index", "patch_row", "patch_column", "spatial_key", "field_reference_x",
                     "field_reference_y", "reference_x", "reference_y", "legacy_cell_line_row"]
        meta = {k: image[k] for k in meta_keys if k in image}
        geom = Polygon(polygon)
        centroid = geom.centroid
        objects.append(dict(meta, object_uid=uid, object_type=source_key, source_object_index=len(objects),
                            source_line_number=line_no, local_id=len(objects), class_id=class_id,
                            image_name=image["image_name"], source_path=asset["resolved_path"],
                            segmentation_id=asset["segmentation_id"], segmentation_acquisition_id=asset["acquisition_id"],
                            patch_idx=int(image["patch_index"]), patch_size=size,
                            is_edge=bool(is_edge_polygon(local, size, edge_threshold)), geometry_valid=bool(geom.is_valid),
                            centroid_field_x=float(centroid.x), centroid_field_y=float(centroid.y),
                            centroid_local_x=float(centroid.x-offset[0]), centroid_local_y=float(centroid.y-offset[1]),
                            local_polygon=local, polygon=polygon, bbox=compute_bbox(polygon), offset=offset,
                            analysis_included=True, exclusion_reason=""))
    return objects


def object_mask(obj, shape):
    mask = np.zeros(shape, dtype=bool)
    rr, cc = _rasterize_polygon_indices(obj["local_polygon"], shape)
    mask[rr, cc] = True
    return mask


def metadata_row(obj):
    return {k: v for k, v in obj.items() if k not in {"polygon", "local_polygon", "bbox", "offset"}}


def packed_geometry(objects):
    points, offsets = [], [0]
    for obj in objects:
        points.extend(obj["local_polygon"].tolist())
        offsets.append(len(points))
    return dict(object_uids=np.asarray([o["object_uid"] for o in objects], dtype=str),
                points=np.asarray(points, dtype=np.float32).reshape(-1, 2), offsets=np.asarray(offsets, dtype=np.int64),
                field_offsets=np.asarray([o["offset"] for o in objects], dtype=np.float32).reshape(-1, 2))
