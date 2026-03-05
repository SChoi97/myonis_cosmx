from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from shapely.geometry import Polygon
except ImportError as e:  # pragma: no cover - runtime dependency check
    raise ImportError(
        "shapely is required for de-duplication. Install with: pip install shapely"
    ) from e

# Make sure sibling utils are importable when this module is used as a script dependency
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from utils.cosmx_visualisation_utils import visualise_labels


def parse_yolov8_seg_file(mask_path: Path, patch_size: int) -> List[Dict]:
    """
    Parse a YOLOv8 segmentation txt file.

    Returns a list of dicts with:
        cls: str
        coords: List[float]  (normalized xy list)
        polygon: np.ndarray  (N,2) pixel coords
    """
    objects: List[Dict] = []
    with open(mask_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                # malformed line
                continue
            cls = parts[0]
            coords = [float(v) for v in parts[1:]]
            poly = np.array(coords, dtype=np.float32).reshape(-1, 2) * float(
                patch_size
            )
            objects.append({"cls": cls, "coords": coords, "polygon": poly})
    return objects


def _make_polygon(poly_np: np.ndarray) -> Polygon:
    poly = Polygon(poly_np)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def deduplicate_objects(
    objects: List[Dict], iou_threshold: float, object_threshold: int
) -> Tuple[List[Dict], Dict[int, List[int]], List[int]]:
    """
    Apply rule-based de-duplication.

    Rules:
      - For each larger object, find smaller objects that are (almost) entirely inside
        it based on containment >= iou_threshold.
      - If a larger object contains 1 or 2 smaller objects, remove those smaller ones.
      - If it contains >= object_threshold smaller objects, remove the larger object
        UNLESS the larger object is the single largest polygon in the mask set
        (helps keep an outer shell when inner holes are present); in that case,
        keep the largest object and drop its enclosed smaller objects.

    Returns:
      kept_objects, containment_map, removed_indices
    """
    if not objects:
        return [], {}, []

    polys = [_make_polygon(o["polygon"]) for o in objects]
    areas = [p.area for p in polys]

    containment_map: Dict[int, List[int]] = {i: [] for i in range(len(objects))}
    largest_idx = int(np.argmax(areas)) if areas else -1
    for small_idx in range(len(objects)):
        for large_idx in range(len(objects)):
            if small_idx == large_idx:
                continue
            if areas[small_idx] >= areas[large_idx]:
                continue  # only consider strictly smaller vs larger

            inter_area = polys[small_idx].intersection(polys[large_idx]).area
            if inter_area <= 0:
                continue

            containment = inter_area / max(areas[small_idx], 1e-8)
            if containment >= iou_threshold:
                containment_map[large_idx].append(small_idx)

    to_remove: set[int] = set()
    for large_idx, inner_list in containment_map.items():
        if len(inner_list) >= object_threshold:
            if large_idx == largest_idx:
                # Prefer keeping the largest shell; drop enclosed holes instead
                to_remove.update(inner_list)
            else:
                to_remove.add(large_idx)  # drop the big one
        elif len(inner_list) >= 1:
            to_remove.update(inner_list)  # drop the enclosed smaller ones

    kept_indices = [i for i in range(len(objects)) if i not in to_remove]
    kept_objects = [objects[i] for i in kept_indices]
    removed_indices = sorted(list(to_remove))
    return kept_objects, containment_map, removed_indices


def write_yolov8_seg_file(out_path: Path, objects: List[Dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for obj in objects:
            coord_str = " ".join(f"{v:.6f}" for v in obj["coords"])
            f.write(f"{obj['cls']} {coord_str}\n")


def find_image_for_mask(
    mask_path: Path, image_dir: Path | None, candidate_exts: List[str]
) -> Path | None:
    stem = mask_path.stem
    search_dirs = [image_dir] if image_dir is not None else [mask_path.parent]
    for d in search_dirs:
        if d is None:
            continue
        for ext in candidate_exts:
            candidate = d / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def save_visualisation(
    image_path: Path,
    polygons: List[np.ndarray],
    out_path: Path,
    alpha: float = 0.35,
    linewidth: int = 2,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[warn] Unable to read image {image_path}; skipping visualisation")
        return
    contours_int = [np.asarray(p, dtype=np.int32) for p in polygons]
    overlay = visualise_labels(
        img,
        contours=contours_int,
        alpha=alpha,
        linewidth=linewidth,
        target_gene_counts=None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def run_deduplication(
    contour_path: Path,
    iou_threshold: float = 0.9,
    object_threshold: int = 3,
    patch_size: int = 1024,
    image_dir: Path | None = None,
    image_ext: str | None = ".png",
    savepath: Path | None = None,
) -> Tuple[int, int]:
    """
    Execute de-duplication over a directory or single mask file.

    Returns:
      total_files_processed, total_removed_objects
    """
    contour_path = contour_path.resolve()
    if not contour_path.exists():
        raise FileNotFoundError(f"contour_path not found: {contour_path}")

    if contour_path.is_dir():
        mask_files = sorted(contour_path.glob("*.txt"), key=lambda p: p.name)
        inferred_root = contour_path
    else:
        mask_files = [contour_path]
        inferred_root = contour_path.parent

    out_root = savepath.resolve() if savepath is not None else inferred_root

    out_contour_dir = out_root / "contours"
    out_vis_dir = out_root / "visualisation"
    candidate_exts = [image_ext] if image_ext else []
    candidate_exts.extend([".png", ".jpg", ".jpeg", ".tif", ".tiff"])

    print(
        f"[info] Found {len(mask_files)} mask file(s). "
        f"IOU threshold={iou_threshold}, object_threshold={object_threshold}"
    )

    total_removed = 0
    for mask_file in tqdm(mask_files, desc="Deduplicating masks"):
        objects = parse_yolov8_seg_file(mask_file, patch_size)
        kept, containment_map, removed_indices = deduplicate_objects(
            objects, iou_threshold, object_threshold
        )

        out_mask_path = out_contour_dir / mask_file.name
        write_yolov8_seg_file(out_mask_path, kept)

        img_path = find_image_for_mask(mask_file, image_dir, candidate_exts)
        if img_path:
            vis_out = out_vis_dir / f"{mask_file.stem}.png"
            save_visualisation(img_path, [o["polygon"] for o in kept], vis_out)
        else:
            print(f"[warn] No matching image found for {mask_file.name}; skipping overlay")

        total_removed += len(removed_indices)
        print(
            f"[done] {mask_file.name}: kept {len(kept)}/{len(objects)} "
            f"(removed {len(removed_indices)})"
        )

    print(f"[summary] Removed {total_removed} object(s) across {len(mask_files)} file(s)")
    return len(mask_files), total_removed


__all__ = [
    "parse_yolov8_seg_file",
    "deduplicate_objects",
    "write_yolov8_seg_file",
    "find_image_for_mask",
    "save_visualisation",
    "run_deduplication",
]

