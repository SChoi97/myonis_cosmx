# Verbatim numerical routines from generate_h5ad/utils/cosmx_utils.py. See source_manifest.json.
from pathlib import Path
from typing import List, Dict, Tuple
import re
import time
import numpy as np

def extract_patch_index_from_mask(path: Path) -> int:
    m = re.search(r'patch_(\d+)', path.name)
    if not m:
        raise ValueError(f"Could not parse patch index from filename: {path}")
    return int(m.group(1))


def read_yolov8_segmentation_polygons(mask_path: Path, patch_size: int) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    with open(mask_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # YOLOv8 segmentation: cls x1 y1 x2 y2 ... (normalized [0,1])
            coords = [float(v) for v in parts[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            xy = np.array(coords, dtype=np.float32).reshape(-1, 2)
            xy *= float(patch_size)
            polygons.append(xy)
    return polygons


def compute_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    min_x = float(np.min(poly[:, 0]))
    min_y = float(np.min(poly[:, 1]))
    max_x = float(np.max(poly[:, 0]))
    max_y = float(np.max(poly[:, 1]))
    return (min_x, min_y, max_x, max_y)


def is_edge_polygon(local_poly: np.ndarray, patch_size: int, edge_threshold: int) -> bool:
    if local_poly.size == 0:
        return False
    min_x = float(np.min(local_poly[:, 0]))
    min_y = float(np.min(local_poly[:, 1]))
    max_x = float(np.max(local_poly[:, 0]))
    max_y = float(np.max(local_poly[:, 1]))
    margin = float(edge_threshold)
    # True if any part of polygon lies within margin of patch boundary
    if min_x <= margin:
        return True
    if min_y <= margin:
        return True
    if max_x >= (patch_size - margin):
        return True
    if max_y >= (patch_size - margin):
        return True
    return False


def pack_contours(polygons: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(p) for p in polygons], dtype=np.int32)
    if lengths.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.array([0], dtype=np.int32)
    points = np.vstack([p.astype(np.float32) for p in polygons])
    offsets = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(lengths, dtype=np.int32)])
    return points, offsets


def make_contours_storage(polygons: List[np.ndarray]):
    if not polygons:
        return {'points': np.zeros((0, 2), dtype=np.float32), 'offsets': np.array([0], dtype=np.int32)}
    lengths = [p.shape[0] for p in polygons]
    all_equal = len(set(lengths)) == 1
    if all_equal:
        arr = np.stack([p.astype(np.float32, copy=False) for p in polygons], axis=0)
        return arr
    pts, offs = pack_contours(polygons)
    return {'points': pts, 'offsets': offs}


def _polygon_area(poly: np.ndarray) -> float:
    """Signed area via shoelace; returns absolute area in px^2."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0].astype(np.float64)
    y = poly[:, 1].astype(np.float64)
    # Close the polygon by rolling
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _polygon_perimeter(poly: np.ndarray) -> float:
    """Perimeter length in pixels."""
    if poly is None or len(poly) < 2:
        return 0.0
    pts = poly.astype(np.float64)
    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.sum(seg_lengths))


def _polygon_major_axis_length(poly: np.ndarray) -> float:
    """Approximate major axis length as the maximum pairwise vertex distance (px)."""
    if poly is None or len(poly) < 2:
        return 0.0
    pts = poly.astype(np.float64)
    # Vectorized pairwise distances (O(n^2)) – polygons typically small
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    max_d = np.sqrt(float(np.max(d2)))
    return max_d


def _polygon_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0 or area <= 0.0:
        return 0.0
    circ = (4.0 * np.pi * area) / (perimeter * perimeter)
    # Numerical stability; bound to [0, 1]
    circ = float(np.clip(circ, 0.0, 1.0))
    return circ


def compute_morphology_features(polygons: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute per-object morphology features in pixel space from boundary polygons.

    Returns:
        features: (N, 4) array [area_px2, perimeter_px, circularity, major_axis_length_px]
        columns: list of column names in the same order
    """
    n = len(polygons)
    out = np.zeros((n, 4), dtype=np.float32)
    for i, poly in enumerate(polygons):
        area = _polygon_area(poly)
        perim = _polygon_perimeter(poly)
        maj = _polygon_major_axis_length(poly)
        circ = _polygon_circularity(area, perim)
        out[i, 0] = area
        out[i, 1] = perim
        out[i, 2] = circ
        out[i, 3] = maj
    cols = ['area_px2', 'perimeter_px', 'circularity', 'major_axis_length_px']
    return out, cols


def assign_nuclei_to_myotubes(nuclei_list: List[Dict], myotube_list: List[Dict], overlap_threshold: float = 0.8, verbose: bool = False, progress_every: int = 50) -> Dict[int, int]:
    """
    Assigns nuclei to myotubes based on intersection over nucleus area.
    Returns: Dict[nucleus_index, myotube_index]
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("shapely is required for nuclei-myotube assignment. Please install it with: pip install shapely")

    start_all = time.perf_counter()
    if verbose:
        print(f"[assign] Begin nuclei→myotube (N={len(nuclei_list)}, M={len(myotube_list)}, thr={overlap_threshold})", flush=True)

    assignment = {}
    
    # Create shapely objects for myotubes once
    myo_polys = []
    t0 = time.perf_counter()
    for m in myotube_list:
        if len(m['polygon']) >= 3:
            myo_polys.append(Polygon(m['polygon']))
        else:
            myo_polys.append(None)
    if verbose:
        print(f"[assign] Built {sum(p is not None for p in myo_polys)}/{len(myo_polys)} myotube polygons in {time.perf_counter()-t0:.2f}s", flush=True)
            
    last_log = time.perf_counter()
    for n_idx, n_obj in enumerate(nuclei_list):
        n_poly_np = n_obj['polygon']
        if len(n_poly_np) < 3:
            continue
            
        n_poly = Polygon(n_poly_np)
        if not n_poly.is_valid:
            try:
                n_poly = n_poly.buffer(0)
            except Exception:
                continue
            
        n_area = n_poly.area
        if n_area <= 0:
            continue
            
        n_bbox = n_obj['bbox'] # (min_x, min_y, max_x, max_y)
        
        best_overlap = 0.0
        best_m_idx = -1
        
        for m_idx, m_obj in enumerate(myotube_list):
            m_poly = myo_polys[m_idx]
            if m_poly is None:
                continue
                
            # BBox check
            m_bbox = m_obj['bbox']
            # min_x, min_y, max_x, max_y
            if (n_bbox[0] > m_bbox[2] or n_bbox[2] < m_bbox[0] or 
                n_bbox[1] > m_bbox[3] or n_bbox[3] < m_bbox[1]):
                continue
                
            # Intersection
            try:
                intersection_area = n_poly.intersection(m_poly).area
                overlap_frac = intersection_area / n_area
                
                if overlap_frac > overlap_threshold and overlap_frac > best_overlap:
                    best_overlap = overlap_frac
                    best_m_idx = m_idx
            except Exception:
                continue
                
        if best_m_idx != -1:
            assignment[n_idx] = best_m_idx

        if verbose and n_idx > 0 and (n_idx % progress_every == 0):
            now = time.perf_counter()
            if (now - last_log) > 0.5:
                print(f"[assign] Progress nuclei {n_idx}/{len(nuclei_list)} (assigned so far={len(assignment)}) elapsed={now-start_all:.1f}s", flush=True)
                last_log = now
            
    if verbose:
        print(f"[assign] Done. Assigned {len(assignment)}/{len(nuclei_list)} in {time.perf_counter()-start_all:.2f}s", flush=True)
    return assignment

