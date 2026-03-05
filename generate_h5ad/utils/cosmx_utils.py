from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import re
import numpy as np
import pandas as pd
import time

def extract_fov_token_from_target_csv(path: Path) -> str:
    m = re.search(r'FOV(\d+)', path.name)
    if not m:
        raise ValueError(f"Could not parse FOV from filename: {path}")
    digits = m.group(1)
    return f"F{digits}"

def extract_fov_token_from_name(name: str) -> str:
    m = re.search(r'FOV(\d+)', name)
    if m:
        return f"F{m.group(1)}"
    # Allow underscores or other word chars around F00001
    m = re.search(r'F(\d+)', name)
    if m:
        return f"F{m.group(1)}"
    raise ValueError(f"Could not parse FOV from string: {name}")

def parse_fov_index(fov_str: str) -> int:
    m = re.search(r'FOV(\d+)', fov_str)
    if not m:
        m = re.search(r'\bF(\d+)\b', fov_str)
    if not m:
        raise ValueError(f"Could not parse FOV index from: {fov_str}")
    return int(m.group(1))

def extract_patch_index_from_mask(path: Path) -> int:
    m = re.search(r'patch_(\d+)', path.name)
    if not m:
        raise ValueError(f"Could not parse patch index from filename: {path}")
    return int(m.group(1))

def mask_belongs_to_fov(path: Path, fov_token: str) -> bool:
    return f"_{fov_token}_" in path.name

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

def translate_polygons(polygons: List[np.ndarray], dx: float, dy: float) -> List[np.ndarray]:
    if dx == 0 and dy == 0:
        return polygons
    offset = np.array([dx, dy], dtype=np.float32)
    return [poly + offset for poly in polygons]

def compute_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    min_x = float(np.min(poly[:, 0]))
    min_y = float(np.min(poly[:, 1]))
    max_x = float(np.max(poly[:, 0]))
    max_y = float(np.max(poly[:, 1]))
    return (min_x, min_y, max_x, max_y)

def load_polygons_for_fov(mask_paths: List[Path], fov_token: str, patch_size: int, grid_cols: int) -> List[Dict]:
    objects: List[Dict] = []
    for p in mask_paths:
        if not mask_belongs_to_fov(p, fov_token):
            continue
        patch_idx = extract_patch_index_from_mask(p)
        col = patch_idx % grid_cols
        row = patch_idx // grid_cols
        dx = float(col * patch_size)
        dy = float(row * patch_size)
        local_polys = read_yolov8_segmentation_polygons(p, patch_size)
        global_polys = translate_polygons(local_polys, dx, dy)
        for local_poly, global_poly in zip(local_polys, global_polys):
            bbox = compute_bbox(global_poly)
            objects.append({
                'polygon': global_poly,
                'local_polygon': local_poly,
                'offset': (dx, dy),
                'bbox': bbox,
                'patch_idx': patch_idx,
                'image_name': p.name.replace('.txt', ''),
            })
    return objects

def load_polygons_from_paths(mask_paths: List[Path], patch_size: int, grid_cols: int) -> List[Dict]:
    """Load polygons directly from the provided mask paths (no FOV filtering).
    Assumes YOLOv8 segmentation in patch-local coords, denormalized by patch_size.
    """
    objects: List[Dict] = []
    for p in mask_paths:
        try:
            patch_idx = extract_patch_index_from_mask(p)
        except Exception:
            continue
        col = patch_idx % grid_cols
        row = patch_idx // grid_cols
        dx = float(col * patch_size)
        dy = float(row * patch_size)
        local_polys = read_yolov8_segmentation_polygons(p, patch_size)
        if not local_polys:
            continue
        global_polys = translate_polygons(local_polys, dx, dy)
        for local_poly, global_poly in zip(local_polys, global_polys):
            bbox = compute_bbox(global_poly)
            objects.append({
                'polygon': global_poly,
                'local_polygon': local_poly,
                'offset': (dx, dy),
                'bbox': bbox,
                'patch_idx': patch_idx,
                'image_name': p.name.replace('.txt', ''),
            })
    return objects

def group_mask_paths_by_fov(mask_paths: List[Path]) -> Dict[str, List[Path]]:
    """Group segmentation mask txt paths by FOV token (e.g., 'F00001')."""
    mapping: Dict[str, List[Path]] = {}
    for p in mask_paths:
        try:
            token = extract_fov_token_from_name(p.name)
        except Exception:
            continue
        mapping.setdefault(token, []).append(p)
    return mapping

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

def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    # Ray casting algorithm
    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside

def assign_counts(objects: List[Dict], genes: List[str], target_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    gene_to_col = {g: i for i, g in enumerate(genes)}
    genes_out: List[str] = list(genes)
    num_objects = len(objects)
    counts = np.zeros((num_objects, len(genes_out)), dtype=np.int32)

    # Ensure required columns
    if not {'target', 'x', 'y'}.issubset(set(target_df.columns)):
        missing = {'target', 'x', 'y'} - set(target_df.columns)
        raise ValueError(f"Missing required columns in target coords CSV: {missing}")

    # Iterate through spots and assign to the first containing object
    for _, row in target_df.iterrows():
        gene = str(row['target'])
        if gene not in gene_to_col:
            # add new gene column
            gene_to_col[gene] = len(genes_out)
            genes_out.append(gene)
            counts = np.pad(counts, ((0, 0), (0, 1)), mode='constant')
        gx = float(row['x'])
        gy = float(row['y'])
        col_idx = gene_to_col[gene]

        for obj_idx, obj in enumerate(objects):
            min_x, min_y, max_x, max_y = obj['bbox']
            if gx < min_x or gx > max_x or gy < min_y or gy > max_y:
                continue
            if point_in_polygon(gx, gy, obj['polygon']):
                counts[obj_idx, col_idx] += 1
                break  # assign to first match

    return counts, genes_out

def assign_counts_fast(objects: List[Dict], genes: List[str], target_df: pd.DataFrame, patch_size: int, grid_cols: int, assignment_rule: str = 'greedy', random_seed: Optional[int] = 0) -> Tuple[np.ndarray, List[str]]:
    """
    Faster assignment by:
    - grouping objects by patch_idx (computed during loading)
    - vectorized AABB pre-filter per point
    - polygon test only on candidates
    assignment_rule:
    - greedy: assign to first matching object (legacy behavior)
    - shared: assign to all matching objects
    - null: assign only if exactly one matching object
    - random: assign overlaps to a single object using balanced random sharing
    """
    num_objects = len(objects)
    gene_to_col = {g: i for i, g in enumerate(genes)}
    genes_out: List[str] = list(genes)
    counts = np.zeros((num_objects, len(genes_out)), dtype=np.int32)

    if num_objects == 0 or target_df.shape[0] == 0:
        return counts, genes_out

    # Arrays for objects
    bboxes = np.array([obj['bbox'] for obj in objects], dtype=np.float32)  # (N,4)
    patch_indices = np.array([obj['patch_idx'] for obj in objects], dtype=np.int32)
    polys = [obj['polygon'] for obj in objects]  # global coords

    # Patch mapping
    patch_to_obj: Dict[int, np.ndarray] = {}
    for p in np.unique(patch_indices):
        patch_to_obj[int(p)] = np.where(patch_indices == p)[0]

    max_row = int(np.max(patch_indices // max(1, grid_cols))) if num_objects > 0 else 0
    max_col = int(np.max(patch_indices % max(1, grid_cols))) if num_objects > 0 else 0

    # Points from target_df
    if not {'target', 'x', 'y'}.issubset(target_df.columns):
        missing = {'target', 'x', 'y'} - set(target_df.columns)
        raise ValueError(f"Missing required columns in target coords CSV: {missing}")

    gx = target_df['x'].to_numpy(dtype=np.float32)
    gy = target_df['y'].to_numpy(dtype=np.float32)
    gnames = target_df['target'].astype(str).to_numpy()

    # Compute patch index per point (clamped to existing grid extents)
    cols = (gx // float(patch_size)).astype(np.int32)
    rows = (gy // float(patch_size)).astype(np.int32)
    cols = np.clip(cols, 0, max_col)
    rows = np.clip(rows, 0, max_row)
    point_patches = rows * int(grid_cols) + cols

    if assignment_rule not in {'greedy', 'shared', 'null', 'random'}:
        raise ValueError(
            f"Unknown assignment_rule '{assignment_rule}'. "
            "Expected one of: ['greedy', 'shared', 'null', 'random']"
        )
    rng = np.random.default_rng(random_seed) if assignment_rule == 'random' else None

    # Process per patch
    unique_patches = np.unique(point_patches)
    for p in unique_patches:
        pt_idx = np.where(point_patches == p)[0]
        obj_idx = patch_to_obj.get(int(p))
        if obj_idx is None or obj_idx.size == 0:
            continue
        boxes = bboxes[obj_idx]
        overlap_events = {} if assignment_rule == 'random' else None

        for k in pt_idx:
            gene = gnames[k]
            col_idx = gene_to_col.get(gene)
            if col_idx is None:
                col_idx = len(genes_out)
                gene_to_col[gene] = col_idx
                genes_out.append(gene)
                # expand counts matrix by 1 column
                counts = np.pad(counts, ((0, 0), (0, 1)), mode='constant')

            x = gx[k]
            y = gy[k]
            # vectorized AABB filter over this patch's objects
            in_box = (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])
            if not np.any(in_box):
                continue
            cand_obj_idx = obj_idx[in_box]
            if assignment_rule == 'greedy':
                # Legacy fast path: first-hit assignment.
                for oi in cand_obj_idx:
                    if point_in_polygon(float(x), float(y), polys[int(oi)]):
                        counts[int(oi), int(col_idx)] += 1
                        break
            elif assignment_rule == 'shared':
                for oi in cand_obj_idx:
                    if point_in_polygon(float(x), float(y), polys[int(oi)]):
                        counts[int(oi), int(col_idx)] += 1
            elif assignment_rule == 'null':
                hit_idx = -1
                hit_count = 0
                for oi in cand_obj_idx:
                    if point_in_polygon(float(x), float(y), polys[int(oi)]):
                        hit_count += 1
                        if hit_count == 1:
                            hit_idx = int(oi)
                        else:
                            # More than one match means overlap; do not count.
                            break
                if hit_count == 1:
                    counts[hit_idx, int(col_idx)] += 1
            elif assignment_rule == 'random':
                matched = []
                for oi in cand_obj_idx:
                    if point_in_polygon(float(x), float(y), polys[int(oi)]):
                        matched.append(int(oi))
                if not matched:
                    continue
                if len(matched) == 1:
                    counts[matched[0], int(col_idx)] += 1
                    continue
                key = tuple(matched)
                if key not in overlap_events:
                    overlap_events[key] = []
                overlap_events[key].append(int(col_idx))

        if assignment_rule == 'random' and overlap_events:
            for key in sorted(overlap_events.keys()):
                gene_cols = overlap_events[key]
                n_events = len(gene_cols)
                n_objs = len(key)
                reps = (n_events + n_objs - 1) // n_objs
                assigned_objs = np.tile(np.asarray(key, dtype=np.int32), reps)[:n_events]
                rng.shuffle(assigned_objs)
                for col_idx, oi in zip(gene_cols, assigned_objs):
                    counts[int(oi), int(col_idx)] += 1

    return counts, genes_out

def load_gene_list(genes_csv: Path) -> List[str]:
    df = pd.read_csv(genes_csv)
    if 'gene' in df.columns:
        genes = df['gene'].astype(str).tolist()
    elif 'target' in df.columns:
        genes = df['target'].astype(str).tolist()
    else:
        # take first column
        genes = df.iloc[:, 0].astype(str).tolist()
    # Preserve order, drop duplicates
    seen = set()
    ordered: List[str] = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered

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

def load_field_to_cell_line(field_ids_csv: Path, slide_name: Optional[str] = None) -> Dict[str, str]:
    df = pd.read_csv(field_ids_csv)
    mapping: Dict[str, str] = {}
    if slide_name is not None:
        if slide_name not in df.columns:
            raise ValueError(f"slide_name '{slide_name}' not found in columns: {list(df.columns)}")
        col = slide_name
        for idx, row in df.iterrows():
            fov_idx = idx + 1
            v = row[col]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            mapping[f"F{fov_idx:05d}"] = s
            mapping[f"FOV{fov_idx:05d}"] = s
            mapping[f"F{fov_idx:04d}"] = s
            mapping[f"FOV{fov_idx:04d}"] = s
    else:
        for idx, row in df.iterrows():
            fov_idx = idx + 1
            value = None
            for col in df.columns:
                v = row[col]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    value = s
                    break
            if value is not None:
                key = f"F{fov_idx:05d}"
                mapping[key] = value
                mapping[f"FOV{fov_idx:05d}"] = value
                mapping[f"F{fov_idx:04d}"] = value
                mapping[f"FOV{fov_idx:04d}"] = value
    return mapping

def make_cell_line_resolver(field_ids_csv: Path, slide_name: Optional[str] = None) -> Callable[[str], str]:
    df = pd.read_csv(field_ids_csv)
    columns = list(df.columns)
    use_col = None
    if slide_name is not None:
        if slide_name not in columns:
            raise ValueError(f"slide_name '{slide_name}' not found in columns: {columns}")
        use_col = slide_name
    def resolve(fov_token: str) -> str:
        try:
            idx = parse_fov_index(fov_token)
        except Exception:
            # try raw digits in the token
            m = re.search(r'(\d+)', fov_token)
            idx = int(m.group(1)) if m else -1
        if idx < 0 or idx >= len(df):
            return ''
        if use_col is not None:
            v = df.iloc[idx][use_col]
            return '' if pd.isna(v) else str(v).strip()
        # fall back: first non-empty across columns
        row = df.iloc[idx]
        for col in columns:
            v = row[col]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                return s
        return ''
    return resolve

def print_adata_summary(name: str, adata: Any) -> None:
    print(f"Summary for {name}:")
    try:
        n_obs = getattr(adata, 'n_obs', None)
        n_vars = getattr(adata, 'n_vars', None)
        print(f"- cells: {n_obs}, genes: {n_vars}")
        obs_cols = list(getattr(adata, 'obs', pd.DataFrame()).columns)
        print(f"- obs columns: {obs_cols}")
        try:
            print("- first 5 obs rows:")
            print(getattr(adata, 'obs', pd.DataFrame()).head(5))
        except Exception:
            pass
        uns_items = []
        for k in getattr(adata, 'uns_keys', lambda: [])():
            v = adata.uns[k]
            vtype = type(v).__name__
            if k == 'Object Contours' and isinstance(v, dict):
                sub = []
                if 'Contours' in v:
                    c = v['Contours']
                    if isinstance(c, dict):
                        pts = c.get('points')
                        offs = c.get('offsets')
                        sub.append(f"Contours=packed(points={getattr(pts,'shape',None)},{getattr(getattr(pts,'dtype',None),'name',None)}; offsets={getattr(offs,'shape',None)},{getattr(getattr(offs,'dtype',None),'name',None)})")
                    elif isinstance(c, np.ndarray):
                        sub.append(f"Contours=array(shape={c.shape}, dtype={getattr(getattr(c,'dtype',None),'name',None)})")
                if 'Contour offsets' in v:
                    o = v['Contour offsets']
                    sub.append(f"Contour offsets=array(shape={getattr(o,'shape',None)}, dtype={getattr(getattr(o,'dtype',None),'name',None)})")
                vtype = 'dict(' + '; '.join(sub) + ')'
            uns_items.append(f"{k}: {vtype}")
        print(f"- uns: {{ {', '.join(uns_items)} }}")
    except Exception as e:
        print(f"- summary unavailable: {e}")

# ---
# Morphology feature helpers (pixel space)
# ---

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

def assign_counts_raster(objects: List[Dict], genes: List[str], target_df: pd.DataFrame, patch_size: int, grid_cols: int, verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Fast transcript assignment by patch rasterization.
    Requirements: scikit-image
    Strategy:
      - For each patch that has objects and points, build a label image (H=W=patch_size)
      - Fill each object's local polygon with its label (obj_idx+1)
      - For each point in the patch, look up label at integer pixel; assign counts if label>0
    """
    try:
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        raise ImportError("scikit-image is required for rasterized assignment. Please install it with: pip install scikit-image")

    t_start = time.perf_counter()

    num_objects = len(objects)
    gene_to_col = {g: i for i, g in enumerate(genes)}
    genes_out: List[str] = list(genes)
    counts = np.zeros((num_objects, len(genes_out)), dtype=np.int32)

    if num_objects == 0 or target_df.shape[0] == 0:
        if verbose:
            print(f"[raster] Early exit: objects={num_objects}, points={target_df.shape[0]}", flush=True)
        return counts, genes_out

    # Points from target_df
    if not {'target', 'x', 'y'}.issubset(target_df.columns):
        missing = {'target', 'x', 'y'} - set(target_df.columns)
        raise ValueError(f"Missing required columns in target coords CSV: {missing}")

    gx = target_df['x'].to_numpy(dtype=np.float32)
    gy = target_df['y'].to_numpy(dtype=np.float32)
    gnames = target_df['target'].astype(str).to_numpy()

    # Compute patch index per point
    cols = (gx // float(patch_size)).astype(np.int32)
    rows = (gy // float(patch_size)).astype(np.int32)
    # Clamp to non-negative
    cols = np.maximum(cols, 0)
    rows = np.maximum(rows, 0)
    point_patches = rows * int(grid_cols) + cols

    # Group objects by patch
    obj_patch_indices = np.array([obj['patch_idx'] for obj in objects], dtype=np.int32)
    patch_to_obj: Dict[int, np.ndarray] = {}
    for p in np.unique(obj_patch_indices):
        patch_to_obj[int(p)] = np.where(obj_patch_indices == p)[0]

    unique_point_patches = np.unique(point_patches)

    if verbose:
        print(f"[raster] Unique point patches: {len(unique_point_patches)}, object patches: {len(patch_to_obj)}", flush=True)

    for p in unique_point_patches:
        obj_idx = patch_to_obj.get(int(p))
        if obj_idx is None or obj_idx.size == 0:
            continue
        pt_idx = np.where(point_patches == p)[0]
        if pt_idx.size == 0:
            continue

        # Build label image for this patch
        label_img = np.zeros((patch_size, patch_size), dtype=np.int32)
        for oi in obj_idx:
            poly = objects[int(oi)]['local_polygon']
            if poly is None or len(poly) < 3:
                continue
            # local coords: x->col, y->row
            xs = np.clip(np.round(poly[:, 0]).astype(np.int32), 0, patch_size - 1)
            ys = np.clip(np.round(poly[:, 1]).astype(np.int32), 0, patch_size - 1)
            try:
                rr, cc = sk_polygon(ys, xs, shape=label_img.shape)
                label_img[rr, cc] = int(oi) + 1  # store 1-based object index
            except Exception:
                # If polygon rasterization fails, skip this object
                continue

        # Assign points in this patch via label lookup
        patch_col = int(p % int(grid_cols))
        patch_row = int(p // int(grid_cols))
        dx = float(patch_col * patch_size)
        dy = float(patch_row * patch_size)

        for k in pt_idx:
            gene = gnames[k]
            col_idx = gene_to_col.get(gene)
            if col_idx is None:
                col_idx = len(genes_out)
                gene_to_col[gene] = col_idx
                genes_out.append(gene)
                counts = np.pad(counts, ((0, 0), (0, 1)), mode='constant')

            lx = int(gx[k] - dx)
            ly = int(gy[k] - dy)
            if lx < 0 or ly < 0 or lx >= patch_size or ly >= patch_size:
                continue
            label = label_img[ly, lx]
            if label <= 0:
                continue
            oi = int(label - 1)
            counts[oi, col_idx] += 1

    if verbose:
        print(f"[raster] Done in {time.perf_counter()-t_start:.2f}s (objects={num_objects}, points={target_df.shape[0]})", flush=True)

    return counts, genes_out
