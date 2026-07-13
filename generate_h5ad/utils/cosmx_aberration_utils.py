from pathlib import Path
import time

import numpy as np

from utils.cosmx_utils import extract_patch_index_from_mask


_POLYGON_RASTERIZER = None


def _load_nuclear_aberration_prediction(prediction_path: Path):
    """Load a 0-255 per-patch sigmoid PNG as float32 probabilities in [0, 1]."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "imageio is required to read --nuclear_aberration_path PNG files. "
                "Install imageio or run in the image-processing environment."
            ) from exc

    arr = np.asarray(imageio.imread(prediction_path))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D PNG prediction at {prediction_path}, got shape {arr.shape}")

    integer_encoded = np.issubdtype(arr.dtype, np.integer)
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    if integer_encoded:
        arr = arr / 255.0
    elif arr.size and float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _normalise_prediction_paths_by_model(prediction_paths_by_model):
    if not prediction_paths_by_model:
        return []

    first_item = prediction_paths_by_model[0]
    if isinstance(first_item, (str, Path)):
        return [list(prediction_paths_by_model)]

    return [list(paths or []) for paths in prediction_paths_by_model]


def _group_prediction_paths_by_patch(prediction_paths):
    prediction_paths_by_patch = {}
    duplicates = 0
    skipped = 0

    for raw_path in sorted(prediction_paths or [], key=lambda p: str(p)):
        path = Path(raw_path)
        try:
            patch_idx = int(extract_patch_index_from_mask(path))
        except Exception:
            skipped += 1
            continue

        if patch_idx in prediction_paths_by_patch:
            duplicates += 1
            continue

        prediction_paths_by_patch[patch_idx] = path

    return prediction_paths_by_patch, duplicates, skipped


def _load_nuclear_aberration_predictions_by_patch(prediction_paths_by_model, patch_size: int, fov_token: str, verbose: bool = False):
    """Load one or more model prediction sets and return mean-ensembled maps by patch."""
    model_path_lists = _normalise_prediction_paths_by_model(prediction_paths_by_model)
    predictions_by_patch = {}
    if not model_path_lists:
        return predictions_by_patch, {
            "ensemble_method": "mean",
            "n_models": 0,
            "patch_model_counts": {},
        }

    grouped_by_model = []
    total_duplicates = 0
    total_skipped = 0
    for model_idx, model_paths in enumerate(model_path_lists, start=1):
        grouped, duplicates, skipped = _group_prediction_paths_by_patch(model_paths)
        grouped_by_model.append(grouped)
        total_duplicates += duplicates
        total_skipped += skipped
        if verbose:
            print(
                f"[{fov_token}] nuclear aberration model {model_idx}: "
                f"{len(grouped)} patches (duplicates={duplicates}, skipped={skipped})",
                flush=True,
            )

    patch_indices = sorted(set().union(*(set(grouped.keys()) for grouped in grouped_by_model)))
    missing_model_events = []
    patch_model_counts = {}

    for patch_idx in patch_indices:
        ensemble_sum = None
        model_count = 0
        expected_shape = None
        missing_models = []

        for model_idx, grouped in enumerate(grouped_by_model, start=1):
            prediction_path = grouped.get(patch_idx)
            if prediction_path is None:
                missing_models.append(model_idx)
                continue

            prediction = _load_nuclear_aberration_prediction(prediction_path)
            if expected_shape is None:
                expected_shape = prediction.shape
            elif prediction.shape != expected_shape:
                raise ValueError(
                    f"Nuclear aberration ensemble shape mismatch for {fov_token} patch_idx={patch_idx}: "
                    f"expected {expected_shape}, got {prediction.shape} from {prediction_path}"
                )

            if verbose and prediction.shape != (patch_size, patch_size):
                print(
                    f"[{fov_token}] warning: nuclear aberration patch {prediction_path.name} has shape "
                    f"{prediction.shape}, expected ({patch_size}, {patch_size}); polygon coordinates will be scaled",
                    flush=True,
                )

            if ensemble_sum is None:
                ensemble_sum = prediction.astype(np.float32, copy=True)
            else:
                ensemble_sum += prediction
            model_count += 1

        if model_count > 0:
            predictions_by_patch[patch_idx] = ensemble_sum / float(model_count)
            patch_model_counts[str(int(patch_idx))] = int(model_count)
        if missing_models:
            missing_model_events.append((int(patch_idx), missing_models))

    if missing_model_events:
        preview = ", ".join(
            f"patch {patch_idx}: models {missing_models}"
            for patch_idx, missing_models in missing_model_events[:10]
        )
        if len(missing_model_events) > 10:
            preview += f", ... ({len(missing_model_events)} patches total)"
        print(
            f"Warning: {fov_token} nuclear aberration ensemble has missing model predictions; "
            f"using available maps for {preview}",
            flush=True,
        )

    if verbose:
        print(
            f"[{fov_token}] loaded mean nuclear aberration ensemble for {len(predictions_by_patch)} patches "
            f"(models={len(model_path_lists)}, duplicates={total_duplicates}, skipped={total_skipped})",
            flush=True,
        )

    return predictions_by_patch, {
        "ensemble_method": "mean",
        "n_models": int(len(model_path_lists)),
        "patch_model_counts": patch_model_counts,
    }


def _get_polygon_rasterizer():
    """Return the best available polygon rasterizer without adding a hard import dependency."""
    global _POLYGON_RASTERIZER
    if _POLYGON_RASTERIZER is not None:
        return _POLYGON_RASTERIZER

    try:
        from skimage.draw import polygon as sk_polygon
        _POLYGON_RASTERIZER = ("skimage", sk_polygon)
        return _POLYGON_RASTERIZER
    except ImportError:
        pass

    try:
        import cv2
        _POLYGON_RASTERIZER = ("cv2", cv2)
        return _POLYGON_RASTERIZER
    except ImportError:
        pass

    _POLYGON_RASTERIZER = ("numpy", None)
    return _POLYGON_RASTERIZER


def _points_in_polygon_vectorized(x, y, poly: np.ndarray):
    inside = np.zeros(x.shape, dtype=bool)
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        intersects = ((yi > y) != (yj > y)) & (
            x < ((xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        )
        inside ^= intersects
        j = i
    return inside


def _rasterize_polygon_indices_numpy(poly: np.ndarray, image_shape):
    height, width = image_shape
    if poly is None or len(poly) < 3 or height <= 0 or width <= 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    min_x = max(0, int(np.floor(np.min(poly[:, 0]))))
    max_x = min(width - 1, int(np.ceil(np.max(poly[:, 0]))))
    min_y = max(0, int(np.floor(np.min(poly[:, 1]))))
    max_y = min(height - 1, int(np.ceil(np.max(poly[:, 1]))))
    if max_x < min_x or max_y < min_y:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs + 0.5, ys + 0.5)
    inside = _points_in_polygon_vectorized(xx.ravel(), yy.ravel(), poly.astype(np.float32, copy=False))
    if not np.any(inside):
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    rr_grid, cc_grid = np.meshgrid(
        np.arange(min_y, max_y + 1, dtype=np.intp),
        np.arange(min_x, max_x + 1, dtype=np.intp),
        indexing="ij",
    )
    return rr_grid.ravel()[inside], cc_grid.ravel()[inside]


def _rasterize_polygon_indices(poly: np.ndarray, image_shape):
    """Rasterize one polygon once and return row/column pixels inside it."""
    height, width = image_shape
    if poly is None or len(poly) < 3 or height <= 0 or width <= 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    kind, rasterizer = _get_polygon_rasterizer()
    if kind == "skimage":
        xs = np.clip(np.round(poly[:, 0]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.round(poly[:, 1]).astype(np.int32), 0, height - 1)
        rr, cc = rasterizer(ys, xs, shape=(height, width))
        return rr.astype(np.intp, copy=False), cc.astype(np.intp, copy=False)

    if kind == "cv2":
        mask = np.zeros((height, width), dtype=np.uint8)
        xs = np.clip(np.round(poly[:, 0]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.round(poly[:, 1]).astype(np.int32), 0, height - 1)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        rasterizer.fillPoly(mask, [pts], 1)
        return np.nonzero(mask)

    return _rasterize_polygon_indices_numpy(poly, (height, width))


def _scale_local_polygon_to_prediction(local_poly: np.ndarray, prediction_shape, patch_size: int):
    height, width = prediction_shape
    if patch_size <= 0:
        return local_poly.astype(np.float32, copy=True)
    scaled = local_poly.astype(np.float32, copy=True)
    scaled[:, 0] *= float(width) / float(patch_size)
    scaled[:, 1] *= float(height) / float(patch_size)
    return scaled


def _compute_object_nuclear_aberration_stats(objects, predictions_by_patch, patch_size: int):
    """Compute per-object sigmoid total, mean, and raster area with one rasterization per object."""
    n = len(objects)
    totals = np.full(n, np.nan, dtype=np.float32)
    means = np.full(n, np.nan, dtype=np.float32)
    areas = np.full(n, np.nan, dtype=np.float32)
    missing_patches = set()

    for obj_idx, obj in enumerate(objects):
        patch_idx = int(obj["patch_idx"])
        prediction = predictions_by_patch.get(patch_idx)
        if prediction is None:
            missing_patches.add(patch_idx)
            continue

        local_poly = obj.get("local_polygon")
        if local_poly is None or len(local_poly) < 3:
            totals[obj_idx] = 0.0
            areas[obj_idx] = 0.0
            continue

        scaled_poly = _scale_local_polygon_to_prediction(local_poly, prediction.shape, patch_size)
        rr, cc = _rasterize_polygon_indices(scaled_poly, prediction.shape)
        area = int(rr.size)
        areas[obj_idx] = float(area)
        if area == 0:
            totals[obj_idx] = 0.0
            continue

        total = float(np.sum(prediction[rr, cc], dtype=np.float64))
        totals[obj_idx] = total
        means[obj_idx] = total / float(area)

    return {
        "total": totals,
        "mean": means,
        "area": areas,
        "missing_patches": missing_patches,
    }


def add_nuclear_aberration_features(
    adata_nuclei,
    adata_myotubes,
    nuclei_objs,
    myotube_objs,
    assignment_map,
    nuclear_aberration_paths,
    patch_size: int,
    fov_token: str,
    verbose: bool = False,
):
    """Add optional sigmoid-derived structural aberration features to nuclei and myotube obs."""
    t0 = time.perf_counter()
    predictions_by_patch, provenance = _load_nuclear_aberration_predictions_by_patch(
        nuclear_aberration_paths,
        patch_size,
        fov_token,
        verbose=verbose,
    )
    adata_nuclei.uns["nuclear_aberration_ensemble"] = provenance
    adata_myotubes.uns["nuclear_aberration_ensemble"] = provenance

    if not predictions_by_patch:
        print(
            f"Warning: no nuclear aberration predictions found for {fov_token}; "
            "nuclear_aberration columns will be NaN",
            flush=True,
        )

    nuclei_stats = _compute_object_nuclear_aberration_stats(nuclei_objs, predictions_by_patch, patch_size)
    myotube_stats = _compute_object_nuclear_aberration_stats(myotube_objs, predictions_by_patch, patch_size)

    adata_nuclei.obs["nuclear_aberration_mean_sigmoid"] = nuclei_stats["mean"]
    adata_nuclei.obs["nuclear_aberration_total_sigmoid"] = nuclei_stats["total"]

    n_myotubes = len(myotube_objs)
    assigned_nuclei_total = np.zeros(n_myotubes, dtype=np.float64)
    assigned_nuclei_area = np.zeros(n_myotubes, dtype=np.float64)
    assigned_nuclei_count = np.zeros(n_myotubes, dtype=np.int32)
    assigned_nuclei_valid = np.zeros(n_myotubes, dtype=np.int32)

    for n_idx, m_idx in assignment_map.items():
        if n_idx < 0 or n_idx >= len(nuclei_objs) or m_idx < 0 or m_idx >= n_myotubes:
            continue
        assigned_nuclei_count[m_idx] += 1
        nuc_total = float(nuclei_stats["total"][n_idx])
        nuc_area = float(nuclei_stats["area"][n_idx])
        if np.isfinite(nuc_total) and np.isfinite(nuc_area):
            assigned_nuclei_total[m_idx] += nuc_total
            assigned_nuclei_area[m_idx] += nuc_area
            assigned_nuclei_valid[m_idx] += 1

    assigned_total_col = np.full(n_myotubes, np.nan, dtype=np.float32)
    assigned_total_col[assigned_nuclei_count == 0] = 0.0
    complete_assigned = assigned_nuclei_count == assigned_nuclei_valid
    valid_assigned_total = (assigned_nuclei_count > 0) & complete_assigned
    assigned_total_col[valid_assigned_total] = assigned_nuclei_total[valid_assigned_total].astype(np.float32)

    assigned_per_nuc_area = np.full(n_myotubes, np.nan, dtype=np.float32)
    myotube_per_nuc_area = np.full(n_myotubes, np.nan, dtype=np.float32)
    valid_denominator = valid_assigned_total & (assigned_nuclei_area > 0)
    assigned_per_nuc_area[valid_denominator] = (
        assigned_nuclei_total[valid_denominator] / assigned_nuclei_area[valid_denominator]
    ).astype(np.float32)

    myotube_total = myotube_stats["total"]
    valid_myotube_ratio = valid_denominator & np.isfinite(myotube_total)
    myotube_per_nuc_area[valid_myotube_ratio] = (
        myotube_total[valid_myotube_ratio].astype(np.float64) / assigned_nuclei_area[valid_myotube_ratio]
    ).astype(np.float32)

    adata_myotubes.obs["nuclear_aberration_myotube_total_sigmoid"] = myotube_stats["total"]
    adata_myotubes.obs["nuclear_aberration_myotube_mean_sigmoid"] = myotube_stats["mean"]
    adata_myotubes.obs["nuclear_aberration_assigned_nuclei_total_sigmoid"] = assigned_total_col
    adata_myotubes.obs["nuclear_aberration_assigned_nuclei_total_per_nuclear_area"] = assigned_per_nuc_area
    adata_myotubes.obs["nuclear_aberration_myotube_total_per_nuclear_area"] = myotube_per_nuc_area

    missing_patches = sorted(nuclei_stats["missing_patches"] | myotube_stats["missing_patches"])
    if missing_patches:
        print(
            f"Warning: {fov_token} missing nuclear aberration predictions for patch_idx values "
            f"{missing_patches}; affected feature values are NaN",
            flush=True,
        )

    if verbose:
        print(
            f"[{fov_token}] nuclear aberration features in {time.perf_counter()-t0:.2f}s "
            f"(nuclei={len(nuclei_objs)}, myotubes={len(myotube_objs)})",
            flush=True,
        )
