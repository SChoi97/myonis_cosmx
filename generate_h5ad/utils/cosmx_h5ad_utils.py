from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from utils.cosmx_utils import (
    compute_morphology_features,
    is_edge_polygon,
    make_contours_storage,
)


def build_myotube_intensity_drop_map(metadata_path: Path, intensity_threshold: float):
    """Build image-stem -> set(object_idx) for myotubes to remove by intensity."""
    meta = pd.read_csv(metadata_path)
    required_cols = {"image_name", "average_intensity"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in --myhc_metadata: {missing}")

    idx_col = None
    for candidate in ("contour_idx", "assignment_idx", "myotube_id"):
        if candidate in meta.columns:
            idx_col = candidate
            break
    if idx_col is None:
        raise ValueError(
            "Could not map metadata rows to contour objects: expected one of "
            "['contour_idx', 'assignment_idx', 'myotube_id'] in --myhc_metadata"
        )

    work = meta.copy()
    work["_image_stem"] = work["image_name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    intensity_vals = pd.to_numeric(work["average_intensity"], errors="coerce")
    idx_vals = pd.to_numeric(work[idx_col], errors="coerce")
    fail_mask = intensity_vals < float(intensity_threshold)
    valid_mask = fail_mask & idx_vals.notna() & work["_image_stem"].notna()

    to_drop = work.loc[valid_mask, ["_image_stem"]].copy()
    to_drop["obj_idx"] = idx_vals.loc[valid_mask].astype(np.int64).to_numpy()

    drop_map = {}
    if not to_drop.empty:
        for image_stem, sub_df in to_drop.groupby("_image_stem", sort=False):
            drop_map[str(image_stem)] = set(int(v) for v in sub_df["obj_idx"].tolist())

    summary = {
        "rows_total": int(work.shape[0]),
        "rows_below_threshold": int(fail_mask.sum()),
        "rows_mapped": int(valid_mask.sum()),
        "index_column": idx_col,
    }
    return drop_map, summary


def filter_myotubes_by_intensity(myotube_objs, drop_map):
    """Remove objects whose (image_name-stem, per-image index) appears in drop_map."""
    if not drop_map:
        return myotube_objs, 0

    kept = []
    removed = 0
    image_to_next_idx = {}
    for obj in myotube_objs:
        image_stem = Path(str(obj.get("image_name", ""))).stem
        local_idx = image_to_next_idx.get(image_stem, 0)
        image_to_next_idx[image_stem] = local_idx + 1

        drop_set = drop_map.get(image_stem)
        if drop_set is not None and local_idx in drop_set:
            removed += 1
            continue
        kept.append(obj)
    return kept, removed


def create_anndata_from_objects(objects, counts, genes_out, fov_token, cell_line, patch_size, edge_threshold, prefix="obj", verbose=False):
    """Create AnnData for a set of segmented objects."""
    t0 = time.perf_counter()
    is_edge = [is_edge_polygon(obj["local_polygon"], patch_size, edge_threshold) for obj in objects]
    image_names = [obj["image_name"] for obj in objects]
    if verbose:
        print(f"[{fov_token}] computed edge flags for {len(objects)} objs in {time.perf_counter()-t0:.2f}s", flush=True)

    obs = pd.DataFrame({
        "object_id": [f"{fov_token}_{prefix}_{i}" for i in range(len(objects))],
        "field": [fov_token] * len(objects),
        "patch_idx": [obj["patch_idx"] for obj in objects],
        "Cell Line": [cell_line] * len(objects),
        "is_edge": is_edge,
        "image_name": image_names,
    })

    var = pd.DataFrame(index=pd.Index(genes_out, name="gene"))
    X = csr_matrix(counts)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    t2 = time.perf_counter()
    local_contours = [obj["local_polygon"] for obj in objects]
    contours_store = make_contours_storage(local_contours)
    offsets_arr = np.asarray([obj["offset"] for obj in objects], dtype=np.float32)

    adata.uns["Object Contours"] = {
        "Contours": contours_store,
        "Contour offsets": offsets_arr,
    }
    if verbose:
        print(f"[{fov_token}] stored contours ({len(local_contours)} objects) in {time.perf_counter()-t2:.2f}s", flush=True)

    t3 = time.perf_counter()
    morph_arr, morph_cols = compute_morphology_features(local_contours)
    adata.obsm["morphology_features"] = morph_arr
    adata.uns["morphology_feature_columns"] = morph_cols
    if verbose:
        print(f"[{fov_token}] computed morphology for {len(local_contours)} objs in {time.perf_counter()-t3:.2f}s", flush=True)

    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)

    return adata
