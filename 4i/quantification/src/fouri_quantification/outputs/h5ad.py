from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from ..features.registry import MORPHOLOGY_COLUMNS, feature_group, describe

_WRITE_LOCK = threading.RLock()


def serializable(value):
    """Use standard AnnData encodings only; no pickles or opaque Python objects."""
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.index = frame.index.astype(str)
        frame.index.name = frame.index.name or "row_id"
        for col in frame.columns:
            series = frame[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                frame[col] = series.astype(object).fillna("").astype(str).astype(object)
            elif pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
                frame[col] = series.map(lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)).astype(object)
            elif pd.api.types.is_extension_array_dtype(series.dtype):
                frame[col] = series.astype(float if series.isna().any() else series.dtype.numpy_dtype)
        return frame
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return np.array([], dtype=str)
        if any(isinstance(v, (dict, list, tuple)) for v in value):
            return json.dumps(value, default=str)
        return np.asarray(value)
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write(adata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        # Dense attribute storage supports long DataFrame column-order attributes.
        # Standard AnnData encoding is retained; ordinary read_h5ad still works.
        # Construct locally, then stream one file to network storage before rename.
        with tempfile.TemporaryDirectory(prefix="myonis-4i-", dir=os.environ.get("SLURM_TMPDIR")) as scratch:
            local = Path(scratch)/"data.h5ad"
            with _WRITE_LOCK:
                previous = h5py.get_config().track_order
                h5py.get_config().track_order = True
                try:
                    adata.write_h5ad(local, compression="gzip", convert_strings_to_categoricals=False)
                finally:
                    h5py.get_config().track_order = previous
            shutil.copyfile(local, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_partition(payload, path):
    container = ad.AnnData(np.empty((0, 0), dtype=np.float32))
    container.uns["payload"] = serializable(payload)
    atomic_write(container, path)


def read_partition(path):
    return ad.read_h5ad(path).uns["payload"]


def make_adata(nuclei, myotubes, structures, markers, extra):
    nuclei = nuclei.copy()
    for column in MORPHOLOGY_COLUMNS:
        if column not in nuclei:
            nuclei[column] = np.nan
    if not nuclei.index.is_unique:
        raise ValueError("Nuclear object UIDs are not unique")
    feature_cols = [c for c in nuclei if feature_group(c)]
    obs = serializable(nuclei.drop(columns=feature_cols))
    var = pd.DataFrame([dict(acquisition_id=m["acquisition_id"], marker=m["key"],
                             measurement=int(m["acquisition_id"].split("_")[1]),
                             texture_min=m["texture_range"][0], texture_max=m["texture_range"][1]) for m in markers])
    if var.empty:
        var = pd.DataFrame(columns=["acquisition_id", "marker", "measurement", "texture_min", "texture_max"])
    var = var.set_index("acquisition_id", drop=False)
    matrix = np.full((len(nuclei), len(markers)), np.nan, dtype=np.float32)
    for j, marker in enumerate(markers):
        column = f"{marker['key']}__whole_nucleus__mean"
        if column in nuclei:
            matrix[:, j] = nuclei[column].to_numpy(dtype=np.float32)
    run = ad.AnnData(X=matrix, obs=obs, var=serializable(var))
    dictionary = []
    for group in ["morphology_features", "additional_morphology_features", "intensity_features", "texture_features", "structure_features"]:
        columns = [c for c in feature_cols if feature_group(c) == group]
        if group == "morphology_features":
            columns = [c for c in MORPHOLOGY_COLUMNS if c in nuclei]
            run.obsm[group] = nuclei[columns].to_numpy(dtype=np.float32)
            run.uns["morphology_feature_columns"] = np.array(columns, dtype=str)
        else:
            run.obsm[group] = nuclei[columns].astype(float)
        for column in columns:
            entry = describe(column, "nucleus", f'obsm["{group}"]')
            if entry: dictionary.append(entry)
    for column in obs:
        entry = describe(column, "nucleus", "obs")
        if entry: dictionary.append(entry)
    for scope, frame, location in [("myotube", myotubes, 'uns["myotubes"]')] + [("structure", f, f'uns["structures"]["{k}"]') for k, f in structures.items()]:
        for column in frame:
            entry = describe(column, scope, location)
            if entry: dictionary.append(entry)
    run.uns["myotubes"] = serializable(myotubes)
    run.uns["structures"] = serializable(structures)
    run.uns["feature_dictionary"] = serializable(pd.DataFrame(dictionary))
    run.uns.update(serializable(extra))
    run.uns["schema_version"] = "1.0"
    run.uns["X_definition"] = "Whole-nucleus mean intensity per acquisition under uns['measurement']['overlap_policy']; no count normalization or log transform. NaN means missing/undefined."
    return run
