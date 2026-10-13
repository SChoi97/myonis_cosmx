"""Convenient access to linked measurements without knowledge of storage details."""
from __future__ import annotations

import copy

import anndata as ad
import numpy as np
import pandas as pd


def get_nuclei(run):
    result = run.obs.copy()
    for key, value in run.obsm.items():
        if isinstance(value, pd.DataFrame):
            result = result.join(value)
        elif key == "morphology_features":
            result = result.join(pd.DataFrame(value, index=run.obs_names, columns=run.uns["morphology_feature_columns"]))
    return result


def get_myotubes(run, as_anndata=False):
    frame = run.uns["myotubes"].copy()
    if not as_anndata:
        return frame
    columns = [f"{m}__whole_myotube__mean" for m in run.var.marker]
    matrix = frame.reindex(columns=columns).to_numpy(dtype=np.float32)
    result = ad.AnnData(matrix, obs=frame.copy(), var=run.var.copy())
    result.uns["feature_dictionary"] = run.uns["feature_dictionary"].copy()
    result.uns["schema_version"] = run.uns["schema_version"]
    result.uns["X_definition"] = "Whole-myotube mean intensity per acquisition."
    result.uns["provenance"] = copy.deepcopy(run.uns["provenance"])
    return result


def get_structures(run, marker=None):
    tables = run.uns["structures"]
    if marker is not None:
        return tables[marker].copy()
    return pd.concat(list(tables.values()), axis=0) if tables else pd.DataFrame()


def subset_by_myotubes(run, myotube_uids):
    """Select complete parent families. Original aggregates remain valid for kept parents.

    Arbitrary nuclear subsets cannot retain valid myotube aggregates; use explicit
    tables for those analyses rather than silently carrying stale summaries.
    """
    keep_m = set(myotube_uids)
    if not keep_m.issubset(run.uns["myotubes"].index):
        raise KeyError("Unknown myotube UID")
    keep_n = set(run.obs_names[run.obs.parent_myotube_uid.isin(keep_m)])
    subset = run[run.obs_names.isin(keep_n)].copy()
    subset.uns["myotubes"] = run.uns["myotubes"].loc[run.uns["myotubes"].index.isin(keep_m)].copy()
    subset.uns["structures"] = {key: table[table.parent_myotube_uid.isin(keep_m)].copy() if "parent_myotube_uid" in table else table.copy()
                                for key, table in run.uns["structures"].items()}
    all_ids = keep_n | keep_m | {uid for table in subset.uns["structures"].values() for uid in table.index}
    subset.uns["object_registry"] = run.uns["object_registry"].loc[run.uns["object_registry"].index.isin(all_ids)].copy()
    subset.uns["relationships"] = {key: table[table.child_uid.isin(all_ids)].copy() if "child_uid" in table else table.copy()
                                    for key, table in run.uns["relationships"].items()}
    # Keep geometry explicitly aligned to retained IDs, never stale position indices.
    from .outputs.aggregate import subset_geometry
    subset.uns["geometry"] = subset_geometry(run.uns["geometry"], all_ids)
    if "source_geometry" in run.uns:
        subset.uns["source_geometry"] = subset_geometry(run.uns["source_geometry"], all_ids)
    subset.uns["provenance"]["subset_definition"] = "Complete families for explicitly selected myotube UIDs; aggregate values preserved."
    subset.uns["qc"]["scope"] = "QC describes the original run; this object is a complete-family subset."
    return subset
