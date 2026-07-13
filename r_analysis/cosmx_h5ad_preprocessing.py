#!/usr/bin/env python3

"""AnnData counterpart to cosmx_h5ad_preprocessing.R."""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_NUCLEI_COMBINED_H5AD = Path(
    "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/"
    "processed_files/cosmx_slides_combined/r_dataset/r_ready/"
    "greedy_nuclei_combined.rready.h5ad"
)
DEFAULT_MYOTUBE_COMBINED_H5AD = Path(
    "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/"
    "processed_files/cosmx_slides_combined/r_dataset/r_ready/"
    "greedy_myotube_combined.rready.h5ad"
)
DEFAULT_FILTERED_NUCLEI_H5AD = Path(
    "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/"
    "processed_files/cosmx_slides_combined/r_dataset/r_ready/"
    "greedy_filtered_nuclei.rready.h5ad"
)
DEFAULT_METADATA_CSV_PATH = Path(
    "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/"
    "processed_files/cosmx_slides_combined/r_dataset/r_ready/"
    "greedy_classifier_metadata.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/"
    "processed_files/cosmx_slides_combined/r_dataset/rds"
)


LIST_ARGS = (
    "metadata_cell_line_candidates",
    "counts_assay_candidates",
    "morphology_feature_names",
    "morphology_feature_indices",
    "normal_class_values",
    "abnormal_class_values",
    "slide_col_candidates",
    "field_col_candidates",
    "patch_col_candidates",
    "cell_line_col_candidates",
    "local_id_col_candidates",
    "myotube_id_col_candidates",
)


DECOUPLER_EXPORT_SPECS = {
    "myonuclei": [
        {
            "filename": "myonuclei_mlm_zscore.csv",
            "obsm_key": "decoupleR_mlm_zscore",
            "availability_col": "pathwaysmlm_zscore_available",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myonuclei_mlm_pvalue.csv",
            "obsm_key": "decoupleR_mlm_pvalue",
            "availability_col": "pathwaysmlm_pvalue_available",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myonuclei_ulm_zscore.csv",
            "obsm_key": "decoupleR_ulm_zscore",
            "availability_col": "tfsulm_zscore_available",
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
        {
            "filename": "myonuclei_ulm_pvalue.csv",
            "obsm_key": "decoupleR_ulm_pvalue",
            "availability_col": "tfsulm_pvalue_available",
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
    ],
    "myotubes": [
        {
            "filename": "myotubes_mlm_zscore.csv",
            "obsm_key": "decoupleR_mlm_zscore",
            "availability_col": "pathwaysmlm_zscore_available",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myotubes_mlm_pvalue.csv",
            "obsm_key": "decoupleR_mlm_pvalue",
            "availability_col": "pathwaysmlm_pvalue_available",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myotubes_ulm_zscore.csv",
            "obsm_key": "decoupleR_ulm_zscore",
            "availability_col": "tfsulm_zscore_available",
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
        {
            "filename": "myotubes_ulm_pvalue.csv",
            "obsm_key": "decoupleR_ulm_pvalue",
            "availability_col": "tfsulm_pvalue_available",
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
    ],
}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {value}")


def flatten_cli_list(values: object) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]

    out = []
    for value in values:
        parts = [part.strip() for part in str(value).split(",")]
        out.extend(part for part in parts if part)
    return out


def extract_provided_option_names(argv: Iterable[str]) -> Set[str]:
    provided = set()
    for token in argv:
        if token.startswith("--"):
            key = token[2:].split("=", 1)[0].replace("-", "_")
            if key:
                provided.add(key)
    return provided


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AnnData counterpart to cosmx_h5ad_preprocessing.R. "
            "Reads R-ready H5AD files, applies the same preprocessing logic, "
            "and writes H5AD outputs for downstream Scanpy/anndata workflows."
        )
    )

    parser.add_argument("--nuclei_combined_h5ad", type=Path, default=DEFAULT_NUCLEI_COMBINED_H5AD)
    parser.add_argument("--myotube_combined_h5ad", type=Path, default=DEFAULT_MYOTUBE_COMBINED_H5AD)
    parser.add_argument("--filtered_nuclei_h5ad", type=Path, default=DEFAULT_FILTERED_NUCLEI_H5AD)
    parser.add_argument("--metadata_csv_path", type=Path, default=DEFAULT_METADATA_CSV_PATH)
    parser.add_argument("--savepath", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="")

    parser.add_argument("--classification_column", type=str, default="Predicted Class")
    parser.add_argument("--sigmoid_logits_column", type=str, default="Sigmoid Logits")
    parser.add_argument(
        "--metadata_cell_line_candidates",
        nargs="+",
        default=["Cell Line", "Cell.Line", "cell_line", "CellLine"],
    )

    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_myonuclei_filename", type=str, default="processed_myonuclei.h5ad")
    parser.add_argument(
        "--output_myonuclei_nonmyonuclei_filename",
        type=str,
        default="processed_myonuclei_nonmyonuclei.h5ad",
    )
    parser.add_argument("--output_myotube_filename", type=str, default="processed_myotube_filtered.h5ad")
    parser.add_argument(
        "--decoupler_basepath",
        "--decoupleR_basepath",
        dest="decoupler_basepath",
        type=Path,
        default=None,
        help=(
            "Optional folder containing run_decoupleR.R CSV exports. "
            "When provided, the script will try to attach the MLM/ULM z-score "
            "and p-value matrices to the output H5ADs."
        ),
    )

    # Present for CLI compatibility with the R script; unused in Python.
    parser.add_argument("--cache_in_same_dir_as_h5ad", type=parse_bool, default=True)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force_rebuild_cache", type=parse_bool, default=False)

    parser.add_argument(
        "--counts_assay_candidates",
        nargs="+",
        default=["X", "raw_counts", "layer_counts", "raw", "matrix", "data"],
    )

    parser.add_argument("--min_cell_total_nuclei", type=int, default=100)
    parser.add_argument("--min_gene_ncells_nuclei", type=int, default=100)
    parser.add_argument("--min_cell_total_myotube", type=int, default=100)
    parser.add_argument("--min_gene_ncells_myotube", type=int, default=100)
    parser.add_argument("--remove_gene_pattern", type=str, default="SystemControl|Negative")

    parser.add_argument("--morphology_reduced_dim_name", type=str, default="morphology_features")
    parser.add_argument(
        "--morphology_feature_names",
        nargs="+",
        default=["area_px2", "perimeter_px", "major_axis_length_px"],
    )
    parser.add_argument("--morphology_feature_indices", nargs="+", default=["1", "2", "4"])

    parser.add_argument("--is_myonucleus_column", type=str, default="is_myonucleus")
    parser.add_argument("--myotube_id_column", type=str, default="myotube_id")
    parser.add_argument("--myotube_id_unassigned_value", type=int, default=-1)

    parser.add_argument("--normal_class_values", nargs="+", default=["0", "normal"])
    parser.add_argument("--abnormal_class_values", nargs="+", default=["1", "abnormal"])

    parser.add_argument(
        "--slide_col_candidates",
        nargs="+",
        default=["Slide Name", "Slide.Name", "slide_name", "slide"],
    )
    parser.add_argument(
        "--field_col_candidates",
        nargs="+",
        default=["field", "Field", "field_key"],
    )
    parser.add_argument(
        "--patch_col_candidates",
        nargs="+",
        default=["patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"],
    )
    parser.add_argument(
        "--cell_line_col_candidates",
        nargs="+",
        default=["Cell Line", "Cell.Line", "cell_line", "CellLine", "cell_line_key"],
    )
    parser.add_argument(
        "--local_id_col_candidates",
        nargs="+",
        default=["local_id", "local.id", "Local ID", "Local.ID"],
    )
    parser.add_argument(
        "--myotube_id_col_candidates",
        nargs="+",
        default=["myotube_id", "local_id", "local.id"],
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    provided = extract_provided_option_names(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    for name in LIST_ARGS:
        setattr(args, name, flatten_cli_list(getattr(args, name)))

    try:
        args.morphology_feature_indices = [int(x) for x in args.morphology_feature_indices]
    except ValueError as exc:
        parser.error(f"--morphology_feature_indices must be integers: {exc}")

    if any(idx < 1 for idx in args.morphology_feature_indices):
        parser.error("--morphology_feature_indices must be 1-based positive integers")

    if len(args.morphology_feature_names) != len(args.morphology_feature_indices):
        parser.error(
            "--morphology_feature_names and --morphology_feature_indices must have the same length"
        )

    if args.savepath is not None:
        prefix = args.prefix or ""
        derived = {
            "nuclei_combined_h5ad": args.savepath / f"{prefix}nuclei_combined.rready.h5ad",
            "myotube_combined_h5ad": args.savepath / f"{prefix}myotube_combined.rready.h5ad",
            "filtered_nuclei_h5ad": args.savepath / f"{prefix}filtered_nuclei.rready.h5ad",
            "metadata_csv_path": args.savepath / f"{prefix}classifier_metadata.csv",
        }
        for key, value in derived.items():
            if key not in provided:
                setattr(args, key, value)

    return args


def stop_if_missing_file(path: Path, label: str) -> None:
    if path is None:
        raise FileNotFoundError(f"{label} is required.")
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def pick_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        available = ", ".join(map(str, df.columns))
        tried = ", ".join(map(str, candidates))
        raise KeyError(
            f"Missing required column for {label}. Tried: {tried}\nAvailable columns: {available}"
        )
    return None


def add_alias_col(
    df: pd.DataFrame,
    new_name: str,
    candidates: Iterable[str],
    label: Optional[str] = None,
    required: bool = True,
) -> pd.DataFrame:
    src = pick_col(df, candidates, label=label or new_name, required=required)
    if src is not None:
        df[new_name] = df[src].to_numpy(copy=False)
    return df


def as_key_series(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values, copy=False)
    return series.map(lambda x: pd.NA if pd.isna(x) else str(x).strip())


def normalize_label_series(values: Iterable[object]) -> pd.Series:
    return as_key_series(values).map(lambda x: pd.NA if pd.isna(x) else x.lower())


def canonicalize_binary_class(
    values: Iterable[object],
    normal_values: Iterable[object],
    abnormal_values: Iterable[object],
    unclassified_values: Iterable[object] = ("-1", ""),
    label: str = "classification",
) -> pd.Series:
    normal_norm = {x for x in normalize_label_series(list(normal_values)).dropna().tolist()}
    abnormal_norm = {x for x in normalize_label_series(list(abnormal_values)).dropna().tolist()}
    unclassified_norm = {
        x for x in normalize_label_series(list(unclassified_values)).dropna().tolist()
    }

    overlap = sorted(normal_norm & abnormal_norm)
    if overlap:
        raise ValueError(
            "normal_values and abnormal_values overlap after normalization: "
            + ", ".join(overlap)
        )

    raw_series = pd.Series(values, copy=False).map(lambda x: pd.NA if pd.isna(x) else str(x))
    norm_series = normalize_label_series(values)

    is_missing = norm_series.isna()
    is_normal = (~is_missing) & norm_series.isin(normal_norm)
    is_abnormal = (~is_missing) & norm_series.isin(abnormal_norm)
    is_unclassified = is_missing | norm_series.isin(unclassified_norm)

    unexpected = ~(is_normal | is_abnormal | is_unclassified)
    if bool(unexpected.any()):
        unexpected_vals = raw_series[unexpected].copy()
        unexpected_vals = unexpected_vals.fillna("<NA>").replace({"": "<blank>"})
        unexpected_tbl = unexpected_vals.value_counts(sort=True)
        unexpected_msg = ", ".join(f"{idx} ({count})" for idx, count in unexpected_tbl.items())
        raise ValueError(
            f"Unexpected {label} values. Allowed normal aliases: {', '.join(map(str, normal_values))}; "
            f"allowed abnormal aliases: {', '.join(map(str, abnormal_values))}; "
            f"allowed unclassified values: -1, blank, NA. Found: {unexpected_msg}"
        )

    out = pd.Series(np.repeat("unclassified", len(norm_series)), index=norm_series.index, dtype=object)
    out.loc[is_normal] = "0"
    out.loc[is_abnormal] = "1"
    return out


def to_myonucleus_flag(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values, copy=False)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.notna() & (numeric == 1)

    norm = normalize_label_series(series)
    return norm.isin({"1", "true", "t", "yes", "y"})


def to_bool_series(values: Iterable[object], label: str = "values") -> pd.Series:
    series = pd.Series(values, copy=False)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.fillna(0).astype(float) != 0

    norm = normalize_label_series(series)
    true_mask = norm.isin({"1", "true", "t", "yes", "y"})
    false_mask = norm.isna() | norm.isin({"0", "false", "f", "no", "n", ""})
    unexpected = ~(true_mask | false_mask)
    if bool(unexpected.any()):
        unexpected_vals = (
            pd.Series(series[unexpected], copy=False)
            .fillna("<NA>")
            .astype(str)
            .value_counts(sort=True)
        )
        warnings.warn(
            "Unexpected boolean-like values in {label}; treating them as False: {vals}".format(
                label=label,
                vals=", ".join("{} ({})".format(idx, count) for idx, count in unexpected_vals.items()),
            ),
            stacklevel=2,
        )

    out = pd.Series(False, index=series.index, dtype=bool)
    out.loc[true_mask] = True
    return out


def read_decoupler_export_csv(
    csv_path: Path,
    availability_col: str,
    reduction_prefixes: Iterable[str],
) -> Dict[str, object]:
    export_df = pd.read_csv(
        csv_path,
        index_col=0,
        keep_default_na=True,
        na_values=["NaN"],
    )
    export_df.index = export_df.index.map(str)

    if export_df.index.has_duplicates:
        dupes = export_df.index[export_df.index.duplicated()].unique().tolist()[:10]
        raise ValueError(
            "decoupleR export has duplicate row identifiers in {}. Examples: {}".format(
                csv_path, dupes
            )
        )

    if availability_col not in export_df.columns:
        raise KeyError(
            "Expected availability column '{}' in decoupleR export '{}'. Available columns start with: {}".format(
                availability_col,
                csv_path,
                ", ".join(map(str, export_df.columns[:20])),
            )
        )

    columns = list(export_df.columns)
    start_idx = columns.index(availability_col) + 1
    trailing_cols = columns[start_idx:]

    score_cols = []
    for col in trailing_cols:
        col_str = str(col)
        if col_str == "decoupleR_export_row":
            continue
        if any(col_str.startswith(prefix) for prefix in reduction_prefixes):
            continue
        score_cols.append(col_str)

    if not score_cols:
        raise ValueError("No decoupleR score columns were found in '{}'.".format(csv_path))

    scores_df = export_df.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")
    scores_df.index = export_df.index

    availability = to_bool_series(export_df[availability_col], label=availability_col)
    availability.index = export_df.index

    return {
        "scores": scores_df,
        "availability": availability,
        "n_rows": int(export_df.shape[0]),
        "n_features": int(scores_df.shape[1]),
    }


def choose_decoupler_alignment_keys(
    adata: ad.AnnData,
    export_index: pd.Index,
    label: str,
) -> Dict[str, object]:
    export_index = pd.Index(export_index.map(str))

    candidates = []
    if "object_id" in adata.obs.columns:
        candidates.append(("object_id", as_key_series(adata.obs["object_id"])))
    candidates.append(
        (
            "obs_names",
            pd.Series(adata.obs_names.astype(str), index=adata.obs_names, copy=False),
        )
    )
    if "decoupleR_cell_id" in adata.obs.columns:
        candidates.append(("decoupleR_cell_id", as_key_series(adata.obs["decoupleR_cell_id"])))

    best_name = None
    best_values = None
    best_overlap = -1

    for name, values in candidates:
        values = pd.Series(values, index=adata.obs_names, copy=False)
        non_missing = values.notna()
        if not values.loc[non_missing].is_unique:
            continue

        overlap = int(values.loc[non_missing].isin(export_index).sum())
        if overlap > best_overlap:
            best_name = name
            best_values = values
            best_overlap = overlap

    if best_name is None or best_values is None or best_overlap <= 0:
        raise ValueError(
            "Could not align decoupleR export '{}' to AnnData rows. "
            "Tried join keys: object_id, obs_names, decoupleR_cell_id.".format(label)
        )

    if best_overlap < adata.n_obs:
        warnings.warn(
            "decoupleR export '{}' matched {} of {} AnnData rows using '{}'. "
            "Unmatched rows will be filled with NaN/False.".format(
                label, best_overlap, adata.n_obs, best_name
            ),
            stacklevel=2,
        )

    return {
        "alignment_name": best_name,
        "alignment_values": best_values,
        "matched_obs": int(best_overlap),
    }


def align_decoupler_frame_to_obs(frame: pd.DataFrame, alignment_values: pd.Series) -> pd.DataFrame:
    indexer = [None if pd.isna(x) else str(x) for x in alignment_values.tolist()]
    aligned = frame.reindex(indexer)
    aligned.index = alignment_values.index
    return aligned


def align_decoupler_series_to_obs(series: pd.Series, alignment_values: pd.Series) -> pd.Series:
    indexer = [None if pd.isna(x) else str(x) for x in alignment_values.tolist()]
    aligned = series.reindex(indexer)
    aligned.index = alignment_values.index
    return aligned


def attach_decoupler_exports(
    adata: ad.AnnData,
    dataset_key: str,
    decoupler_basepath: Path,
) -> Dict[str, object]:
    specs = DECOUPLER_EXPORT_SPECS[dataset_key]
    missing_files = [spec["filename"] for spec in specs if not (decoupler_basepath / spec["filename"]).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing expected decoupleR exports in {}: {}".format(
                decoupler_basepath, ", ".join(missing_files)
            )
        )

    attached = []
    export_metadata = {}
    alignment_name = None
    alignment_values = None
    matched_obs = None

    for spec in specs:
        csv_path = decoupler_basepath / spec["filename"]
        export = read_decoupler_export_csv(
            csv_path=csv_path,
            availability_col=spec["availability_col"],
            reduction_prefixes=spec["reduction_prefixes"],
        )

        if alignment_values is None:
            alignment_info = choose_decoupler_alignment_keys(
                adata=adata,
                export_index=export["scores"].index,
                label=str(csv_path),
            )
            alignment_name = alignment_info["alignment_name"]
            alignment_values = alignment_info["alignment_values"]
            matched_obs = alignment_info["matched_obs"]

        aligned_scores = align_decoupler_frame_to_obs(export["scores"], alignment_values)
        aligned_availability = (
            align_decoupler_series_to_obs(export["availability"], alignment_values)
            .fillna(False)
            .astype(bool)
        )

        adata.obsm[spec["obsm_key"]] = aligned_scores
        adata.obs[spec["availability_col"]] = aligned_availability.to_numpy()

        attached.append(spec["obsm_key"])
        export_metadata[spec["obsm_key"]] = {
            "source_file": str(csv_path),
            "availability_col": spec["availability_col"],
            "n_rows_in_csv": export["n_rows"],
            "n_features": export["n_features"],
        }

    adata.uns["decoupleR"] = {
        "basepath": str(decoupler_basepath),
        "dataset_key": dataset_key,
        "alignment_key": alignment_name,
        "matched_obs": matched_obs,
        "attached_exports": attached,
        "files": export_metadata,
    }

    return {
        "attached_exports": attached,
        "alignment_key": alignment_name,
        "matched_obs": matched_obs,
    }


def matrix_copy(matrix):
    return matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)


def ensure_counts_layer(adata: ad.AnnData, obj_name: str, candidates: Iterable[str]) -> ad.AnnData:
    if "counts" in adata.layers:
        print(f"[{obj_name}] counts layer already present.")
        return adata

    selected = None
    hit = None
    for candidate in candidates:
        if candidate == "X":
            selected = matrix_copy(adata.X)
            hit = "X"
            break
        if candidate == "counts" and "counts" in adata.layers:
            selected = matrix_copy(adata.layers["counts"])
            hit = "counts"
            break
        if candidate in adata.layers:
            selected = matrix_copy(adata.layers[candidate])
            hit = candidate
            break

    if selected is None:
        available = ["X", *adata.layers.keys()]
        raise ValueError(
            f"[{obj_name}] No counts-like matrix found. Available matrices/layers: "
            + ", ".join(map(str, available))
        )

    print(f"[{obj_name}] Mapping matrix '{hit}' -> 'counts'")
    adata.layers["counts"] = selected
    return adata


def get_counts_matrix(adata: ad.AnnData):
    return adata.layers["counts"] if "counts" in adata.layers else adata.X


def matrix_axis_sum(matrix, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=axis)).ravel()
    return np.asarray(matrix).sum(axis=axis)


def matrix_axis_nonzero_count(matrix, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray((matrix > 0).sum(axis=axis)).ravel()
    return (np.asarray(matrix) > 0).sum(axis=axis)


def add_morphology_obs_columns(
    adata: ad.AnnData,
    reduced_dim_name: str,
    feature_names: List[str],
    feature_indices_1_based: List[int],
) -> None:
    if reduced_dim_name not in adata.obsm:
        warnings.warn(
            f"ReducedDim '{reduced_dim_name}' not found in filtered_nuclei input. "
            "Skipping morphology feature assignment.",
            stacklevel=2,
        )
        return

    morph = adata.obsm[reduced_dim_name]
    n_cols = morph.shape[1]
    max_idx = max(feature_indices_1_based)
    if max_idx > n_cols:
        warnings.warn(
            f"ReducedDim '{reduced_dim_name}' has {n_cols} columns; expected >= {max_idx}. "
            "Skipping morphology column assignment.",
            stacklevel=2,
        )
        return

    zero_based = [idx - 1 for idx in feature_indices_1_based]
    if isinstance(morph, pd.DataFrame):
        for name, idx in zip(feature_names, zero_based):
            adata.obs[name] = morph.iloc[:, idx].to_numpy()
    else:
        morph_arr = np.asarray(morph)
        for name, idx in zip(feature_names, zero_based):
            adata.obs[name] = morph_arr[:, idx]


def filter_adata_by_counts(
    adata: ad.AnnData,
    min_cell_total: int,
    min_gene_ncells: int,
    remove_gene_pattern: str,
) -> ad.AnnData:
    counts = get_counts_matrix(adata)
    cell_totals = matrix_axis_sum(counts, axis=1)
    keep_cells = cell_totals >= min_cell_total
    adata = adata[keep_cells, :].copy()

    counts = get_counts_matrix(adata)
    gene_ncells = matrix_axis_nonzero_count(counts, axis=0)
    keep_genes = gene_ncells >= min_gene_ncells
    adata = adata[:, keep_genes].copy()

    gene_regex = re.compile(remove_gene_pattern, flags=re.IGNORECASE)
    bad_genes = np.fromiter(
        (bool(gene_regex.search(str(gene))) for gene in adata.var_names),
        dtype=bool,
        count=adata.n_vars,
    )
    adata = adata[:, ~bad_genes].copy()
    return adata


def print_config(args: argparse.Namespace) -> None:
    print("========== CONFIG ==========")
    print(f"nuclei_combined_h5ad:   {args.nuclei_combined_h5ad}")
    print(f"myotube_combined_h5ad:  {args.myotube_combined_h5ad}")
    print(f"filtered_nuclei_h5ad:   {args.filtered_nuclei_h5ad}")
    print(f"metadata_csv_path:      {args.metadata_csv_path}")
    print(f"classification_column:  {args.classification_column}")
    print(f"sigmoid_logits_column:  {args.sigmoid_logits_column}")
    print(f"decoupler_basepath:     {args.decoupler_basepath}")
    print(f"output_dir:             {args.output_dir}")
    print(f"output_myonuclei:       {args.output_myonuclei_filename}")
    print(f"output_all_nuclei:      {args.output_myonuclei_nonmyonuclei_filename}")
    print(f"output_myotube:         {args.output_myotube_filename}")
    print(f"force_rebuild_cache:    {args.force_rebuild_cache} (ignored in Python)")
    print(f"min_cell_total_nuclei:  {args.min_cell_total_nuclei}")
    print(f"min_gene_ncells_nuclei: {args.min_gene_ncells_nuclei}")
    print(f"min_cell_total_myotube: {args.min_cell_total_myotube}")
    print(f"min_gene_ncells_myotube:{args.min_gene_ncells_myotube}")
    print("============================\n")


def read_h5ad(path: Path, label: str) -> ad.AnnData:
    print(f"Reading {label}: {path}")
    return ad.read_h5ad(path)


def summarize_metadata(metadata_df: pd.DataFrame, classification_column: str, cell_line_col: str) -> pd.DataFrame:
    summary = (
        metadata_df.assign(
            cell_line=pd.Series(metadata_df[cell_line_col], copy=False).map(
                lambda x: pd.NA if pd.isna(x) else str(x)
            ),
            class_value=pd.Series(metadata_df[classification_column], copy=False).map(
                lambda x: pd.NA if pd.isna(x) else str(x)
            ),
        )
        .groupby(["cell_line", "class_value"], dropna=False, observed=False)
        .size()
        .reset_index(name="n")
    )

    if not summary.empty:
        summary["pct"] = (
            summary.groupby("cell_line", dropna=False, observed=False)["n"]
            .transform(lambda s: np.round(100 * s / s.sum(), 2))
            .astype(float)
        )
        summary = summary.sort_values(["cell_line", "n"], ascending=[True, False], kind="stable")

    return summary


def build_metadata_keep_df(args: argparse.Namespace, metadata_df: pd.DataFrame) -> pd.DataFrame:
    work = metadata_df.copy()
    work = add_alias_col(work, "slide_name", args.slide_col_candidates, "Slide Name")
    work = add_alias_col(work, "field_key", args.field_col_candidates, "field")
    work = add_alias_col(work, "patch_idx_key", args.patch_col_candidates, "patch_idx")
    work = add_alias_col(work, "cell_line_key", args.cell_line_col_candidates, "Cell Line")
    work = add_alias_col(work, "local_id_key", args.local_id_col_candidates, "local_id")
    work = add_alias_col(
        work,
        "classification_key",
        [
            args.classification_column,
            "Classification",
            "classification",
            "Predicted Class",
            "predicted_class",
        ],
        "classification",
    )
    work = add_alias_col(
        work,
        "sigmoid_logits_key",
        [
            args.sigmoid_logits_column,
            "Sigmoid Logits",
            "sigmoid_logits",
            "sigmoid.logits",
        ],
        "sigmoid_logits",
    )

    work["slide_name"] = as_key_series(work["slide_name"]).map(
        lambda x: pd.NA if pd.isna(x) else x.upper()
    )
    work["field_key"] = as_key_series(work["field_key"])
    work["cell_line_key"] = as_key_series(work["cell_line_key"])
    work["patch_idx_key"] = pd.to_numeric(work["patch_idx_key"], errors="coerce")
    work["local_id_key"] = pd.to_numeric(work["local_id_key"], errors="coerce")
    work["classification_key"] = pd.Series(work["classification_key"], copy=False).map(
        lambda x: pd.NA if pd.isna(x) else str(x)
    )
    work["sigmoid_logits_key"] = pd.to_numeric(work["sigmoid_logits_key"], errors="coerce")

    keep_cols = [
        "slide_name",
        "field_key",
        "patch_idx_key",
        "cell_line_key",
        "local_id_key",
        "classification_key",
        "sigmoid_logits_key",
    ]
    work = work.loc[:, keep_cols].rename(
        columns={
            "classification_key": "Classification",
            "sigmoid_logits_key": "Sigmoid_Logits",
        }
    )
    work = work.drop_duplicates(
        subset=["slide_name", "field_key", "patch_idx_key", "cell_line_key", "local_id_key"],
        keep="first",
    )
    return work


def merge_metadata_into_filtered_nuclei(
    filtered_nuclei: ad.AnnData,
    metadata_keep_df: pd.DataFrame,
    args: argparse.Namespace,
) -> ad.AnnData:
    cd_nuc = filtered_nuclei.obs.copy()
    cd_nuc = add_alias_col(cd_nuc, "slide_name", args.slide_col_candidates, "Slide Name")
    cd_nuc = add_alias_col(cd_nuc, "field_key", args.field_col_candidates, "field")
    cd_nuc = add_alias_col(cd_nuc, "patch_idx_key", args.patch_col_candidates, "patch_idx")
    cd_nuc = add_alias_col(cd_nuc, "cell_line_key", args.cell_line_col_candidates, "Cell Line")
    cd_nuc = add_alias_col(cd_nuc, "local_id_key", args.local_id_col_candidates, "local_id")

    cd_nuc["slide_name"] = as_key_series(cd_nuc["slide_name"]).map(
        lambda x: pd.NA if pd.isna(x) else x.upper()
    )
    cd_nuc["field_key"] = as_key_series(cd_nuc["field_key"])
    cd_nuc["cell_line_key"] = as_key_series(cd_nuc["cell_line_key"])
    cd_nuc["patch_idx_key"] = pd.to_numeric(cd_nuc["patch_idx_key"], errors="coerce")
    cd_nuc["local_id_key"] = pd.to_numeric(cd_nuc["local_id_key"], errors="coerce")
    cd_nuc["_row_id"] = filtered_nuclei.obs_names.to_numpy()
    cd_nuc = cd_nuc.drop(columns=["Classification", "Sigmoid_Logits"], errors="ignore")

    merged = cd_nuc.merge(
        metadata_keep_df,
        how="left",
        on=["slide_name", "field_key", "patch_idx_key", "cell_line_key", "local_id_key"],
        sort=False,
    )
    merged = merged.set_index("_row_id").loc[filtered_nuclei.obs_names]
    merged.index.name = None
    filtered_nuclei.obs = merged
    return filtered_nuclei


def build_slide_stats_df(
    filtered_nuclei: ad.AnnData,
    filtered_class_canonical: pd.Series,
) -> pd.DataFrame:
    cd_stats = filtered_nuclei.obs.copy()
    slide_col_stats = pick_col(
        cd_stats,
        ["Slide Name", "Slide.Name", "slide_name", "slide", "slide_name"],
        "Slide Name",
    )
    class_col_stats = pick_col(cd_stats, ["Classification"], "Classification", required=False)

    counts = get_counts_matrix(filtered_nuclei)
    cell_counts_stats = matrix_axis_sum(counts, axis=1)
    cell_unique_stats = matrix_axis_nonzero_count(counts, axis=1)

    cd_stats["slide_key"] = pd.Series(cd_stats[slide_col_stats], copy=False).map(
        lambda x: pd.NA if pd.isna(x) else str(x)
    )
    if class_col_stats is not None:
        cd_stats["class_key"] = filtered_class_canonical.to_numpy(copy=False)
    else:
        cd_stats["class_key"] = pd.NA
    cd_stats["_cell_counts"] = cell_counts_stats
    cd_stats["_cell_unique"] = cell_unique_stats

    slide_stats_df = (
        cd_stats.groupby("slide_key", dropna=False, observed=False)
        .agg(
            **{
                "Avg Count (All)": ("_cell_counts", "mean"),
                "Avg Unique Genes (All)": ("_cell_unique", "mean"),
            }
        )
        .reset_index()
    )

    class_levels = [cls for cls in ("0", "1") if cls in set(cd_stats["class_key"].dropna())]
    for cls in class_levels:
        tmp = (
            cd_stats.loc[cd_stats["class_key"] == cls]
            .groupby("slide_key", dropna=False, observed=False)
            .agg(avg_count=("_cell_counts", "mean"), avg_unique=("_cell_unique", "mean"))
            .reset_index()
        )
        slide_stats_df = slide_stats_df.merge(tmp, on="slide_key", how="left", sort=False)
        slide_stats_df = slide_stats_df.rename(
            columns={
                "avg_count": f"Class_{cls}_Avg_Count",
                "avg_unique": f"Class_{cls}_Avg_Unique",
            }
        )

    numeric_cols = slide_stats_df.select_dtypes(include=[np.number]).columns
    slide_stats_df[numeric_cols] = slide_stats_df[numeric_cols].round(2)
    slide_stats_df = slide_stats_df.rename(columns={"slide_key": "Slide Name"})
    return slide_stats_df


def assign_myonucleus_flag_if_needed(filtered_nuclei: ad.AnnData, args: argparse.Namespace) -> None:
    if args.is_myonucleus_column in filtered_nuclei.obs.columns:
        return

    if args.myotube_id_column not in filtered_nuclei.obs.columns:
        raise KeyError(
            f"Neither '{args.is_myonucleus_column}' nor '{args.myotube_id_column}' "
            "is present in filtered_nuclei obs."
        )

    raw = filtered_nuclei.obs[args.myotube_id_column]
    numeric = pd.to_numeric(raw, errors="coerce")
    if bool(numeric.isna().all()):
        is_myonucleus = normalize_label_series(raw) != str(args.myotube_id_unassigned_value)
    else:
        is_myonucleus = numeric.notna() & (numeric != args.myotube_id_unassigned_value)

    filtered_nuclei.obs[args.is_myonucleus_column] = is_myonucleus.fillna(False).astype(np.int64)


def derive_myotube_nucleus_counts(
    myonuclei: ad.AnnData,
    myotube_combined: ad.AnnData,
    args: argparse.Namespace,
) -> ad.AnnData:
    cd_nuclei_for_tube = myonuclei.obs.copy()
    cd_myotube = myotube_combined.obs.copy()

    slide_n = pick_col(cd_nuclei_for_tube, ["Slide Name", "Slide.Name", "slide_name", "slide"], "Slide Name")
    field_n = pick_col(cd_nuclei_for_tube, ["field", "Field", "field_key"], "field")
    patch_n = pick_col(
        cd_nuclei_for_tube,
        ["patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"],
        "patch_idx",
    )
    tube_n = pick_col(cd_nuclei_for_tube, args.myotube_id_col_candidates, "myotube_id/local_id")
    class_n = pick_col(
        cd_nuclei_for_tube,
        ["Classification", args.classification_column],
        "Classification",
    )

    slide_t = pick_col(cd_myotube, ["Slide Name", "Slide.Name", "slide_name", "slide"], "Slide Name")
    field_t = pick_col(cd_myotube, ["field", "Field", "field_key"], "field")
    patch_t = pick_col(
        cd_myotube,
        ["patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"],
        "patch_idx",
    )
    tube_t = pick_col(cd_myotube, args.myotube_id_col_candidates, "myotube_id/local_id")

    nuc_df = pd.DataFrame(
        {
            "slide_key": as_key_series(cd_nuclei_for_tube[slide_n]).map(
                lambda x: pd.NA if pd.isna(x) else x.upper()
            ),
            "field_key": as_key_series(cd_nuclei_for_tube[field_n]),
            "patch_key": as_key_series(cd_nuclei_for_tube[patch_n]),
            "tube_key": as_key_series(cd_nuclei_for_tube[tube_n]),
            "class_raw": pd.Series(cd_nuclei_for_tube[class_n], copy=False).map(
                lambda x: pd.NA if pd.isna(x) else str(x)
            ),
        },
        index=myonuclei.obs_names,
    )

    nuc_df["class_canonical"] = canonicalize_binary_class(
        nuc_df["class_raw"],
        normal_values=args.normal_class_values,
        abnormal_values=args.abnormal_class_values,
        label="myonuclei Classification",
    )
    nuc_df["class_group"] = pd.Series(pd.NA, index=nuc_df.index, dtype=object)
    nuc_df.loc[nuc_df["class_canonical"] == "0", "class_group"] = "Normal"
    nuc_df.loc[nuc_df["class_canonical"] == "1", "class_group"] = "Abnormal"
    nuc_df = nuc_df.loc[nuc_df["class_group"].notna()].copy()

    if nuc_df.empty:
        counts_by_tube = pd.DataFrame(
            columns=["slide_key", "field_key", "patch_key", "tube_key", "n_normal_nuclei", "n_abnormal_nuclei"]
        )
    else:
        counts_by_tube = (
            nuc_df.groupby(
                ["slide_key", "field_key", "patch_key", "tube_key", "class_group"],
                dropna=False,
                observed=False,
            )
            .size()
            .reset_index(name="n")
            .pivot_table(
                index=["slide_key", "field_key", "patch_key", "tube_key"],
                columns="class_group",
                values="n",
                fill_value=0,
                aggfunc="sum",
                observed=False,
            )
            .reset_index()
        )
        counts_by_tube.columns.name = None

    if "Normal" not in counts_by_tube.columns:
        counts_by_tube["Normal"] = 0
    if "Abnormal" not in counts_by_tube.columns:
        counts_by_tube["Abnormal"] = 0

    counts_by_tube = counts_by_tube.rename(
        columns={"Normal": "n_normal_nuclei", "Abnormal": "n_abnormal_nuclei"}
    )

    tube_df = pd.DataFrame(
        {
            "slide_key": as_key_series(cd_myotube[slide_t]).map(
                lambda x: pd.NA if pd.isna(x) else x.upper()
            ),
            "field_key": as_key_series(cd_myotube[field_t]),
            "patch_key": as_key_series(cd_myotube[patch_t]),
            "tube_key": as_key_series(cd_myotube[tube_t]),
            "_row_id": myotube_combined.obs_names.to_numpy(),
        }
    )

    merge_df = tube_df.merge(
        counts_by_tube,
        how="left",
        on=["slide_key", "field_key", "patch_key", "tube_key"],
        sort=False,
    )
    merge_df = merge_df.set_index("_row_id").loc[myotube_combined.obs_names]

    merge_df["n_normal_nuclei"] = (
        pd.to_numeric(merge_df["n_normal_nuclei"], errors="coerce").fillna(0).astype(np.int64)
    )
    merge_df["n_abnormal_nuclei"] = (
        pd.to_numeric(merge_df["n_abnormal_nuclei"], errors="coerce").fillna(0).astype(np.int64)
    )

    total_nuclei = merge_df["n_normal_nuclei"] + merge_df["n_abnormal_nuclei"]
    merge_df["pct_abnormal_nuclei"] = np.where(
        total_nuclei > 0,
        merge_df["n_abnormal_nuclei"] / total_nuclei,
        np.nan,
    )

    morphology_class = np.full(len(merge_df), -1, dtype=np.int64)
    only_normal = (merge_df["n_normal_nuclei"] > 0) & (merge_df["n_abnormal_nuclei"] == 0)
    only_abnormal = (merge_df["n_normal_nuclei"] == 0) & (merge_df["n_abnormal_nuclei"] > 0)
    mixed = (merge_df["n_normal_nuclei"] > 0) & (merge_df["n_abnormal_nuclei"] > 0)
    morphology_class[only_normal.to_numpy()] = 1
    morphology_class[only_abnormal.to_numpy()] = 2
    morphology_class[mixed.to_numpy()] = 3
    merge_df["morphology_class"] = morphology_class

    myotube_combined.obs["n_normal_nuclei"] = merge_df["n_normal_nuclei"].to_numpy()
    myotube_combined.obs["n_abnormal_nuclei"] = merge_df["n_abnormal_nuclei"].to_numpy()
    myotube_combined.obs["pct_abnormal_nuclei"] = merge_df["pct_abnormal_nuclei"].to_numpy()
    myotube_combined.obs["morphology_class"] = merge_df["morphology_class"].to_numpy()
    return myotube_combined


def print_summary(
    raw_counts: Dict[str, int],
    filtered_nuclei: ad.AnnData,
    myonuclei: ad.AnnData,
    myotube_combined: ad.AnnData,
    myotube_filtered: ad.AnnData,
    classification_assigned: int,
    classification_missing: int,
    sigmoid_logits_assigned: int,
    sigmoid_logits_missing: int,
    metadata_summary: pd.DataFrame,
    slide_stats_df: pd.DataFrame,
    filtered_nuclei_file: Path,
    myonuclei_file: Path,
    myotube_file: Path,
    args: argparse.Namespace,
    decoupler_summary: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    print("\n========== SUMMARY ==========")
    print("Nuclei")
    print(
        f"  Raw nuclei_combined: cells={raw_counts['nuclei_cells']}, "
        f"genes={raw_counts['nuclei_genes']}"
    )
    print(
        f"  Raw filtered_nuclei input: cells={raw_counts['filtered_nuclei_cells']}, "
        f"genes={raw_counts['filtered_nuclei_genes']}"
    )
    print(f"  Final filtered_nuclei (QC): cells={filtered_nuclei.n_obs}, genes={filtered_nuclei.n_vars}")

    print("\nMyotubes")
    print(
        f"  Raw myotube_combined: cells={raw_counts['myotube_cells']}, "
        f"genes={raw_counts['myotube_genes']}"
    )
    print(f"  Final myotube_combined (QC): cells={myotube_combined.n_obs}, genes={myotube_combined.n_vars}")
    print(f"  Kept with >=1 nucleus: {myotube_filtered.n_obs}")

    print("\nMetadata merge")
    print(f"  classification_column used: {args.classification_column}")
    print(f"  sigmoid_logits_column used: {args.sigmoid_logits_column}")
    print(f"  Filtered nuclei with Classification assigned: {classification_assigned}")
    print(f"  Filtered nuclei missing Classification: {classification_missing}")
    print(f"  Filtered nuclei with Sigmoid_Logits assigned: {sigmoid_logits_assigned}")
    print(f"  Filtered nuclei missing Sigmoid_Logits: {sigmoid_logits_missing}")

    print("\nMyonuclei")
    print(f"  Count: {myonuclei.n_obs}")

    print("\nFiltered nuclei Classification counts:")
    class_counts = (
        pd.Series(filtered_nuclei.obs["Classification"], copy=False)
        .map(lambda x: "<NA>" if pd.isna(x) else str(x))
        .value_counts(sort=True)
    )
    print(class_counts.to_string())

    print("\nMyotube nucleus class counts (derived) summary:")
    print(pd.Series(myotube_combined.obs["n_normal_nuclei"]).describe().to_string())
    print(pd.Series(myotube_combined.obs["n_abnormal_nuclei"]).describe().to_string())

    print("\nMyotube morphology_class counts (-1 none, 1 normal-only, 2 abnormal-only, 3 mixed):")
    morph_counts = pd.Series(myotube_combined.obs["morphology_class"]).value_counts(sort=False, dropna=False)
    print(morph_counts.to_string())

    print("\nMetadata class distribution (first 20 rows):")
    if metadata_summary.empty:
        print("<empty>")
    else:
        print(metadata_summary.head(20).to_string(index=False))

    print("\nSlide-level stats:")
    if slide_stats_df.empty:
        print("<empty>")
    else:
        print(slide_stats_df.to_string(index=False))

    if decoupler_summary:
        print("\ndecoupleR attachment:")
        for label in ("myonuclei", "myotubes"):
            if label not in decoupler_summary:
                continue
            info = decoupler_summary[label]
            print(
                "  {}: alignment_key={}, matched_obs={}, exports={}".format(
                    label,
                    info.get("alignment_key"),
                    info.get("matched_obs"),
                    ", ".join(info.get("attached_exports", [])),
                )
            )

    print("\nSaved files:")
    print(f" - {filtered_nuclei_file}")
    print(f" - {myonuclei_file}")
    print(f" - {myotube_file}")
    print("=============================")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    stop_if_missing_file(args.nuclei_combined_h5ad, "nuclei_combined_h5ad")
    stop_if_missing_file(args.myotube_combined_h5ad, "myotube_combined_h5ad")
    stop_if_missing_file(args.filtered_nuclei_h5ad, "filtered_nuclei_h5ad")
    stop_if_missing_file(args.metadata_csv_path, "metadata_csv_path")
    if args.decoupler_basepath is not None:
        stop_if_missing_file(args.decoupler_basepath, "decoupler_basepath")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_config(args)

    nuclei_combined = read_h5ad(args.nuclei_combined_h5ad, "nuclei_combined")
    myotube_combined = read_h5ad(args.myotube_combined_h5ad, "myotube_combined")
    filtered_nuclei = read_h5ad(args.filtered_nuclei_h5ad, "filtered_nuclei")

    raw_counts = {
        "nuclei_cells": int(nuclei_combined.n_obs),
        "nuclei_genes": int(nuclei_combined.n_vars),
        "myotube_cells": int(myotube_combined.n_obs),
        "myotube_genes": int(myotube_combined.n_vars),
        "filtered_nuclei_cells": int(filtered_nuclei.n_obs),
        "filtered_nuclei_genes": int(filtered_nuclei.n_vars),
    }

    nuclei_combined = ensure_counts_layer(nuclei_combined, "nuclei_combined", args.counts_assay_candidates)
    myotube_combined = ensure_counts_layer(myotube_combined, "myotube_combined", args.counts_assay_candidates)
    filtered_nuclei = ensure_counts_layer(filtered_nuclei, "filtered_nuclei", args.counts_assay_candidates)

    assign_myonucleus_flag_if_needed(filtered_nuclei, args)
    add_morphology_obs_columns(
        filtered_nuclei,
        reduced_dim_name=args.morphology_reduced_dim_name,
        feature_names=args.morphology_feature_names,
        feature_indices_1_based=args.morphology_feature_indices,
    )

    filtered_nuclei = filter_adata_by_counts(
        filtered_nuclei,
        min_cell_total=args.min_cell_total_nuclei,
        min_gene_ncells=args.min_gene_ncells_nuclei,
        remove_gene_pattern=args.remove_gene_pattern,
    )
    myotube_combined = filter_adata_by_counts(
        myotube_combined,
        min_cell_total=args.min_cell_total_myotube,
        min_gene_ncells=args.min_gene_ncells_myotube,
        remove_gene_pattern=args.remove_gene_pattern,
    )

    metadata_df = pd.read_csv(args.metadata_csv_path)
    if args.classification_column not in metadata_df.columns:
        raise KeyError(
            f"classification_column='{args.classification_column}' not found in metadata. "
            f"Available: {', '.join(map(str, metadata_df.columns))}"
        )
    if args.sigmoid_logits_column not in metadata_df.columns:
        raise KeyError(
            f"sigmoid_logits_column='{args.sigmoid_logits_column}' not found in metadata. "
            f"Available: {', '.join(map(str, metadata_df.columns))}"
        )

    canonicalize_binary_class(
        metadata_df[args.classification_column],
        normal_values=args.normal_class_values,
        abnormal_values=args.abnormal_class_values,
        label=f"metadata column '{args.classification_column}'",
    )

    cell_line_col_meta = pick_col(
        metadata_df,
        args.metadata_cell_line_candidates,
        "Cell Line",
    )
    metadata_summary = summarize_metadata(
        metadata_df,
        classification_column=args.classification_column,
        cell_line_col=cell_line_col_meta,
    )

    metadata_keep_df = build_metadata_keep_df(args, metadata_df)
    filtered_nuclei = merge_metadata_into_filtered_nuclei(filtered_nuclei, metadata_keep_df, args)

    classification_assigned = int(filtered_nuclei.obs["Classification"].notna().sum())
    classification_missing = int(filtered_nuclei.obs["Classification"].isna().sum())
    sigmoid_logits_assigned = int(filtered_nuclei.obs["Sigmoid_Logits"].notna().sum())
    sigmoid_logits_missing = int(filtered_nuclei.obs["Sigmoid_Logits"].isna().sum())
    filtered_class_canonical = canonicalize_binary_class(
        filtered_nuclei.obs["Classification"],
        normal_values=args.normal_class_values,
        abnormal_values=args.abnormal_class_values,
        label="filtered_nuclei Classification",
    )

    slide_stats_df = build_slide_stats_df(filtered_nuclei, filtered_class_canonical)

    if args.is_myonucleus_column not in filtered_nuclei.obs.columns:
        available = ", ".join(map(str, filtered_nuclei.obs.columns))
        raise KeyError(
            f"Column '{args.is_myonucleus_column}' not found in filtered_nuclei obs. "
            f"Available: {available}"
        )

    is_myonucleus_flag = to_myonucleus_flag(filtered_nuclei.obs[args.is_myonucleus_column])
    myonuclei = filtered_nuclei[is_myonucleus_flag.to_numpy(), :].copy()

    myotube_combined = derive_myotube_nucleus_counts(myonuclei, myotube_combined, args)
    has_nuclei = (
        (pd.Series(myotube_combined.obs["n_normal_nuclei"], copy=False) > 0)
        | (pd.Series(myotube_combined.obs["n_abnormal_nuclei"], copy=False) > 0)
    )
    myotube_filtered = myotube_combined[has_nuclei.to_numpy(), :].copy()

    decoupler_summary = None
    if args.decoupler_basepath is not None:
        decoupler_summary = {
            "myonuclei": attach_decoupler_exports(
                adata=myonuclei,
                dataset_key="myonuclei",
                decoupler_basepath=args.decoupler_basepath,
            ),
            "myotubes": attach_decoupler_exports(
                adata=myotube_filtered,
                dataset_key="myotubes",
                decoupler_basepath=args.decoupler_basepath,
            ),
        }

    filtered_nuclei_file = args.output_dir / args.output_myonuclei_nonmyonuclei_filename
    myonuclei_file = args.output_dir / args.output_myonuclei_filename
    myotube_file = args.output_dir / args.output_myotube_filename
    filtered_nuclei.write_h5ad(filtered_nuclei_file, compression="gzip")
    myonuclei.write_h5ad(myonuclei_file, compression="gzip")
    myotube_filtered.write_h5ad(myotube_file, compression="gzip")

    print_summary(
        raw_counts=raw_counts,
        filtered_nuclei=filtered_nuclei,
        myonuclei=myonuclei,
        myotube_combined=myotube_combined,
        myotube_filtered=myotube_filtered,
        classification_assigned=classification_assigned,
        classification_missing=classification_missing,
        sigmoid_logits_assigned=sigmoid_logits_assigned,
        sigmoid_logits_missing=sigmoid_logits_missing,
        metadata_summary=metadata_summary,
        slide_stats_df=slide_stats_df,
        filtered_nuclei_file=filtered_nuclei_file,
        myonuclei_file=myonuclei_file,
        myotube_file=myotube_file,
        args=args,
        decoupler_summary=decoupler_summary,
    )


if __name__ == "__main__":
    main()
