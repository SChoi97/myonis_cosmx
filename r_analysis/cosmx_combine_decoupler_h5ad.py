#!/usr/bin/env python3

"""Combine CosMx H5AD inputs with classifier metadata and decoupleR CSV outputs."""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd


MERGE_KEY_COLS = [
    "slide_key",
    "field_key",
    "patch_idx_key",
    "cell_line_key",
    "local_id_key",
]

CLASSIFIER_IMAGE_PATTERN = (
    r"^field_(?P<field>[^_]+)_patch_(?P<patch_idx>\d+)_"
    r"cellline_(?P<cell_line>.+?)_localid_(?P<local_id>\d+)$"
)

DECOUPLER_EXPORT_SPECS = {
    "nuclei": [
        {
            "filename": "myonuclei_mlm_zscore.csv",
            "obsm_key": "decoupler_pathwaysmlm_zscore",
            "availability_col": "pathwaysmlm_zscore_available",
            "obs_prefix": "mlm_z_",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myonuclei_mlm_pvalue.csv",
            "obsm_key": "decoupler_pathwaysmlm_pvalue",
            "availability_col": "pathwaysmlm_pvalue_available",
            "obs_prefix": "mlm_p_",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myonuclei_ulm_zscore.csv",
            "obsm_key": "decoupler_tfsulm_zscore",
            "availability_col": "tfsulm_zscore_available",
            "obs_prefix": None,
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
        {
            "filename": "myonuclei_ulm_pvalue.csv",
            "obsm_key": "decoupler_tfsulm_pvalue",
            "availability_col": "tfsulm_pvalue_available",
            "obs_prefix": None,
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
    ],
    "myotubes": [
        {
            "filename": "myotubes_mlm_zscore.csv",
            "obsm_key": "decoupler_pathwaysmlm_zscore",
            "availability_col": "pathwaysmlm_zscore_available",
            "obs_prefix": "mlm_z_",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myotubes_mlm_pvalue.csv",
            "obsm_key": "decoupler_pathwaysmlm_pvalue",
            "availability_col": "pathwaysmlm_pvalue_available",
            "obs_prefix": "mlm_p_",
            "reduction_prefixes": ["pathwaysmlm_pca_", "pathwaysmlm_umap_"],
        },
        {
            "filename": "myotubes_ulm_zscore.csv",
            "obsm_key": "decoupler_tfsulm_zscore",
            "availability_col": "tfsulm_zscore_available",
            "obs_prefix": None,
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
        {
            "filename": "myotubes_ulm_pvalue.csv",
            "obsm_key": "decoupler_tfsulm_pvalue",
            "availability_col": "tfsulm_pvalue_available",
            "obs_prefix": None,
            "reduction_prefixes": ["tfsulm_pca_", "tfsulm_umap_"],
        },
    ],
}


@dataclass
class DecouplerExport:
    ids: pd.Index
    metadata: pd.DataFrame
    scores: pd.DataFrame
    availability: pd.Series
    source_file: Path
    availability_col: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine T5R5/T6R6 CosMx H5ADs with classifier metadata and "
            "run_decoupleR.R CSV exports."
        )
    )
    parser.add_argument("--t5r5_nuclei_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t5r5_myotube_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t6r6_nuclei_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t6r6_myotube_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t5r5_metadata_path", type=Path, required=True)
    parser.add_argument("--t6r6_metadata_path", type=Path, required=True)
    parser.add_argument(
        "--decoupler_basepath",
        "--decoupleR_basepath",
        dest="decoupler_basepath",
        type=Path,
        required=True,
    )
    parser.add_argument("--savepath", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument(
        "--output_nuclei_filename",
        type=str,
        default="nuclei_decoupler_combined.h5ad",
    )
    parser.add_argument(
        "--output_myotube_filename",
        type=str,
        default="myotubes_decoupler_combined.h5ad",
    )
    return parser.parse_args()


def stop_if_missing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def stop_if_missing_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")


def validate_inputs(args: argparse.Namespace) -> None:
    stop_if_missing_file(args.t5r5_nuclei_combined_adata_path, "t5r5_nuclei_combined_adata_path")
    stop_if_missing_file(args.t5r5_myotube_combined_adata_path, "t5r5_myotube_combined_adata_path")
    stop_if_missing_file(args.t6r6_nuclei_combined_adata_path, "t6r6_nuclei_combined_adata_path")
    stop_if_missing_file(args.t6r6_myotube_combined_adata_path, "t6r6_myotube_combined_adata_path")
    stop_if_missing_file(args.t5r5_metadata_path, "t5r5_metadata_path")
    stop_if_missing_file(args.t6r6_metadata_path, "t6r6_metadata_path")
    stop_if_missing_dir(args.decoupler_basepath, "decoupler_basepath")

    for dataset_key, specs in DECOUPLER_EXPORT_SPECS.items():
        missing = [
            spec["filename"]
            for spec in specs
            if not (args.decoupler_basepath / spec["filename"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing expected {dataset_key} decoupleR exports in "
                f"{args.decoupler_basepath}: {', '.join(missing)}"
            )


def normalize_string_key(values: Iterable[object], uppercase: bool = False) -> pd.Series:
    series = pd.Series(values, copy=False)
    out = series.map(lambda x: pd.NA if pd.isna(x) else str(x).strip())
    if uppercase:
        out = out.map(lambda x: pd.NA if pd.isna(x) else str(x).upper())
    return out


def normalize_integer_key(values: Iterable[object], label: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    out = numeric.map(lambda x: pd.NA if pd.isna(x) else str(int(x)))
    bad = pd.Series(values, copy=False).notna() & out.isna()
    if bool(bad.any()):
        examples = pd.Series(values, copy=False).loc[bad].head(5).tolist()
        raise ValueError(f"Could not parse {label} values as integers. Examples: {examples}")
    return out


def make_obs_merge_keys(obs: pd.DataFrame) -> pd.DataFrame:
    required = ["Slide Name", "field", "patch_idx", "Cell Line", "local_id"]
    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise KeyError(
            "Missing required nuclei obs columns for classifier merge: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        {
            "slide_key": normalize_string_key(obs["Slide Name"], uppercase=True).to_numpy(),
            "field_key": normalize_string_key(obs["field"]).to_numpy(),
            "patch_idx_key": normalize_integer_key(obs["patch_idx"], "patch_idx").to_numpy(),
            "cell_line_key": normalize_string_key(obs["Cell Line"]).to_numpy(),
            "local_id_key": normalize_integer_key(obs["local_id"], "local_id").to_numpy(),
        },
        index=obs.index,
    )


def read_classifier_metadata(metadata_path: Path, slide: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = ["Image Name", "Predicted Class", "Sigmoid Logits"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"{metadata_path} is missing classifier columns: {', '.join(missing)}"
        )

    base = (
        df["Image Name"]
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.replace(r"\.[^.]+$", "", regex=True)
    )
    parsed = base.str.extract(CLASSIFIER_IMAGE_PATTERN)
    bad = parsed.isna().any(axis=1)
    if bool(bad.any()):
        examples = df.loc[bad, "Image Name"].head(5).tolist()
        raise ValueError(
            f"{slide}: {int(bad.sum())} classifier image names failed parsing. "
            f"Examples: {examples}"
        )

    out = pd.DataFrame(
        {
            "slide_key": slide.upper(),
            "field_key": normalize_string_key(parsed["field"]).to_numpy(),
            "patch_idx_key": normalize_integer_key(parsed["patch_idx"], "classifier patch_idx").to_numpy(),
            "cell_line_key": normalize_string_key(parsed["cell_line"]).to_numpy(),
            "local_id_key": normalize_integer_key(parsed["local_id"], "classifier local_id").to_numpy(),
            "Classification": df["Predicted Class"].to_numpy(),
            "Sigmoid_Logits": pd.to_numeric(df["Sigmoid Logits"], errors="coerce").to_numpy(),
            "classifier_image_name": df["Image Name"].astype(str).to_numpy(),
        }
    )

    duplicated = out.duplicated(MERGE_KEY_COLS, keep=False)
    if bool(duplicated.any()):
        warnings.warn(
            f"{slide}: classifier metadata has {int(duplicated.sum())} rows with "
            "duplicate merge keys; keeping the first row for each key.",
            stacklevel=2,
        )
        out = out.drop_duplicates(MERGE_KEY_COLS, keep="first")

    return out


def attach_classifier_metadata(
    nuclei: ad.AnnData,
    t5r5_metadata_path: Path,
    t6r6_metadata_path: Path,
) -> dict:
    metadata = pd.concat(
        [
            read_classifier_metadata(t5r5_metadata_path, "T5R5"),
            read_classifier_metadata(t6r6_metadata_path, "T6R6"),
        ],
        ignore_index=True,
    )

    obs = nuclei.obs.copy()
    obs["_row_id"] = nuclei.obs_names.to_numpy()
    obs = obs.drop(
        columns=[
            *MERGE_KEY_COLS,
            "Classification",
            "Sigmoid_Logits",
            "classifier_image_name",
        ],
        errors="ignore",
    )
    obs = obs.join(make_obs_merge_keys(obs))

    merged = obs.merge(metadata, how="left", on=MERGE_KEY_COLS, sort=False)
    merged = merged.set_index("_row_id").loc[nuclei.obs_names]
    merged.index.name = None
    merged = merged.drop(columns=MERGE_KEY_COLS, errors="ignore")
    nuclei.obs = merged

    return {
        "classifier_rows": int(metadata.shape[0]),
        "classification_assigned": int(nuclei.obs["Classification"].notna().sum()),
        "classification_missing": int(nuclei.obs["Classification"].isna().sum()),
        "sigmoid_logits_assigned": int(nuclei.obs["Sigmoid_Logits"].notna().sum()),
        "sigmoid_logits_missing": int(nuclei.obs["Sigmoid_Logits"].isna().sum()),
    }


def prepare_slide_adata(adata_obj: ad.AnnData, slide: str) -> ad.AnnData:
    adata_obj.obs["Slide Name"] = slide
    adata_obj.obs_names = pd.Index(
        [f"{slide}_{obs_name}" for obs_name in adata_obj.obs_names.astype(str)]
    )
    adata_obj.obs_names_make_unique()
    adata_obj.var_names_make_unique()
    return adata_obj


def read_and_prepare_h5ad(path: Path, slide: str, label: str) -> ad.AnnData:
    print(f"Reading {label}: {path}")
    return prepare_slide_adata(ad.read_h5ad(path), slide)


def combine_input_h5ads(args: argparse.Namespace) -> tuple[ad.AnnData, ad.AnnData]:
    t5r5_nuclei = read_and_prepare_h5ad(
        args.t5r5_nuclei_combined_adata_path,
        "T5R5",
        "T5R5 nuclei",
    )
    t6r6_nuclei = read_and_prepare_h5ad(
        args.t6r6_nuclei_combined_adata_path,
        "T6R6",
        "T6R6 nuclei",
    )
    t5r5_myotubes = read_and_prepare_h5ad(
        args.t5r5_myotube_combined_adata_path,
        "T5R5",
        "T5R5 myotubes",
    )
    t6r6_myotubes = read_and_prepare_h5ad(
        args.t6r6_myotube_combined_adata_path,
        "T6R6",
        "T6R6 myotubes",
    )

    print("Concatenating nuclei inputs")
    nuclei = ad.concat([t5r5_nuclei, t6r6_nuclei], join="outer")
    nuclei.obs_names_make_unique()
    nuclei.var_names_make_unique()
    nuclei.obs["decoupleR_cell_id"] = nuclei.obs_names.astype(str)

    print("Concatenating myotube inputs")
    myotubes = ad.concat([t5r5_myotubes, t6r6_myotubes], join="outer")
    myotubes.obs_names_make_unique()
    myotubes.var_names_make_unique()
    myotubes.obs["decoupleR_cell_id"] = myotubes.obs_names.astype(str)

    return nuclei, myotubes


def parse_bool_series(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0).astype(bool)

    normalized = values.astype(str).str.strip().str.upper()
    parsed = normalized.map(
        {
            "TRUE": True,
            "FALSE": False,
            "T": True,
            "F": False,
            "YES": True,
            "NO": False,
            "Y": True,
            "N": False,
            "1": True,
            "0": False,
        }
    )
    unexpected = values.notna() & parsed.isna()
    if bool(unexpected.any()):
        examples = values.loc[unexpected].head(5).tolist()
        warnings.warn(
            f"Unexpected boolean-like values in {label}; treating as False. "
            f"Examples: {examples}",
            stacklevel=2,
        )
    return parsed.fillna(False).astype(bool)


def make_unique_names(names: Iterable[object]) -> list[str]:
    seen = {}
    out = []
    for raw_name in names:
        name = str(raw_name)
        n_seen = seen.get(name, 0)
        out.append(name if n_seen == 0 else f"{name}__dup{n_seen}")
        seen[name] = n_seen + 1
    return out


def read_decoupler_export(
    csv_path: Path,
    availability_col: str,
    reduction_prefixes: Iterable[str],
) -> DecouplerExport:
    print(f"Reading decoupleR export: {csv_path}")
    df = pd.read_csv(csv_path, keep_default_na=True, na_values=["NaN"], low_memory=False)
    columns = list(df.columns)

    if "decoupleR_cell_id" not in df.columns:
        raise KeyError(f"Expected 'decoupleR_cell_id' in decoupleR export: {csv_path}")
    if availability_col not in df.columns:
        raise KeyError(
            f"Expected '{availability_col}' in decoupleR export {csv_path}. "
            f"Available columns start with: {', '.join(map(str, columns[:20]))}"
        )

    raw_ids = df["decoupleR_cell_id"]
    missing_ids = raw_ids.isna() | raw_ids.astype(str).str.strip().eq("")
    if bool(missing_ids.any()):
        raise ValueError(
            f"{csv_path} has {int(missing_ids.sum())} missing decoupleR_cell_id values."
        )

    ids = pd.Index(raw_ids.astype(str).str.strip(), name="decoupleR_cell_id")
    if ids.has_duplicates:
        dupes = ids[ids.duplicated()].unique().tolist()[:10]
        raise ValueError(
            f"{csv_path} has duplicate decoupleR_cell_id values. Examples: {dupes}"
        )

    availability_idx = columns.index(availability_col)
    metadata = df.iloc[:, :availability_idx].copy()
    metadata.index = ids

    stop_idx = columns.index("decoupleR_export_row") if "decoupleR_export_row" in columns else len(columns)
    score_positions = []
    score_names = []
    for col_idx in range(availability_idx + 1, stop_idx):
        col_name = str(columns[col_idx])
        if any(col_name.startswith(prefix) for prefix in reduction_prefixes):
            continue
        score_positions.append(col_idx)
        score_names.append(col_name)

    if not score_positions:
        raise ValueError(f"No decoupleR score columns were found in {csv_path}")

    score_names = make_unique_names(score_names)
    scores = df.iloc[:, score_positions].copy()
    scores.columns = score_names
    scores = scores.apply(pd.to_numeric, errors="coerce")
    scores.index = ids

    availability = parse_bool_series(df[availability_col], label=availability_col)
    availability.index = ids

    return DecouplerExport(
        ids=ids,
        metadata=metadata,
        scores=scores,
        availability=availability,
        source_file=csv_path,
        availability_col=availability_col,
    )


def require_same_decoupler_ids(primary_ids: pd.Index, export: DecouplerExport) -> None:
    missing = primary_ids.difference(export.ids)
    extra = export.ids.difference(primary_ids)
    if len(missing) or len(extra):
        raise ValueError(
            f"decoupleR export {export.source_file} does not contain the same "
            f"decoupleR_cell_id set as the primary export. "
            f"Missing from this export: {len(missing)}; extra in this export: {len(extra)}."
        )


def subset_adata_to_decoupler_rows(
    adata_obj: ad.AnnData,
    row_ids: pd.Index,
    label: str,
) -> ad.AnnData:
    adata_ids = pd.Index(adata_obj.obs_names.astype(str))
    missing = row_ids.difference(adata_ids)
    if len(missing):
        raise ValueError(
            f"{label}: {len(missing)} decoupleR_cell_id values are missing from "
            f"the combined H5AD. Examples: {missing[:10].tolist()}"
        )

    subset = adata_obj[row_ids.tolist(), :].copy()
    subset.obs["decoupleR_cell_id"] = subset.obs_names.astype(str)
    return subset


def series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    left = pd.Series(left).reset_index(drop=True)
    right = pd.Series(right).reset_index(drop=True)
    if len(left) != len(right):
        return False

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric_comparable = (left.isna() & right.isna()) | (
        left_num.notna() & right_num.notna()
    )
    if bool(numeric_comparable.all()):
        return bool(
            np.allclose(
                left_num.to_numpy(dtype=float),
                right_num.to_numpy(dtype=float),
                equal_nan=True,
            )
        )

    left_str = left.map(lambda x: "<NA>" if pd.isna(x) else str(x).strip())
    right_str = right.map(lambda x: "<NA>" if pd.isna(x) else str(x).strip())
    return bool(left_str.equals(right_str))


def unique_obs_column_name(obs: pd.DataFrame, requested: str) -> str:
    if requested not in obs.columns:
        return requested
    idx = 2
    while f"{requested}_{idx}" in obs.columns:
        idx += 1
    return f"{requested}_{idx}"


def copy_decoupler_metadata_to_obs(
    adata_obj: ad.AnnData,
    metadata: pd.DataFrame,
    label: str,
) -> dict:
    metadata = metadata.reindex(adata_obj.obs_names)
    copied = []
    conflicts = []

    for col in metadata.columns:
        incoming = metadata[col]
        if col in adata_obj.obs.columns:
            if series_values_equal(adata_obj.obs[col], incoming):
                continue
            target = unique_obs_column_name(adata_obj.obs, f"decoupleR_meta_{col}")
            conflicts.append((col, target))
        else:
            target = col

        adata_obj.obs[target] = incoming.to_numpy()
        copied.append(target)

    if conflicts:
        print(
            f"[{label}] decoupleR metadata conflicts copied with prefix: "
            + ", ".join(f"{src}->{dst}" for src, dst in conflicts[:10])
            + (" ..." if len(conflicts) > 10 else "")
        )

    return {
        "metadata_columns_copied": copied,
        "metadata_conflicts": [f"{src}->{dst}" for src, dst in conflicts],
    }


def attach_decoupler_scores(
    adata_obj: ad.AnnData,
    export: DecouplerExport,
    primary_ids: pd.Index,
    obsm_key: str,
    obs_prefix: Optional[str],
) -> dict:
    scores = export.scores.loc[primary_ids].copy()
    availability = export.availability.loc[primary_ids].fillna(False).astype(bool)
    scores.loc[~availability.to_numpy(), :] = np.nan

    adata_obj.obsm[obsm_key] = scores.to_numpy(dtype=float, copy=True)
    adata_obj.uns[f"{obsm_key}_cols"] = list(map(str, scores.columns))
    adata_obj.obs[export.availability_col] = availability.to_numpy(dtype=bool)

    if obs_prefix is not None:
        for col in scores.columns:
            adata_obj.obs[f"{obs_prefix}{col}"] = scores[col].to_numpy(dtype=float)

    return {
        "source_file": str(export.source_file),
        "availability_col": export.availability_col,
        "available_obs": int(availability.sum()),
        "n_rows_in_csv": int(export.scores.shape[0]),
        "n_features": int(scores.shape[1]),
    }


def attach_decoupler_bundle(
    adata_obj: ad.AnnData,
    dataset_key: str,
    decoupler_basepath: Path,
) -> tuple[ad.AnnData, dict]:
    specs = DECOUPLER_EXPORT_SPECS[dataset_key]
    primary_spec = specs[0]
    primary_export = read_decoupler_export(
        decoupler_basepath / primary_spec["filename"],
        availability_col=primary_spec["availability_col"],
        reduction_prefixes=primary_spec["reduction_prefixes"],
    )
    primary_ids = primary_export.ids

    adata_obj = subset_adata_to_decoupler_rows(
        adata_obj=adata_obj,
        row_ids=primary_ids,
        label=dataset_key,
    )
    metadata_summary = copy_decoupler_metadata_to_obs(
        adata_obj=adata_obj,
        metadata=primary_export.metadata,
        label=dataset_key,
    )

    files_summary = {
        primary_spec["obsm_key"]: attach_decoupler_scores(
            adata_obj=adata_obj,
            export=primary_export,
            primary_ids=primary_ids,
            obsm_key=primary_spec["obsm_key"],
            obs_prefix=primary_spec["obs_prefix"],
        )
    }

    for spec in specs[1:]:
        export = read_decoupler_export(
            decoupler_basepath / spec["filename"],
            availability_col=spec["availability_col"],
            reduction_prefixes=spec["reduction_prefixes"],
        )
        require_same_decoupler_ids(primary_ids, export)
        files_summary[spec["obsm_key"]] = attach_decoupler_scores(
            adata_obj=adata_obj,
            export=export,
            primary_ids=primary_ids,
            obsm_key=spec["obsm_key"],
            obs_prefix=spec["obs_prefix"],
        )

    adata_obj.uns["decoupleR"] = {
        "basepath": str(decoupler_basepath),
        "dataset_key": dataset_key,
        "row_key": "decoupleR_cell_id",
        "row_scope": "rows present in decoupleR CSV exports",
        "n_obs": int(adata_obj.n_obs),
        "files": files_summary,
        "metadata": metadata_summary,
    }

    return adata_obj, {
        "n_obs": int(adata_obj.n_obs),
        "files": files_summary,
        "metadata": metadata_summary,
    }


def sanitize_obs_for_h5ad(adata_obj: ad.AnnData) -> None:
    for col in list(adata_obj.obs.columns):
        series = adata_obj.obs[col]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if bool((series.isna() | numeric.notna()).all()):
            adata_obj.obs[col] = numeric
            continue

        adata_obj.obs[col] = series.map(lambda x: "" if pd.isna(x) else str(x))


def write_h5ad_atomic(adata_obj: ad.AnnData, output_path: Path) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    adata_obj.write_h5ad(tmp_path, compression="gzip")
    tmp_path.replace(output_path)


def print_summary(
    nuclei: ad.AnnData,
    myotubes: ad.AnnData,
    classifier_summary: dict,
    decoupler_summary: dict,
    nuclei_out: Path,
    myotubes_out: Path,
) -> None:
    print("\n========== SUMMARY ==========")
    print(f"Nuclei output: cells={nuclei.n_obs}, genes={nuclei.n_vars}")
    print(f"Myotube output: cells={myotubes.n_obs}, genes={myotubes.n_vars}")

    print("\nClassifier metadata")
    print(f"  rows read: {classifier_summary['classifier_rows']}")
    print(f"  final nuclei Classification assigned: {int(nuclei.obs['Classification'].notna().sum())}")
    print(f"  final nuclei Classification missing: {int(nuclei.obs['Classification'].isna().sum())}")
    print(f"  final nuclei Sigmoid_Logits assigned: {int(nuclei.obs['Sigmoid_Logits'].notna().sum())}")
    print(f"  final nuclei Sigmoid_Logits missing: {int(nuclei.obs['Sigmoid_Logits'].isna().sum())}")

    print("\ndecoupleR attachment")
    for label in ["nuclei", "myotubes"]:
        info = decoupler_summary[label]
        print(f"  {label}: rows={info['n_obs']}")
        for obsm_key, file_info in info["files"].items():
            print(
                f"    {obsm_key}: features={file_info['n_features']}, "
                f"available={file_info['available_obs']}"
            )

    print("\nSaved files:")
    print(f" - {nuclei_out}")
    print(f" - {myotubes_out}")
    print("=============================")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.savepath.mkdir(parents=True, exist_ok=True)

    nuclei, myotubes = combine_input_h5ads(args)
    classifier_summary = attach_classifier_metadata(
        nuclei=nuclei,
        t5r5_metadata_path=args.t5r5_metadata_path,
        t6r6_metadata_path=args.t6r6_metadata_path,
    )

    nuclei, nuclei_decoupler_summary = attach_decoupler_bundle(
        adata_obj=nuclei,
        dataset_key="nuclei",
        decoupler_basepath=args.decoupler_basepath,
    )
    myotubes, myotubes_decoupler_summary = attach_decoupler_bundle(
        adata_obj=myotubes,
        dataset_key="myotubes",
        decoupler_basepath=args.decoupler_basepath,
    )

    sanitize_obs_for_h5ad(nuclei)
    sanitize_obs_for_h5ad(myotubes)

    nuclei_out = args.savepath / f"{args.prefix}{args.output_nuclei_filename}"
    myotubes_out = args.savepath / f"{args.prefix}{args.output_myotube_filename}"

    print(f"Writing nuclei H5AD: {nuclei_out}")
    write_h5ad_atomic(nuclei, nuclei_out)
    print(f"Writing myotube H5AD: {myotubes_out}")
    write_h5ad_atomic(myotubes, myotubes_out)

    print_summary(
        nuclei=nuclei,
        myotubes=myotubes,
        classifier_summary=classifier_summary,
        decoupler_summary={
            "nuclei": nuclei_decoupler_summary,
            "myotubes": myotubes_decoupler_summary,
        },
        nuclei_out=nuclei_out,
        myotubes_out=myotubes_out,
    )


if __name__ == "__main__":
    main()
