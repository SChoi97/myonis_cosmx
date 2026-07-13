#!/usr/bin/env python3
"""Encode MYONIS CosMx h5ad files into a trained scVI latent space."""

import argparse
import copy
import json
import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scvi.model import SCVI

import scvi_main


CONFIG_NAME = "myonis_scvi_config.json"
CLASSIFIER_KEY_COLUMNS = [
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

try:
    SCVI_TOOLS_VERSION = version("scvi-tools")
except PackageNotFoundError:
    SCVI_TOOLS_VERSION = "unknown"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a latent .h5ad with the same cells/rows and obs metadata as a "
            "MYONIS CosMx input h5ad, replacing expression counts in .X with scVI embeddings."
        )
    )
    parser.add_argument(
        "--h5ad-paths",
        "--h5ad-path",
        "--datapath",
        nargs="+",
        required=True,
        dest="h5ad_paths",
        help="One or more input .h5ad files. Multiple files are preprocessed and concatenated like scvi_main.py.",
    )
    parser.add_argument("--savepath", required=True, help="Output .h5ad path or directory")
    parser.add_argument(
        "--pretrained-modelpath",
        "--model-dir",
        required=True,
        dest="pretrained_modelpath",
        help="Path to trained scVI model directory",
    )
    parser.add_argument(
        "--reference-h5ad",
        "--scvi_train_datapath",
        nargs="*",
        default=None,
        dest="reference_h5ad",
        help=(
            "Optional training/reference h5ad path(s). Only needed if the model was saved "
            "without embedded AnnData."
        ),
    )
    parser.add_argument("--input-mode", choices=["auto", "nuclei", "myotubes"], default=None)
    parser.add_argument(
        "--is-myonucleus",
        "--is_myonucleus",
        type=parse_bool,
        nargs="?",
        const=True,
        default=None,
        help=(
            "For nuclei inputs, keep only obs['myotube_id'] >= 0 before inference. "
            "If omitted, uses the training config when present, otherwise True."
        ),
    )
    parser.add_argument(
        "--covariates-to-remove",
        nargs="*",
        default=None,
        help="Covariates used during training. If omitted, uses the saved model config.",
    )
    parser.add_argument(
        "--continuous-covariate-transform",
        choices=["none", "zscore", "log1p-zscore"],
        default=None,
        help="Numeric covariate transform. If omitted, uses the saved model config.",
    )
    parser.add_argument("--counts-layer", default=None, help="Layer containing raw counts. Default uses config/.X")
    parser.add_argument("--var-join", choices=["inner", "outer"], default=None)
    parser.add_argument("--min-counts", type=float, default=None)
    parser.add_argument("--min-cells", type=int, default=None)
    parser.add_argument("--control-probe-pattern", default=None)
    parser.add_argument("--min-nuclei", type=int, default=None)
    parser.add_argument(
        "--filter-edge-nuclei",
        type=parse_bool,
        nargs="?",
        const=True,
        default=None,
        help=(
            "For nuclei inputs, keep only nuclei with obs['is_edge'] false. "
            "If omitted, uses the saved model config when present, otherwise True."
        ),
    )
    parser.add_argument(
        "--no-filter-edge-nuclei",
        dest="filter_edge_nuclei",
        action="store_false",
        default=None,
        help="Do not filter nuclei by obs['is_edge'].",
    )
    parser.add_argument(
        "--nuclei-h5ad-paths",
        nargs="*",
        default=None,
        help="Optional paired nuclei h5ad path(s) used to compute myotube n_nuclei.",
    )
    parser.add_argument(
        "--classifier-metadata-paths",
        "--classifier_metadata_paths",
        nargs="*",
        default=None,
        help=(
            "Optional nuclei classifier CSV(s), one per --h5ad-path, containing "
            "Image Name, Predicted Class, and Sigmoid Logits. For nuclei inference, "
            "these are aligned to obs and written as Classification and Sigmoid_Logits."
        ),
    )
    parser.add_argument(
        "--skip-count-validation",
        action="store_true",
        help="Skip validation of the filtered inference count matrix.",
    )
    parser.add_argument("--use-posterior", "--use_posterior", action="store_true")
    parser.add_argument(
        "--save-counts",
        action="store_true",
        help="Store the original counts matrix in obsm['counts'] and gene names in uns.",
    )
    parser.add_argument(
        "--library-size-adjusted-expression",
        "--library_size_adjusted_expression",
        action="store_true",
        help=(
            "Write an additional h5ad whose X is scVI-decoded expression for every "
            "gene at a common library size and whose obsm['X_scVI'] contains the latent embedding."
        ),
    )
    parser.add_argument(
        "--library-size-adjusted-expression-path",
        "--library_size_adjusted_expression_path",
        default=None,
        help=(
            "Optional path for the additional normalized-expression h5ad. By default, "
            "'<latent_stem>_library_size_adjusted_expression.h5ad' is written beside --savepath."
        ),
    )
    parser.add_argument(
        "--library-size-adjusted-expression-method",
        "--library_size_adjusted_expression_method",
        choices=["latent_covariate_neutral", "posterior_observed_covariates"],
        default="latent_covariate_neutral",
        help=(
            "Decoding method. 'latent_covariate_neutral' decodes posterior-mean z while "
            "fixing continuous nuisance covariates at their training mean and averaging "
            "equally over observed categorical/batch conditions (recommended for these models). "
            "'posterior_observed_covariates' uses SCVI.get_normalized_expression and retains "
            "the query's registered covariate values."
        ),
    )
    parser.add_argument(
        "--normalized-library-size",
        "--normalized_library_size",
        type=float,
        default=10000.0,
        help="Common library size for decoded expression. Default: 10000.",
    )
    parser.add_argument(
        "--normalized-expression-batch-size",
        "--normalized_expression_batch_size",
        type=int,
        default=512,
        help="Decoder minibatch size for the additional expression output. Default: 512.",
    )
    parser.add_argument(
        "--normalized-expression-n-samples",
        "--normalized_expression_n_samples",
        type=int,
        default=10,
        help=(
            "Posterior samples averaged by posterior_observed_covariates. Ignored by the "
            "recommended latent_covariate_neutral method. Default: 10."
        ),
    )
    parser.add_argument(
        "--normalized-expression-compression",
        "--normalized_expression_compression",
        choices=["none", "lzf", "gzip"],
        default="lzf",
        help="HDF5 compression for the additional expression h5ad. Default: lzf.",
    )
    parser.add_argument(
        "--copy-obsm",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Copy row-aligned obsm entries from the input h5ad. Default: True.",
    )
    parser.add_argument(
        "--copy-uns-keys",
        nargs="*",
        default=["morphology_feature_columns", "Object Contours"],
        help="Selected uns keys to copy to the latent h5ad. Use 'none' to copy no uns keys.",
    )
    return parser.parse_args()


def load_config(model_dir):
    path = Path(model_dir) / CONFIG_NAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_covariates(values):
    covariates = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in {"none", "null", "na"}:
                covariates.append(part)
    return list(dict.fromkeys(covariates))


def get_matrix(adata, counts_layer):
    if counts_layer is None:
        return adata.X
    if counts_layer not in adata.layers:
        raise KeyError(f"Layer {counts_layer!r} was requested but is not present in adata.layers")
    return adata.layers[counts_layer]


def row_sums(matrix):
    sums = matrix.sum(axis=1)
    if hasattr(sums, "A1"):
        return sums.A1
    return np.asarray(sums).reshape(-1)


def source_label_from_path(path):
    path = Path(path)
    for part in reversed(path.parts):
        lower = part.lower()
        if re.fullmatch(r"t\d+r\d+", lower):
            return lower
    if path.parent.name:
        return f"{path.parent.name}_{path.stem}"
    return path.stem


def make_unique_source_labels(paths):
    labels = []
    seen = {}
    for idx, path in enumerate(paths, start=1):
        label = source_label_from_path(path)
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or f"source_{idx}"
        count = seen.get(label, 0)
        seen[label] = count + 1
        labels.append(label if count == 0 else f"{label}_{count + 1}")
    return labels


def infer_mode(path, adata, requested_mode):
    if requested_mode and requested_mode != "auto":
        return requested_mode

    name = Path(path).name.lower()
    if "myotube" in name:
        return "myotubes"
    if "nuclei" in name or "nucleus" in name:
        return "nuclei"
    if "myotube_id" in adata.obs.columns:
        return "nuclei"
    return "myotubes"


def myonucleus_mask(adata):
    if "myotube_id" not in adata.obs.columns:
        raise KeyError(
            "Cannot apply --is-myonucleus=True because obs['myotube_id'] is missing. "
            "Use --is-myonucleus False for unassigned/all nuclei inputs."
        )
    values = pd.to_numeric(adata.obs["myotube_id"], errors="coerce").fillna(-1)
    return values.to_numpy() >= 0


def subset_object_contours(uns, keep_indices, n_original):
    if "Object Contours" not in uns:
        return

    obj = uns["Object Contours"]
    if not isinstance(obj, dict):
        return

    keep_indices = np.asarray(keep_indices, dtype=np.int64)

    contours = obj.get("Contours")
    if isinstance(contours, np.ndarray) and contours.shape[0] == n_original:
        obj["Contours"] = contours[keep_indices].copy()
    elif isinstance(contours, dict) and {"points", "offsets"}.issubset(contours):
        offsets = np.asarray(contours["offsets"])
        points = np.asarray(contours["points"])
        if offsets.ndim == 1 and offsets.shape[0] == n_original + 1:
            pieces = [points[offsets[i] : offsets[i + 1]] for i in keep_indices]
            if pieces:
                new_points = np.vstack(pieces).astype(points.dtype, copy=False)
                lengths = np.asarray([piece.shape[0] for piece in pieces], dtype=offsets.dtype)
                new_offsets = np.concatenate(
                    [np.asarray([0], dtype=offsets.dtype), np.cumsum(lengths, dtype=offsets.dtype)]
                )
            else:
                width = points.shape[1] if points.ndim == 2 else 2
                new_points = np.zeros((0, width), dtype=points.dtype)
                new_offsets = np.asarray([0], dtype=offsets.dtype)
            obj["Contours"] = {"points": new_points, "offsets": new_offsets}

    contour_offsets = obj.get("Contour offsets")
    if isinstance(contour_offsets, np.ndarray) and contour_offsets.shape[0] == n_original:
        obj["Contour offsets"] = contour_offsets[keep_indices].copy()


def read_and_filter_h5ad(path, requested_mode, is_myonucleus):
    path = Path(path)
    print(f"[scVI] Loading {path}", flush=True)
    adata = ad.read_h5ad(path)
    mode = infer_mode(path, adata, requested_mode)

    n_before = adata.n_obs
    if mode == "nuclei" and is_myonucleus:
        mask = myonucleus_mask(adata)
        keep_indices = np.flatnonzero(mask)
        adata = adata[mask].copy()
        subset_object_contours(adata.uns, keep_indices, n_before)
        print(
            f"[scVI] {path.name}: kept {adata.n_obs}/{n_before} myonuclei "
            f"(obs['myotube_id'] >= 0)",
            flush=True,
        )
    elif mode == "myotubes" and is_myonucleus:
        print(f"[scVI] {path.name}: input is myotubes, so --is-myonucleus is ignored", flush=True)

    if adata.var_names.has_duplicates:
        warnings.warn(f"{path.name} has duplicate var_names; making them unique for scVI.")
        adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        warnings.warn(f"{path.name} has duplicate obs_names; making them unique before scVI setup.")
        adata.obs_names_make_unique()

    return adata, mode


def load_reference_h5ads(paths, requested_mode, is_myonucleus, var_join):
    adatas = []
    modes = []
    for path in paths:
        adata, mode = read_and_filter_h5ad(path, requested_mode, is_myonucleus)
        adatas.append(adata)
        modes.append(mode)

    if len(set(modes)) != 1:
        raise ValueError(f"Mixed input modes are not supported in one reference model: {modes}")

    if len(adatas) == 1:
        return adatas[0]

    keys = make_unique_source_labels(paths)
    return ad.concat(
        adatas,
        axis=0,
        join=var_join,
        label="source_h5ad",
        keys=keys,
        index_unique="-",
        fill_value=0,
    )


def fit_or_apply_continuous_transform(values, covariate, transform, params=None):
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().all():
        values = values.fillna(0.0)
    else:
        values = values.fillna(float(values.median()))

    arr = values.to_numpy(dtype=np.float64)
    if params is not None:
        transform = params.get("transform", transform)

    used_transform = transform
    if transform == "log1p-zscore":
        if np.min(arr) < 0:
            warnings.warn(
                f"Continuous covariate {covariate!r} has negative values; using zscore instead of log1p-zscore."
            )
            used_transform = "zscore"
        else:
            arr = np.log1p(arr)

    if used_transform in {"zscore", "log1p-zscore"}:
        if params is None:
            mean = float(np.mean(arr)) if arr.size else 0.0
            std = float(np.std(arr)) if arr.size else 1.0
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
        else:
            mean = float(params.get("mean", 0.0))
            std = float(params.get("std", 1.0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
        arr = (arr - mean) / std

    return arr.astype(np.float32)


def prepare_covariates(
    adata,
    covariates,
    counts_layer,
    continuous_covariate_transform="none",
    covariate_transform_params=None,
):
    continuous_keys = []
    categorical_keys = []
    covariate_transform_params = covariate_transform_params or {}

    for covariate in covariates:
        if covariate == "n_counts":
            adata.obs[covariate] = row_sums(get_matrix(adata, counts_layer)).astype(np.float32)

        if covariate not in adata.obs.columns:
            raise KeyError(f"Requested covariate {covariate!r} is missing from adata.obs")

        series = adata.obs[covariate]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            adata.obs[covariate] = fit_or_apply_continuous_transform(
                series,
                covariate=covariate,
                transform=continuous_covariate_transform,
                params=covariate_transform_params.get(covariate),
            )
            continuous_keys.append(covariate)
        else:
            values = series.astype("string").fillna("missing").astype("category")
            if len(values.cat.categories) <= 1:
                warnings.warn(f"obs[{covariate!r}] has one level; ignoring it as a covariate.")
                continue
            adata.obs[covariate] = values
            categorical_keys.append(covariate)

    return continuous_keys, categorical_keys


def copy_obsm(adata):
    copied = {}
    for key in adata.obsm.keys():
        value = adata.obsm[key]
        try:
            copied[key] = value.copy()
        except AttributeError:
            copied[key] = copy.deepcopy(value)
    return copied


def copy_uns_keys(adata, keys):
    if not keys or any(str(key).lower() == "none" for key in keys):
        return {}
    copied = {}
    for key in keys:
        if key in adata.uns:
            copied[key] = copy.deepcopy(adata.uns[key])
    return copied


def _normalise_classifier_string(values, uppercase=False):
    out = pd.Series(values, copy=False).astype("string").str.strip()
    if uppercase:
        out = out.str.upper()
    return out


def _normalise_classifier_integer(values, label):
    original = pd.Series(values, copy=False)
    numeric = pd.to_numeric(original, errors="coerce")
    bad = original.notna() & numeric.isna()
    if bool(bad.any()):
        raise ValueError(
            f"Could not parse {label} values as integers. Examples: "
            f"{original.loc[bad].head(5).tolist()}"
        )
    return numeric.round().astype("Int64").astype("string")


def _classifier_obs_keys(obs):
    required = ["Slide Name", "field", "patch_idx", "Cell Line", "local_id"]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(
            "Missing nuclei obs columns required for classifier metadata alignment: "
            + ", ".join(missing)
        )
    return pd.DataFrame(
        {
            "slide_key": _normalise_classifier_string(obs["Slide Name"], uppercase=True).to_numpy(),
            "field_key": _normalise_classifier_string(obs["field"]).to_numpy(),
            "patch_idx_key": _normalise_classifier_integer(obs["patch_idx"], "patch_idx").to_numpy(),
            "cell_line_key": _normalise_classifier_string(obs["Cell Line"]).to_numpy(),
            "local_id_key": _normalise_classifier_integer(obs["local_id"], "local_id").to_numpy(),
        },
        index=obs.index,
    )


def _read_classifier_metadata(path, slide):
    path = Path(path)
    metadata = pd.read_csv(path)
    required = ["Image Name", "Predicted Class", "Sigmoid Logits"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise KeyError(f"{path} is missing classifier columns: {', '.join(missing)}")

    image_stem = (
        metadata["Image Name"]
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.replace(r"\.[^.]+$", "", regex=True)
    )
    parsed = image_stem.str.extract(CLASSIFIER_IMAGE_PATTERN)
    bad = parsed.isna().any(axis=1)
    if bool(bad.any()):
        raise ValueError(
            f"{path}: {int(bad.sum())} classifier image names could not be parsed. "
            f"Examples: {metadata.loc[bad, 'Image Name'].head(5).tolist()}"
        )

    out = pd.DataFrame(
        {
            "slide_key": str(slide).upper(),
            "field_key": _normalise_classifier_string(parsed["field"]).to_numpy(),
            "patch_idx_key": _normalise_classifier_integer(
                parsed["patch_idx"], "classifier patch_idx"
            ).to_numpy(),
            "cell_line_key": _normalise_classifier_string(parsed["cell_line"]).to_numpy(),
            "local_id_key": _normalise_classifier_integer(
                parsed["local_id"], "classifier local_id"
            ).to_numpy(),
            "Classification": pd.to_numeric(metadata["Predicted Class"], errors="coerce").to_numpy(),
            "Sigmoid_Logits": pd.to_numeric(metadata["Sigmoid Logits"], errors="coerce").to_numpy(),
            "classifier_image_name": metadata["Image Name"].astype(str).to_numpy(),
        }
    )
    duplicated = out.duplicated(CLASSIFIER_KEY_COLUMNS, keep=False)
    if bool(duplicated.any()):
        warnings.warn(
            f"{path}: {int(duplicated.sum())} classifier rows have duplicate alignment keys; "
            "keeping the first row for each key."
        )
        out = out.drop_duplicates(CLASSIFIER_KEY_COLUMNS, keep="first")
    return out


def attach_classifier_metadata(adata, metadata_paths, source_h5ad_paths, input_mode):
    """Add canonical morphology classifier columns to nuclei obs."""
    alias_candidates = {
        "Classification": ["Predicted Class", "classification", "predicted_class"],
        "Sigmoid_Logits": ["Sigmoid Logits", "sigmoid_logits", "sigmoid.logits"],
    }
    for canonical, aliases in alias_candidates.items():
        if canonical not in adata.obs:
            source = next((column for column in aliases if column in adata.obs), None)
            if source is not None:
                adata.obs[canonical] = adata.obs[source].to_numpy()

    if input_mode != "nuclei":
        if metadata_paths:
            warnings.warn("Ignoring --classifier-metadata-paths because the inference input is myotubes.")
        return {"status": "not_applicable_myotubes"}

    if not metadata_paths:
        present = [column for column in ["Classification", "Sigmoid_Logits"] if column in adata.obs]
        if len(present) < 2:
            warnings.warn(
                "Nuclei inputs do not contain Classification/Sigmoid_Logits. Pass "
                "--classifier-metadata-paths to add the morphology classifier metadata."
            )
        return {
            "status": "preserved_from_input" if present else "missing_no_metadata_paths",
            "columns_present": present,
        }

    if len(metadata_paths) != len(source_h5ad_paths):
        raise ValueError(
            "--classifier-metadata-paths must provide exactly one CSV per --h5ad-path "
            f"({len(metadata_paths)} versus {len(source_h5ad_paths)})."
        )

    frames = []
    for metadata_path, source_path in zip(metadata_paths, source_h5ad_paths):
        slide = scvi_main.slide_name_from_path(source_path)
        if slide is None:
            raise ValueError(f"Could not infer slide name from input H5AD path: {source_path}")
        frames.append(_read_classifier_metadata(metadata_path, slide))
    metadata = pd.concat(frames, ignore_index=True)
    if metadata.duplicated(CLASSIFIER_KEY_COLUMNS).any():
        raise ValueError("Classifier metadata alignment keys are duplicated across input CSVs.")

    keys = _classifier_obs_keys(adata.obs).reset_index(drop=True)
    keys["_classifier_row_order"] = np.arange(adata.n_obs)
    merged = keys.merge(metadata, how="left", on=CLASSIFIER_KEY_COLUMNS, sort=False, validate="many_to_one")
    merged = merged.sort_values("_classifier_row_order", kind="stable")
    if len(merged) != adata.n_obs:
        raise RuntimeError("Classifier metadata merge changed the number of inference observations.")

    for column in ["Classification", "Sigmoid_Logits", "classifier_image_name"]:
        values = pd.Series(merged[column].to_numpy(), index=adata.obs.index)
        if column in adata.obs:
            values = values.combine_first(pd.Series(adata.obs[column], index=adata.obs.index))
        adata.obs[column] = values.to_numpy()

    summary = {
        "status": "attached",
        "metadata_paths": [str(Path(path)) for path in metadata_paths],
        "classifier_rows": int(len(metadata)),
        "classification_assigned": int(pd.Series(adata.obs["Classification"]).notna().sum()),
        "classification_missing": int(pd.Series(adata.obs["Classification"]).isna().sum()),
        "sigmoid_logits_assigned": int(pd.Series(adata.obs["Sigmoid_Logits"]).notna().sum()),
        "sigmoid_logits_missing": int(pd.Series(adata.obs["Sigmoid_Logits"]).isna().sum()),
    }
    if summary["classification_assigned"] == 0:
        raise ValueError("No inference nuclei matched the supplied classifier metadata.")
    if summary["classification_missing"] or summary["sigmoid_logits_missing"]:
        warnings.warn(
            "Some inference nuclei did not match classifier metadata: "
            f"Classification missing={summary['classification_missing']:,}, "
            f"Sigmoid_Logits missing={summary['sigmoid_logits_missing']:,}."
        )
    print(f"[scVI] Classifier metadata: {summary}", flush=True)
    return summary


def resolve_savepath(savepath):
    savepath = Path(savepath)
    if savepath.suffix == ".h5ad":
        savepath.parent.mkdir(parents=True, exist_ok=True)
        return savepath
    savepath.mkdir(parents=True, exist_ok=True)
    return savepath / "latent.h5ad"


def resolve_library_size_adjusted_expression_path(latent_path, requested_path=None):
    latent_path = Path(latent_path)
    if requested_path is None:
        output_path = latent_path.with_name(
            f"{latent_path.stem}_library_size_adjusted_expression.h5ad"
        )
    else:
        requested_path = Path(requested_path)
        if requested_path.suffix == ".h5ad":
            output_path = requested_path
        else:
            output_path = requested_path / "library_size_adjusted_expression.h5ad"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == latent_path.resolve():
        raise ValueError(
            "The library-size-adjusted expression path must differ from the latent output path."
        )
    return output_path


def _training_conditioning_values(model):
    """Return training-mean continuous covariates and observed categorical patterns."""
    training_adata = model.adata
    if training_adata is None:
        raise RuntimeError(
            "Covariate-neutral decoding requires the model's training AnnData. "
            "Load the model with embedded AnnData or pass --reference-h5ad."
        )

    if "_scvi_extra_continuous_covs" in training_adata.obsm:
        continuous = np.asarray(
            training_adata.obsm["_scvi_extra_continuous_covs"],
            dtype=np.float32,
        )
        continuous_neutral = np.nanmean(continuous, axis=0, dtype=np.float64).astype(np.float32)
        continuous_neutral[~np.isfinite(continuous_neutral)] = 0.0
    else:
        continuous_neutral = None

    if "_scvi_batch" in training_adata.obs:
        batch_codes = pd.to_numeric(
            training_adata.obs["_scvi_batch"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.int64)
    else:
        batch_codes = np.zeros(training_adata.n_obs, dtype=np.int64)

    if "_scvi_extra_categorical_covs" in training_adata.obsm:
        categorical = np.asarray(training_adata.obsm["_scvi_extra_categorical_covs"])
        if categorical.ndim == 1:
            categorical = categorical.reshape(-1, 1)
        categorical = categorical.astype(np.int64, copy=False)
    else:
        categorical = np.zeros((training_adata.n_obs, 0), dtype=np.int64)

    conditioning = np.column_stack([batch_codes, categorical])
    conditioning_patterns = np.unique(conditioning, axis=0)
    if conditioning_patterns.size == 0:
        conditioning_patterns = np.zeros((1, 1), dtype=np.int64)

    if "_scvi_labels" in training_adata.obs:
        label_codes = pd.to_numeric(
            training_adata.obs["_scvi_labels"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.int64)
        label_code = int(pd.Series(label_codes).mode().iloc[0])
    else:
        label_code = 0

    return continuous_neutral, conditioning_patterns, label_code


def decode_latent_covariate_neutral_expression(
    model,
    latent_mean,
    *,
    library_size,
    batch_size,
):
    """Decode fixed-z expression at common depth and nuisance-neutral covariates.

    Continuous covariates are fixed at their training mean. Registered batch and
    extra categorical covariates are averaged equally across the combinations
    observed during training. This deliberately does not re-encode cells after
    changing covariates, so the output remains a function of the validated latent
    representation rather than raw library size.
    """
    if library_size <= 0 or not np.isfinite(library_size):
        raise ValueError("--normalized-library-size must be a finite positive number.")
    if batch_size < 1:
        raise ValueError("--normalized-expression-batch-size must be at least 1.")
    if getattr(model.module, "use_size_factor_key", False):
        raise NotImplementedError(
            "latent_covariate_neutral currently requires a model trained without "
            "setup_anndata(size_factor_key=...). Use posterior_observed_covariates instead."
        )

    latent_mean = np.asarray(latent_mean, dtype=np.float32)
    if latent_mean.ndim != 2 or latent_mean.shape[1] != int(model.module.n_latent):
        raise ValueError(
            f"Expected latent_mean with shape (n_obs, {model.module.n_latent}), "
            f"got {latent_mean.shape}."
        )

    continuous_neutral, conditioning_patterns, label_code = _training_conditioning_values(model)
    n_obs = latent_mean.shape[0]
    n_genes = int(model.adata.n_vars)
    expression = np.empty((n_obs, n_genes), dtype=np.float32)
    device = next(model.module.parameters()).device
    model.module.eval()

    print(
        f"[scVI] Decoding covariate-neutral expression: n_obs={n_obs}, "
        f"n_genes={n_genes}, conditioning_patterns={len(conditioning_patterns)}, "
        f"library_size={library_size:g}",
        flush=True,
    )

    with torch.inference_mode():
        for start in range(0, n_obs, batch_size):
            stop = min(start + batch_size, n_obs)
            current_n = stop - start
            z = torch.as_tensor(latent_mean[start:stop], dtype=torch.float32, device=device)
            library = torch.full(
                (current_n, 1),
                float(np.log(library_size)),
                dtype=torch.float32,
                device=device,
            )
            y = torch.full(
                (current_n, 1),
                label_code,
                dtype=torch.long,
                device=device,
            )
            if continuous_neutral is None:
                cont_covs = None
            else:
                cont_covs = torch.as_tensor(
                    np.broadcast_to(continuous_neutral, (current_n, len(continuous_neutral))).copy(),
                    dtype=torch.float32,
                    device=device,
                )

            decoded_sum = np.zeros((current_n, n_genes), dtype=np.float32)
            for pattern in conditioning_patterns:
                batch_index = torch.full(
                    (current_n, 1),
                    int(pattern[0]),
                    dtype=torch.long,
                    device=device,
                )
                if len(pattern) == 1:
                    cat_covs = None
                else:
                    categorical_values = np.broadcast_to(
                        pattern[1:],
                        (current_n, len(pattern) - 1),
                    ).copy()
                    cat_covs = torch.as_tensor(
                        categorical_values,
                        dtype=torch.long,
                        device=device,
                    )

                generative_outputs = model.module.generative(
                    z=z,
                    library=library,
                    batch_index=batch_index,
                    cont_covs=cont_covs,
                    cat_covs=cat_covs,
                    y=y,
                )
                if "px" not in generative_outputs or not hasattr(generative_outputs["px"], "scale"):
                    raise RuntimeError(
                        "The loaded scVI module did not return an expression-frequency scale."
                    )
                decoded_sum += (
                    generative_outputs["px"].scale.detach().float().cpu().numpy()
                )

            expression[start:stop] = (
                decoded_sum / float(len(conditioning_patterns)) * float(library_size)
            )
            if start == 0 or stop == n_obs or stop % (batch_size * 20) == 0:
                print(f"[scVI] Decoded {stop:,}/{n_obs:,} observations", flush=True)

    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Decoded expression contains non-finite or negative values.")
    decoded_library_sizes = expression.sum(axis=1, dtype=np.float64)
    if not np.allclose(decoded_library_sizes, library_size, rtol=2e-3, atol=1e-2):
        warnings.warn(
            "Decoded expression rows do not all sum to the requested common library size. "
            f"Observed range: {decoded_library_sizes.min():.6g} to "
            f"{decoded_library_sizes.max():.6g}."
        )

    metadata = {
        "method": "latent_covariate_neutral",
        "library_size": float(library_size),
        "latent_summary": "posterior_mean",
        "continuous_covariate_values": (
            [] if continuous_neutral is None else continuous_neutral.astype(float).tolist()
        ),
        "categorical_conditioning_patterns": conditioning_patterns.astype(int).tolist(),
        "categorical_pattern_weighting": "equal",
        "decoded_row_sum_min": float(decoded_library_sizes.min()),
        "decoded_row_sum_max": float(decoded_library_sizes.max()),
    }
    return expression, metadata


def decode_posterior_observed_covariate_expression(
    model,
    adata_query,
    *,
    library_size,
    batch_size,
    n_samples,
):
    """Use scVI's public posterior-normalized expression API."""
    if library_size <= 0 or not np.isfinite(library_size):
        raise ValueError("--normalized-library-size must be a finite positive number.")
    if batch_size < 1 or n_samples < 1:
        raise ValueError("Normalized-expression batch size and n_samples must be at least 1.")
    print(
        f"[scVI] Decoding posterior normalized expression with observed covariates: "
        f"library_size={library_size:g}, n_samples={n_samples}",
        flush=True,
    )
    expression = model.get_normalized_expression(
        adata=adata_query,
        library_size=float(library_size),
        n_samples=int(n_samples),
        batch_size=int(batch_size),
        return_mean=True,
        return_numpy=True,
    )
    expression = np.asarray(expression, dtype=np.float32)
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Decoded expression contains non-finite or negative values.")
    return expression, {
        "method": "posterior_observed_covariates",
        "library_size": float(library_size),
        "n_samples": int(n_samples),
        "categorical_pattern_weighting": "observed_query_covariates",
    }


def write_library_size_adjusted_expression(
    output_path,
    *,
    model,
    adata_query,
    obs,
    latent_mean,
    latent_output,
    copied_obsm,
    method,
    library_size,
    batch_size,
    n_samples,
    compression,
    source_h5ad,
    model_dir,
    covariates_to_remove,
    continuous_covariate_transform,
    classifier_metadata_summary,
):
    """Decode all trained genes and write expression plus scVI embeddings."""
    if method == "latent_covariate_neutral":
        expression, decoding_metadata = decode_latent_covariate_neutral_expression(
            model,
            latent_mean,
            library_size=library_size,
            batch_size=batch_size,
        )
    elif method == "posterior_observed_covariates":
        expression, decoding_metadata = decode_posterior_observed_covariate_expression(
            model,
            adata_query,
            library_size=library_size,
            batch_size=batch_size,
            n_samples=n_samples,
        )
    else:
        raise ValueError(f"Unknown normalized-expression method: {method!r}")

    trained_var_names = model.adata.var_names.astype(str)
    if expression.shape != (adata_query.n_obs, len(trained_var_names)):
        raise RuntimeError(
            "Decoded expression shape does not match query observations and trained genes: "
            f"{expression.shape} versus ({adata_query.n_obs}, {len(trained_var_names)})."
        )
    var = pd.DataFrame(index=pd.Index(trained_var_names, name=model.adata.var_names.name))
    expression_adata = ad.AnnData(X=expression, obs=obs.copy(), var=var)
    expression_adata.obsm["X_scVI"] = np.asarray(latent_mean, dtype=np.float32)
    if np.asarray(latent_output).shape != np.asarray(latent_mean).shape:
        expression_adata.obsm["X_scVI_output"] = np.asarray(latent_output, dtype=np.float32)
    for key, value in copied_obsm.items():
        if key not in expression_adata.obsm:
            expression_adata.obsm[key] = value

    expression_adata.uns["scvi_library_size_adjusted_expression"] = {
        **decoding_metadata,
        "model_dir": str(model_dir),
        "source_h5ad": [str(Path(path)) for path in source_h5ad],
        "n_obs": int(expression_adata.n_obs),
        "n_genes": int(expression_adata.n_vars),
        "expression_location": "X",
        "latent_location": "obsm['X_scVI']",
        "expression_dtype": str(expression_adata.X.dtype),
        "expression_is_dense": True,
        "values_are_noninteger_model_expectations": True,
        "scvi_tools_version": SCVI_TOOLS_VERSION,
        "registered_nuisance_covariates": list(covariates_to_remove),
        "continuous_covariate_transform": str(continuous_covariate_transform),
        "classifier_metadata": make_json_safe(classifier_metadata_summary),
    }

    compression_value = None if compression == "none" else compression
    print(
        f"[scVI] Saving library-size-adjusted expression h5ad to {output_path} "
        f"with shape {expression_adata.shape}",
        flush=True,
    )
    expression_adata.write_h5ad(output_path, compression=compression_value)
    print("[scVI] Library-size-adjusted expression output complete", flush=True)


def load_model(
    model_dir,
    reference_paths,
    requested_mode,
    is_myonucleus,
    var_join,
    covariates,
    counts_layer,
    continuous_covariate_transform,
    covariate_transform_params,
):
    try:
        return SCVI.load(model_dir)
    except Exception as exc:
        if not reference_paths:
            raise RuntimeError(
                "Could not load model without a reference AnnData. Re-run training without "
                "--no-save-anndata or pass --reference-h5ad/--scvi_train_datapath."
            ) from exc

    print("[scVI] Loading reference h5ad because the model does not contain saved AnnData", flush=True)
    reference = load_reference_h5ads(reference_paths, requested_mode, is_myonucleus, var_join)
    prepare_covariates(
        reference,
        covariates,
        counts_layer,
        continuous_covariate_transform=continuous_covariate_transform,
        covariate_transform_params=covariate_transform_params,
    )
    return SCVI.load(model_dir, adata=reference)


def build_inference_adata(
    paths,
    input_mode,
    is_myonucleus,
    var_join,
    counts_layer,
    min_counts,
    min_cells,
    control_probe_pattern,
    min_nuclei,
    filter_edge_nuclei,
    nuclei_h5ad_paths,
    skip_count_validation,
):
    adata, query_mode, summaries = scvi_main.load_h5ads(
        paths,
        requested_mode=input_mode,
        is_myonucleus=is_myonucleus,
        var_join=var_join,
        filter_edge_nuclei=filter_edge_nuclei,
    )
    adata, filter_summary = scvi_main.apply_standard_filters(
        adata,
        counts_layer=counts_layer,
        min_counts=min_counts,
        min_cells=min_cells,
        control_probe_pattern=control_probe_pattern,
    )

    nuclei_sources = []
    nuclei_edge_summary = None
    nuclei_filter_summary = None
    myotube_assignment_summary = None
    if query_mode == "myotubes" and min_nuclei and min_nuclei > 0:
        nuclei_adata, nuclei_sources = scvi_main.load_paired_nuclei_h5ads(
            paths,
            nuclei_h5ad_paths,
            var_join,
        )
        nuclei_adata, nuclei_edge_summary = scvi_main.filter_nuclei_edges_for_matching(nuclei_adata)
        nuclei_adata, nuclei_filter_summary = scvi_main.apply_standard_filters(
            nuclei_adata,
            counts_layer=counts_layer,
            min_counts=min_counts,
            min_cells=min_cells,
            control_probe_pattern=control_probe_pattern,
        )
        nuclei_counts, nuclei_source, myotube_assignment_summary = (
            scvi_main.compute_myotube_nucleus_counts_from_nuclei(
                adata,
                nuclei_adata,
                ", ".join(nuclei_sources),
            )
        )
        adata, _ = scvi_main.apply_min_nuclei_filter(adata, nuclei_counts, nuclei_source, min_nuclei)

    if not skip_count_validation:
        scvi_main.validate_count_matrix(adata, counts_layer)
    scvi_main.cast_count_matrix_to_float32(adata, counts_layer)

    preprocessing_summary = {
        "input_summaries": summaries,
        "filter_summary": filter_summary,
        "nuclei_sources": nuclei_sources,
        "nuclei_edge_summary": nuclei_edge_summary,
        "nuclei_filter_summary": nuclei_filter_summary,
        "myotube_assignment_summary": myotube_assignment_summary,
    }
    return adata, query_mode, preprocessing_summary


def make_json_safe(value):
    """Convert run metadata into plain JSON-compatible Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Index, pd.Series)):
        return make_json_safe(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    model_dir = Path(args.pretrained_modelpath)
    config = load_config(model_dir)

    input_mode = args.input_mode or config.get("input_mode", "auto")
    is_myonucleus = args.is_myonucleus
    if is_myonucleus is None:
        is_myonucleus = bool(config.get("is_myonucleus", True))

    counts_layer = args.counts_layer
    if counts_layer is None:
        counts_layer = config.get("counts_layer")

    covariates = args.covariates_to_remove
    if covariates is None:
        covariates = config.get("covariates_to_remove", [])
    covariates = flatten_covariates(covariates)
    continuous_covariate_transform = args.continuous_covariate_transform
    if continuous_covariate_transform is None:
        continuous_covariate_transform = config.get("continuous_covariate_transform", "none")
    covariate_transform_params = config.get("covariate_transform_params", {})

    var_join = args.var_join or config.get("var_join", "inner")
    min_counts = args.min_counts
    if min_counts is None:
        min_counts = config.get("min_counts", 0)
    min_cells = args.min_cells
    if min_cells is None:
        min_cells = config.get("min_cells", 0)
    control_probe_pattern = args.control_probe_pattern
    if control_probe_pattern is None:
        control_probe_pattern = config.get("control_probe_pattern", "none")
    min_nuclei = args.min_nuclei
    if min_nuclei is None:
        min_nuclei = config.get("min_nuclei", 0)
    filter_edge_nuclei = args.filter_edge_nuclei
    if filter_edge_nuclei is None:
        filter_edge_nuclei = bool(config.get("filter_edge_nuclei", True))
    nuclei_h5ad_paths = args.nuclei_h5ad_paths
    if nuclei_h5ad_paths is None:
        nuclei_h5ad_paths = config.get("nuclei_h5ad_paths", None)

    reference_h5ad = args.reference_h5ad or config.get("h5ad_paths")

    adata_query, query_mode, preprocessing_summary = build_inference_adata(
        args.h5ad_paths,
        input_mode,
        is_myonucleus,
        var_join,
        counts_layer,
        min_counts,
        min_cells,
        control_probe_pattern,
        min_nuclei,
        filter_edge_nuclei,
        nuclei_h5ad_paths,
        args.skip_count_validation,
    )

    classifier_metadata_summary = attach_classifier_metadata(
        adata_query,
        args.classifier_metadata_paths,
        args.h5ad_paths,
        query_mode,
    )
    preprocessing_summary["classifier_metadata_summary"] = classifier_metadata_summary

    obs_out = adata_query.obs.copy()
    obs_out["total_counts"] = row_sums(get_matrix(adata_query, counts_layer)).astype(np.float32)
    obsm_out = copy_obsm(adata_query) if args.copy_obsm else {}
    uns_out = copy_uns_keys(adata_query, args.copy_uns_keys)

    counts_matrix = None
    counts_var_names = None
    if args.save_counts:
        counts_matrix = get_matrix(adata_query, counts_layer)
        counts_matrix = counts_matrix.copy() if hasattr(counts_matrix, "copy") else np.array(counts_matrix)
        counts_var_names = adata_query.var_names.astype(str).to_numpy()

    prepare_covariates(
        adata_query,
        covariates,
        counts_layer,
        continuous_covariate_transform=continuous_covariate_transform,
        covariate_transform_params=covariate_transform_params,
    )
    model = load_model(
        model_dir,
        reference_h5ad,
        input_mode,
        is_myonucleus,
        var_join,
        covariates,
        counts_layer,
        continuous_covariate_transform,
        covariate_transform_params,
    )

    print("[scVI] Preparing query AnnData", flush=True)
    SCVI.prepare_query_anndata(adata_query, model)

    print("[scVI] Encoding query data to latent space", flush=True)
    if args.use_posterior:
        z_mu, z_var = model.get_latent_representation(adata=adata_query, return_dist=True)
        eps = np.random.randn(*z_mu.shape)
        z = z_mu + eps * np.sqrt(z_var)
        log_l = model.get_latent_library_size(adata=adata_query, give_mean=False).reshape(-1, 1)
        latent = np.concatenate([z, log_l], axis=1)
        latent_mean = np.asarray(z_mu, dtype=np.float32)
    else:
        latent_mean = np.asarray(
            model.get_latent_representation(adata=adata_query),
            dtype=np.float32,
        )
        latent = latent_mean

    latent = np.asarray(latent, dtype=np.float32)
    latent_var = pd.DataFrame(
        index=pd.Index([f"scvi_latent_{i}" for i in range(latent.shape[1])], name="latent_dimension")
    )
    adata_latent = ad.AnnData(X=latent, obs=obs_out, var=latent_var)

    for key, value in obsm_out.items():
        adata_latent.obsm[key] = value
    for key, value in uns_out.items():
        adata_latent.uns[key] = value

    if args.save_counts:
        adata_latent.obsm["counts"] = counts_matrix if sp.issparse(counts_matrix) else sp.csr_matrix(counts_matrix)
        adata_latent.uns["counts_var_names"] = counts_var_names

    out_path = resolve_savepath(args.savepath)
    adjusted_expression_path = None
    if args.library_size_adjusted_expression:
        adjusted_expression_path = resolve_library_size_adjusted_expression_path(
            out_path,
            args.library_size_adjusted_expression_path,
        )

    adata_latent.uns["scvi_latent"] = {
        "model_dir": str(model_dir),
        "source_h5ad": [str(Path(path)) for path in args.h5ad_paths],
        "input_mode": query_mode,
        "is_myonucleus": bool(is_myonucleus),
        "covariates_to_remove": covariates,
        "continuous_covariate_transform": continuous_covariate_transform,
        "counts_layer": "" if counts_layer is None else str(counts_layer),
        "n_latent": int(latent.shape[1]),
        "var_join": var_join,
        "min_counts": min_counts,
        "min_cells": min_cells,
        "control_probe_pattern": control_probe_pattern,
        "min_nuclei": min_nuclei,
        "filter_edge_nuclei": bool(filter_edge_nuclei),
        "preprocessing_summary_format": "json",
        "preprocessing_summary_json": json.dumps(
            make_json_safe(preprocessing_summary),
            indent=2,
            sort_keys=True,
        ),
        "library_size_adjusted_expression_requested": bool(
            args.library_size_adjusted_expression
        ),
        "library_size_adjusted_expression_path": (
            "" if adjusted_expression_path is None else str(adjusted_expression_path)
        ),
        "library_size_adjusted_expression_method": (
            "" if adjusted_expression_path is None else args.library_size_adjusted_expression_method
        ),
        "normalized_library_size": (
            None if adjusted_expression_path is None else float(args.normalized_library_size)
        ),
    }

    print(f"[scVI] Saving latent h5ad to {out_path}", flush=True)
    adata_latent.write_h5ad(out_path)
    print(f"[scVI] Done. Latent shape: {adata_latent.shape}", flush=True)

    if adjusted_expression_path is not None:
        write_library_size_adjusted_expression(
            adjusted_expression_path,
            model=model,
            adata_query=adata_query,
            obs=obs_out,
            latent_mean=latent_mean,
            latent_output=latent,
            copied_obsm=obsm_out,
            method=args.library_size_adjusted_expression_method,
            library_size=args.normalized_library_size,
            batch_size=args.normalized_expression_batch_size,
            n_samples=args.normalized_expression_n_samples,
            compression=args.normalized_expression_compression,
            source_h5ad=args.h5ad_paths,
            model_dir=model_dir,
            covariates_to_remove=covariates,
            continuous_covariate_transform=continuous_covariate_transform,
            classifier_metadata_summary=classifier_metadata_summary,
        )


if __name__ == "__main__":
    main()
