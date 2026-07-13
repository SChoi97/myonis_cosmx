#!/usr/bin/env python3
"""Train scVI on MYONIS CosMx nuclei or myotube h5ad files."""

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch
from lightning.pytorch.callbacks import Callback
from scvi.distributions import _negative_binomial as scvi_nb
from scvi.model import SCVI


CONFIG_NAME = "myonis_scvi_config.json"


class ValidationLossPrinter(Callback):
    def __init__(self, every_n_epochs=1, output_path=None, print_metrics=True):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.output_path = Path(output_path) if output_path is not None else None
        self.print_metrics = bool(print_metrics)
        self.metric_keys = [
            "validation_loss",
            "elbo_validation",
            "reconstruction_loss_validation",
            "kl_local_validation",
            "train_loss",
        ]

    @staticmethod
    def _metric_value(value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value
        return value

    @classmethod
    def _format_metric(cls, value):
        value = cls._metric_value(value)
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)

    def _write_metrics(self, row):
        if self.output_path is None or len(row) <= 1:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        write_header = not self.output_path.exists()
        df.to_csv(self.output_path, mode="a", header=write_header, index=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        epoch = int(getattr(trainer, "current_epoch", 0)) + 1

        metrics = getattr(trainer, "callback_metrics", {})
        row = {"epoch": epoch}
        for key in self.metric_keys:
            if key in metrics:
                row[key] = self._metric_value(metrics[key])
        self._write_metrics(row)

        if not self.print_metrics or epoch % self.every_n_epochs != 0:
            return

        pieces = [f"{key}={self._format_metric(metrics[key])}" for key in self.metric_keys if key in metrics]
        if pieces:
            print(f"[scVI] Epoch {epoch} validation: " + ", ".join(pieces), flush=True)


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
            "Train an scVI model on MYONIS CosMx combined_nuclei.h5ad or "
            "combined_myotubes.h5ad files."
        )
    )
    parser.add_argument(
        "--h5ad-paths",
        "--h5ad-path",
        nargs="+",
        required=True,
        dest="h5ad_paths",
        help="One or more input .h5ad files. Multiple files are concatenated by cells.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory to save the trained scVI model")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "nuclei", "myotubes"],
        default="auto",
        help="Input object type. In auto mode this is inferred from filename/obs columns.",
    )
    parser.add_argument(
        "--is-myonucleus",
        "--is_myonucleus",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "For nuclei inputs, keep only nuclei assigned to a myotube "
            "(obs['myotube_id'] >= 0). Ignored for myotube inputs. Default: True."
        ),
    )
    parser.add_argument(
        "--covariates-to-remove",
        nargs="*",
        default=[],
        help=(
            "List of obs covariates to condition on in scVI. Numeric covariates are "
            "used as continuous covariates, non-numeric as categorical covariates. "
            "Use 'n_counts' to compute total counts per cell if it is not already in obs."
        ),
    )
    parser.add_argument("--counts-layer", default=None, help="Layer containing raw counts. Default: use .X")
    parser.add_argument(
        "--var-join",
        choices=["inner", "outer"],
        default="inner",
        help="Gene join strategy when multiple h5ad files are supplied.",
    )
    parser.add_argument(
        "--min-counts",
        type=float,
        default=100.0,
        help="Minimum total counts per object/cell before training. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=50,
        help="Minimum number of objects/cells with nonzero counts required per gene. Use 0 to disable.",
    )
    parser.add_argument(
        "--control-probe-pattern",
        default="SystemControl|Negative",
        help=(
            "Case-insensitive regex for probes/genes to remove before training. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--min-nuclei",
        type=int,
        default=1,
        help=(
            "For myotube inputs, keep myotubes with at least this many assigned nuclei. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--nuclei-h5ad-paths",
        nargs="*",
        default=None,
        help=(
            "Optional paired nuclei h5ad path(s) used to compute myotube n_nuclei. "
            "If omitted, conventional combined_myotubes.h5ad -> combined_nuclei.h5ad "
            "and count_matrix_myotubes -> count_matrix_nuclei paths are tried."
        ),
    )
    parser.add_argument(
        "--filter-edge-nuclei",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "For nuclei inputs, keep only nuclei with obs['is_edge'] false when that "
            "column exists. Default: True."
        ),
    )
    parser.add_argument(
        "--no-filter-edge-nuclei",
        dest="filter_edge_nuclei",
        action="store_false",
        help="Do not filter nuclei by obs['is_edge'].",
    )

    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--gene-likelihood",
        choices=["zinb", "nb"],
        default="zinb",
        help="Count likelihood for scVI. Default: zinb.",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument(
        "--validation-loss-print-interval",
        type=int,
        default=1,
        help="Print validation metrics every N validation epochs. Default: 1.",
    )
    parser.add_argument(
        "--no-print-validation-loss",
        action="store_true",
        help="Do not print validation metrics during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--inference",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "After training, write an h5ad for the training data with .X replaced by "
            "the scVI latent embedding. Default: True."
        ),
    )
    parser.add_argument(
        "--no-inference",
        dest="inference",
        action="store_false",
        help="Skip writing the post-training latent h5ad.",
    )
    parser.add_argument(
        "--inference-output-path",
        default=None,
        help=(
            "Optional output path for --inference. Default: "
            "<model-dir>/<model-dir-name>.h5ad."
        ),
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16-mixed"], default="32")
    parser.add_argument(
        "--continuous-covariate-transform",
        choices=["none", "zscore", "log1p-zscore"],
        default="log1p-zscore",
        help="Transform numeric covariates before passing them to scVI. Default: log1p-zscore.",
    )
    parser.add_argument(
        "--skip-count-validation",
        action="store_true",
        help="Skip validation that the scVI count matrix is finite and nonnegative.",
    )
    parser.add_argument(
        "--no-cast-counts-float32",
        action="store_true",
        help="Do not cast the registered count matrix to float32 before scVI setup.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["high", "medium", "default"],
        default="high",
        help="torch.set_float32_matmul_precision setting.",
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument(
        "--cuda-lgamma-mode",
        choices=["auto", "torch", "stirling"],
        default="auto",
        help=(
            "How to handle log-gamma in CUDA NB/ZINB losses. 'auto' tests torch.lgamma "
            "and falls back to a differentiable Stirling approximation if the CUDA "
            "kernel is broken. Use 'torch' to require native torch.lgamma."
        ),
    )
    parser.add_argument(
        "--no-save-anndata",
        action="store_true",
        help="Do not store the training AnnData inside the saved scVI model directory.",
    )
    parser.add_argument("--sweep-name", default=None, help="Optional sweep name saved into run metadata.")
    parser.add_argument("--run-id", default=None, help="Optional sweep run identifier saved into run metadata.")
    parser.add_argument("--manifest-path", default=None, help="Optional sweep manifest path saved into run metadata.")
    return parser.parse_args()


def strip_control_chars(value, label):
    if value is None:
        return None
    original = str(value)
    cleaned = original.replace("\r", "").replace("\n", "").replace("\x00", "")
    if cleaned != original:
        warnings.warn(
            f"Removed hidden control characters from {label}. "
            "This prevents corrupted output paths on mounted filesystems."
        )
    return cleaned


def sanitize_runtime_args(args):
    for attr in [
        "model_dir",
        "counts_layer",
        "control_probe_pattern",
        "inference_output_path",
        "sweep_name",
        "run_id",
        "manifest_path",
    ]:
        setattr(args, attr, strip_control_chars(getattr(args, attr), f"--{attr.replace('_', '-')}"))

    args.h5ad_paths = [strip_control_chars(path, "--h5ad-paths") for path in args.h5ad_paths]
    if args.nuclei_h5ad_paths:
        args.nuclei_h5ad_paths = [
            strip_control_chars(path, "--nuclei-h5ad-paths") for path in args.nuclei_h5ad_paths
        ]
    args.covariates_to_remove = [
        strip_control_chars(value, "--covariates-to-remove") for value in (args.covariates_to_remove or [])
    ]
    return args


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


def col_nonzero_counts(matrix):
    if hasattr(matrix, "getnnz"):
        return np.asarray(matrix.getnnz(axis=0)).reshape(-1)
    return np.count_nonzero(np.asarray(matrix), axis=0)


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


def slide_name_from_path(path):
    for part in Path(path).parts:
        lower = part.lower()
        if re.fullmatch(r"t\d+r\d+", lower):
            return lower.upper()
    return None


def ensure_slide_name_from_path(adata, path):
    if "Slide Name" in adata.obs.columns:
        return

    slide_name = slide_name_from_path(path)
    if slide_name is None:
        return

    adata.obs["Slide Name"] = slide_name
    print(f"[scVI] {Path(path).name}: added obs['Slide Name']={slide_name!r} from input path", flush=True)


def infer_nuclei_path(myotube_path):
    path = Path(myotube_path)
    candidates = []

    if "myotubes" in path.name:
        candidates.append(path.with_name(path.name.replace("myotubes", "nuclei")))

    path_str = str(path)
    for old, new in (
        ("count_matrix_myotubes", "count_matrix_nuclei"),
        ("combined_myotubes.h5ad", "combined_nuclei.h5ad"),
        ("myotubes", "nuclei"),
    ):
        if old in path_str:
            candidates.append(Path(path_str.replace(old, new)))

    candidates.append(path.parent / "combined_nuclei.h5ad")
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate
    return None


def infer_mode(path, adata, requested_mode):
    if requested_mode != "auto":
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


def non_edge_nucleus_mask(adata, edge_col="is_edge"):
    if edge_col not in adata.obs.columns:
        warnings.warn(f"Cannot apply edge nucleus filter because obs[{edge_col!r}] is missing.")
        return np.ones(adata.n_obs, dtype=bool)

    values = adata.obs[edge_col]
    if pd.api.types.is_bool_dtype(values):
        is_edge = values.fillna(False).to_numpy(dtype=bool)
    elif pd.api.types.is_numeric_dtype(values):
        is_edge = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy() != 0
    else:
        normalized = values.astype("string").str.strip().str.lower()
        is_edge = normalized.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)

    return ~is_edge


def subset_obs_with_uns(adata, mask):
    mask = np.asarray(mask, dtype=bool)
    keep_indices = np.flatnonzero(mask)
    n_before = adata.n_obs
    filtered = adata[mask].copy()
    subset_object_contours(filtered.uns, keep_indices, n_before)
    return filtered


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


def read_and_filter_h5ad(path, requested_mode, is_myonucleus, filter_edge_nuclei=True):
    path = Path(path)
    print(f"[scVI] Loading {path}", flush=True)
    adata = ad.read_h5ad(path)
    ensure_slide_name_from_path(adata, path)
    mode = infer_mode(path, adata, requested_mode)

    n_before = adata.n_obs
    summary = {
        "path": str(path),
        "n_before": int(n_before),
        "filter_edge_nuclei": bool(filter_edge_nuclei),
        "edge_col": "is_edge",
        "n_before_edge_filter": None,
        "n_after_edge_filter": None,
        "n_before_myonucleus_filter": None,
        "n_after_myonucleus_filter": None,
    }

    if mode == "nuclei" and filter_edge_nuclei:
        before = adata.n_obs
        mask = non_edge_nucleus_mask(adata)
        adata = subset_obs_with_uns(adata, mask)
        summary["n_before_edge_filter"] = int(before)
        summary["n_after_edge_filter"] = int(adata.n_obs)
        print(
            f"[scVI] {path.name}: kept {adata.n_obs}/{before} non-edge nuclei "
            "(obs['is_edge'] is false)",
            flush=True,
        )

    if mode == "nuclei" and is_myonucleus:
        before = adata.n_obs
        mask = myonucleus_mask(adata)
        adata = subset_obs_with_uns(adata, mask)
        summary["n_before_myonucleus_filter"] = int(before)
        summary["n_after_myonucleus_filter"] = int(adata.n_obs)
        print(
            f"[scVI] {path.name}: kept {adata.n_obs}/{before} myonuclei "
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

    summary["n_after"] = int(adata.n_obs)
    return adata, mode, summary


def myotube_nucleus_counts_from_obs(adata):
    for col in ("n_nuclei", "n_myonuclei"):
        if col in adata.obs.columns:
            counts = pd.to_numeric(adata.obs[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            return counts, col

    count_cols = [col for col in ("n_normal_nuclei", "n_abnormal_nuclei") if col in adata.obs.columns]
    if count_cols:
        total = np.zeros(adata.n_obs, dtype=np.float64)
        for col in count_cols:
            total += pd.to_numeric(adata.obs[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        return total, "+".join(count_cols)

    return None, None


def normalized_key_series(df, col):
    if col in {"patch_idx", "local_id", "myotube_id", "myotube_patch_idx", "_match_patch_idx", "_match_local_id"}:
        return pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.int64).astype(str)
    return df[col].astype(str)


def build_match_keys(df, columns):
    values = [normalized_key_series(df, col) for col in columns]
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="__")
    return out


def normalise_col_for_key(series):
    series = pd.Series(series).copy()
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.95:
        out = numeric.round().astype("Int64").astype(str)
        return out.replace("<NA>", "NA")

    out = series.astype(str).str.strip()
    return out.replace({"nan": "NA", "None": "NA", "<NA>": "NA"})


def build_notebook_match_key(df, context_cols, id_col):
    key_parts = []
    for col in context_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found.")
        key_parts.append(normalise_col_for_key(df[col]))

    if id_col not in df.columns:
        raise ValueError(f"ID column {id_col!r} not found.")
    key_parts.append(normalise_col_for_key(df[id_col]))

    return pd.concat(key_parts, axis=1).astype(str).agg("|".join, axis=1)


def compute_myotube_nucleus_counts_from_nuclei(myotube_adata, nuclei_adata, source):
    context_cols = ["field", "patch_idx", "Cell Line", "Slide Name"]
    missing_myo = [col for col in [*context_cols, "local_id"] if col not in myotube_adata.obs.columns]
    missing_nuc = [col for col in [*context_cols, "myotube_id"] if col not in nuclei_adata.obs.columns]
    if missing_myo:
        raise KeyError(f"Missing myotube obs columns needed for --min-nuclei: {missing_myo}")
    if missing_nuc:
        raise KeyError(f"Missing nuclei obs columns needed for --min-nuclei: {missing_nuc}")

    myo_obs = myotube_adata.obs.copy()
    nuc_obs = nuclei_adata.obs.copy()

    myo_obs["_myotube_match_key"] = build_notebook_match_key(
        myo_obs,
        context_cols=context_cols,
        id_col="local_id",
    )
    myo_obs["myotube_uid"] = (
        myo_obs["Slide Name"].astype(str)
        + "__"
        + myo_obs["Cell Line"].astype(str)
        + "__field"
        + myo_obs["field"].astype(str)
        + "__patch"
        + myo_obs["patch_idx"].astype(str)
        + "__myo"
        + myo_obs["local_id"].astype(str)
        + "__row"
        + np.arange(myo_obs.shape[0]).astype(str)
    )

    duplicated_keys = int(myo_obs["_myotube_match_key"].duplicated().sum())
    if duplicated_keys > 0:
        print(
            f"[scVI] Warning: {duplicated_keys:,} duplicated myotube match keys. Keeping first match.",
            flush=True,
        )

    key_to_uid = (
        myo_obs.drop_duplicates("_myotube_match_key")
        .set_index("_myotube_match_key")["myotube_uid"]
        .to_dict()
    )

    nuc_obs["myotube_id"] = pd.to_numeric(nuc_obs["myotube_id"], errors="coerce")
    is_assigned_by_id = nuc_obs["myotube_id"].notna() & (nuc_obs["myotube_id"] != -1)
    nuc_obs["_nucleus_to_myotube_match_key"] = build_notebook_match_key(
        nuc_obs,
        context_cols=context_cols,
        id_col="myotube_id",
    )

    matched_uid = pd.Series(pd.NA, index=nuc_obs.index, dtype="object")
    matched_uid.loc[is_assigned_by_id] = (
        nuc_obs.loc[is_assigned_by_id, "_nucleus_to_myotube_match_key"].map(key_to_uid)
    )
    counts_by_uid = matched_uid.dropna().value_counts(sort=False)
    counts = myo_obs["myotube_uid"].map(counts_by_uid).fillna(0).to_numpy(dtype=np.int64)
    summary = {
        "nuclei_source": source,
        "nuclei_with_non_unassigned_myotube_id": int(is_assigned_by_id.sum()),
        "nuclei_matched_to_myotube": int(matched_uid.notna().sum()),
        "nuclei_with_id_but_no_match": int((is_assigned_by_id & matched_uid.isna()).sum()),
        "myotubes_total": int(myotube_adata.n_obs),
        "myotubes_with_1_or_more_nuclei": int((counts >= 1).sum()),
        "myotubes_with_2_or_more_nuclei": int((counts >= 2).sum()),
    }
    print("[scVI] Nucleus -> myotube matching:", flush=True)
    for key, value in summary.items():
        print(f"[scVI]   {key}: {value:,}" if isinstance(value, int) else f"[scVI]   {key}: {value}", flush=True)
    return counts, str(source), summary


def apply_min_nuclei_filter(adata, counts, source, min_nuclei):
    if min_nuclei is None or min_nuclei <= 0:
        return adata, None

    counts = counts.astype(np.int64)
    adata.obs["n_nuclei"] = counts
    print(
        "[scVI] Combined myotubes: n_nuclei summary before filter: "
        f"==0={int((counts == 0).sum())}, >=1={int((counts >= 1).sum())}, "
        f">=2={int((counts >= 2).sum())}, >=3={int((counts >= 3).sum())}, "
        f"max={int(counts.max()) if counts.size else 0}",
        flush=True,
    )

    mask = counts >= int(min_nuclei)
    before = adata.n_obs
    adata = subset_obs_with_uns(adata, mask)
    print(
        f"[scVI] Combined myotubes: kept {adata.n_obs}/{before} myotubes "
        f"with n_nuclei >= {min_nuclei} using {source}",
        flush=True,
    )
    return adata, source


def load_h5ads(paths, requested_mode, is_myonucleus, var_join, filter_edge_nuclei=True):
    adatas = []
    modes = []
    summaries = []

    for path in paths:
        adata, mode, summary = read_and_filter_h5ad(
            path,
            requested_mode,
            is_myonucleus,
            filter_edge_nuclei=filter_edge_nuclei,
        )
        adatas.append(adata)
        modes.append(mode)
        summaries.append(summary)

    if len(set(modes)) != 1:
        raise ValueError(f"Mixed input modes are not supported in one model: {modes}")

    if len(adatas) == 1:
        return adatas[0], modes[0], summaries

    keys = make_unique_source_labels(paths)
    print(f"[scVI] Concatenating {len(adatas)} h5ad files with var_join={var_join!r}", flush=True)
    adata = ad.concat(
        adatas,
        axis=0,
        join=var_join,
        label="source_h5ad",
        keys=keys,
        index_unique="-",
        fill_value=0,
    )
    return adata, modes[0], summaries


def load_paired_nuclei_h5ads(myotube_paths, nuclei_h5ad_paths, var_join):
    nuclei_h5ad_paths = nuclei_h5ad_paths or []
    if nuclei_h5ad_paths and len(nuclei_h5ad_paths) not in {1, len(myotube_paths)}:
        raise ValueError("--nuclei-h5ad-paths must contain either one path or one path per input h5ad.")

    paths = []
    for idx, myotube_path in enumerate(myotube_paths):
        if nuclei_h5ad_paths:
            nuclei_path = Path(nuclei_h5ad_paths[idx if len(nuclei_h5ad_paths) > 1 else 0])
        else:
            nuclei_path = infer_nuclei_path(myotube_path)
        if nuclei_path is None or not Path(nuclei_path).exists():
            raise FileNotFoundError(
                "Cannot apply --min-nuclei because no paired nuclei h5ad was found. "
                "Pass --nuclei-h5ad-paths or set --min-nuclei 0."
            )
        paths.append(Path(nuclei_path))

    adatas = []
    for path in paths:
        print(f"[scVI] Loading paired nuclei assignments for --min-nuclei: {path}", flush=True)
        nuclei = ad.read_h5ad(path)
        ensure_slide_name_from_path(nuclei, path)
        if nuclei.var_names.has_duplicates:
            warnings.warn(f"{path.name} has duplicate var_names; making them unique for scVI matching.")
            nuclei.var_names_make_unique()
        if not nuclei.obs_names.is_unique:
            warnings.warn(f"{path.name} has duplicate obs_names; making them unique for scVI matching.")
            nuclei.obs_names_make_unique()
        adatas.append(nuclei)

    if len(adatas) == 1:
        return adatas[0], [str(path) for path in paths]

    keys = make_unique_source_labels(paths)
    print(f"[scVI] Concatenating {len(adatas)} paired nuclei h5ad files with var_join={var_join!r}", flush=True)
    nuclei = ad.concat(
        adatas,
        axis=0,
        join=var_join,
        label="source_h5ad",
        keys=keys,
        index_unique="-",
        fill_value=0,
    )
    return nuclei, [str(path) for path in paths]


def filter_nuclei_edges_for_matching(adata, edge_col="is_edge"):
    if edge_col not in adata.obs.columns:
        return adata, {"edge_col": edge_col, "n_before_edge_filter": int(adata.n_obs), "n_after_edge_filter": int(adata.n_obs)}

    keep = ~adata.obs[edge_col].astype(bool).to_numpy()
    before = adata.n_obs
    adata = subset_obs_with_uns(adata, keep)
    print(f"[scVI] Paired nuclei: kept {adata.n_obs}/{before} nuclei after excluding edge objects", flush=True)
    return adata, {"edge_col": edge_col, "n_before_edge_filter": int(before), "n_after_edge_filter": int(adata.n_obs)}


def apply_standard_filters(adata, counts_layer, min_counts, min_cells, control_probe_pattern):
    summary = {
        "n_obs_before_standard_filters": int(adata.n_obs),
        "n_vars_before_standard_filters": int(adata.n_vars),
        "min_counts": None if min_counts is None else float(min_counts),
        "min_cells": None if min_cells is None else int(min_cells),
        "control_probe_pattern": control_probe_pattern,
        "n_control_probes_removed": 0,
        "control_probes_removed": [],
    }

    if min_counts is not None and min_counts > 0:
        counts = row_sums(get_matrix(adata, counts_layer))
        mask = counts >= float(min_counts)
        before = adata.n_obs
        adata = subset_obs_with_uns(adata, mask)
        print(f"[scVI] Kept {adata.n_obs}/{before} objects with total counts >= {min_counts}", flush=True)

    if min_cells is not None and min_cells > 0:
        nonzero = col_nonzero_counts(get_matrix(adata, counts_layer))
        mask = nonzero >= int(min_cells)
        before = adata.n_vars
        adata = adata[:, mask].copy()
        print(f"[scVI] Kept {adata.n_vars}/{before} genes expressed in >= {min_cells} objects", flush=True)

    if control_probe_pattern and str(control_probe_pattern).lower() not in {"none", "null", "na"}:
        var_names = pd.Index(adata.var_names.astype(str))
        control_mask = np.asarray(
            var_names.str.contains(str(control_probe_pattern), case=False, regex=True, na=False),
            dtype=bool,
        )
        removed = var_names[control_mask].tolist()
        if removed:
            adata = adata[:, ~control_mask].copy()
        summary["n_control_probes_removed"] = len(removed)
        summary["control_probes_removed"] = removed
        print(
            f"[scVI] Removed {len(removed)} control probes matching {control_probe_pattern!r}",
            flush=True,
        )

    summary["n_obs_after_standard_filters"] = int(adata.n_obs)
    summary["n_vars_after_standard_filters"] = int(adata.n_vars)
    return adata, summary


def validate_count_matrix(adata, counts_layer):
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Filtered training matrix is empty: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    matrix = get_matrix(adata, counts_layer)
    total_counts = row_sums(matrix)
    total_min = float(total_counts.min()) if total_counts.size else 0.0
    total_median = float(np.median(total_counts)) if total_counts.size else 0.0
    total_max = float(total_counts.max()) if total_counts.size else 0.0
    if hasattr(matrix, "getnnz") and hasattr(matrix, "data"):
        values = np.asarray(matrix.data)
        nnz = int(values.size)
        entry_min = float(values.min()) if nnz else 0.0
        entry_max = float(values.max()) if nnz else 0.0
        finite = np.isfinite(values).all() if nnz else True
        negative = bool((values < 0).any()) if nnz else False
        integer_like = bool(np.allclose(values, np.round(values), rtol=0.0, atol=1e-6)) if nnz else True
    else:
        values = np.asarray(matrix)
        nnz = int(np.count_nonzero(values))
        nonzero_values = values[values != 0]
        entry_min = float(np.nanmin(nonzero_values)) if nonzero_values.size else 0.0
        entry_max = float(np.nanmax(nonzero_values)) if nonzero_values.size else 0.0
        finite = bool(np.isfinite(values).all())
        negative = bool((values < 0).any())
        integer_like = bool(np.allclose(values, np.round(values), rtol=0.0, atol=1e-6))

    print(
        "[scVI] Count matrix validation: "
        f"dtype={getattr(matrix, 'dtype', None)}, nnz={nnz}, "
        f"nonzero_entry_min={entry_min:g}, nonzero_entry_max={entry_max:g}, "
        f"row_total_min={total_min:g}, row_total_median={total_median:g}, row_total_max={total_max:g}",
        flush=True,
    )
    if not finite:
        raise ValueError("Count matrix contains NaN or infinite values after filtering.")
    if negative:
        raise ValueError("Count matrix contains negative values after filtering; scVI expects raw counts.")
    if not integer_like:
        warnings.warn("Count matrix contains non-integer values; scVI expects raw count-like input.")


def cast_count_matrix_to_float32(adata, counts_layer):
    matrix = get_matrix(adata, counts_layer)
    if getattr(matrix, "dtype", None) == np.dtype("float32"):
        return

    casted = matrix.astype(np.float32)
    if counts_layer is None:
        adata.X = casted
    else:
        adata.layers[counts_layer] = casted
    print("[scVI] Cast count matrix to float32 before scVI setup", flush=True)


def fit_or_apply_continuous_transform(values, covariate, transform, params=None):
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().all():
        values = values.fillna(0.0)
    else:
        values = values.fillna(float(values.median()))

    arr = values.to_numpy(dtype=np.float64)
    raw_min = float(np.min(arr)) if arr.size else 0.0
    raw_median = float(np.median(arr)) if arr.size else 0.0
    raw_max = float(np.max(arr)) if arr.size else 0.0

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
    else:
        mean = 0.0
        std = 1.0

    out_params = {
        "transform": used_transform,
        "mean": float(mean),
        "std": float(std),
        "raw_min": raw_min,
        "raw_median": raw_median,
        "raw_max": raw_max,
    }
    return arr.astype(np.float32), out_params


def prepare_covariates(
    adata,
    covariates,
    counts_layer,
    continuous_covariate_transform="log1p-zscore",
    covariate_transform_params=None,
):
    continuous_keys = []
    categorical_keys = []
    fitted_params = {}
    covariate_transform_params = covariate_transform_params or {}

    for covariate in covariates:
        if covariate == "n_counts":
            adata.obs[covariate] = row_sums(get_matrix(adata, counts_layer)).astype(np.float32)

        if covariate not in adata.obs.columns:
            raise KeyError(f"Requested covariate {covariate!r} is missing from adata.obs")

        series = adata.obs[covariate]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            values, params = fit_or_apply_continuous_transform(
                series,
                covariate=covariate,
                transform=continuous_covariate_transform,
                params=covariate_transform_params.get(covariate),
            )
            adata.obs[covariate] = values
            fitted_params[covariate] = params
            print(
                f"[scVI] Continuous covariate {covariate!r}: "
                f"raw_min={params['raw_min']:.4g}, raw_median={params['raw_median']:.4g}, "
                f"raw_max={params['raw_max']:.4g}, transform={params['transform']}, "
                f"mean={params['mean']:.4g}, std={params['std']:.4g}",
                flush=True,
            )
            continuous_keys.append(covariate)
        else:
            values = series.astype("string").fillna("missing").astype("category")
            if len(values.cat.categories) <= 1:
                warnings.warn(f"obs[{covariate!r}] has one level; ignoring it as a covariate.")
                continue
            adata.obs[covariate] = values
            categorical_keys.append(covariate)

    return continuous_keys, categorical_keys, fitted_params


def configure_runtime(args):
    scvi.settings.seed = args.seed
    if args.matmul_precision in {"high", "medium"}:
        torch.set_float32_matmul_precision(args.matmul_precision)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    for attr, value in {
        "dl_num_workers": args.num_workers,
        "dl_pin_memory": bool(args.pin_memory),
        "dl_persistent_workers": bool(args.persistent_workers),
        "dl_prefetch_factor": args.prefetch_factor,
    }.items():
        if value is not None and hasattr(scvi.settings, attr):
            try:
                setattr(scvi.settings, attr, value)
            except Exception:
                pass


def cuda_safe_lgamma_positive(value):
    """Differentiable log-gamma approximation for positive CUDA tensors.

    Some PyTorch/CUDA builds can fail inside the jitted CUDA lgamma kernel on
    otherwise valid tensors. scVI's NB/ZINB losses only call lgamma on positive
    values, so a recurrence to z >= 8 plus a Stirling series is a practical
    CUDA-only fallback that keeps gradients on GPU.
    """
    if not torch.is_tensor(value) or value.device.type != "cuda":
        return torch.lgamma(value)

    original_dtype = value.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    z = value.to(work_dtype)
    z = torch.clamp(z, min=torch.finfo(work_dtype).tiny)
    result = torch.zeros_like(z)

    for _ in range(8):
        mask = z < 8.0
        result = torch.where(mask, result - torch.log(z), result)
        z = torch.where(mask, z + 1.0, z)

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    correction = inv_z * (
        (1.0 / 12.0)
        + inv_z2
        * (
            (-1.0 / 360.0)
            + inv_z2
            * (
                (1.0 / 1260.0)
                + inv_z2
                * (
                    (-1.0 / 1680.0)
                    + inv_z2 * ((1.0 / 1188.0) + inv_z2 * (-691.0 / 360360.0))
                )
            )
        )
    )
    out = result + (z - 0.5) * torch.log(z) - z + 0.9189385332046727 + correction
    if original_dtype in {torch.float16, torch.bfloat16}:
        return out.to(original_dtype)
    return out


def cuda_lgamma_smoke_test():
    if not torch.cuda.is_available():
        return True

    try:
        values = torch.tensor([0.0, 0.25, 1.0, 2.0, 10.0], device="cuda", dtype=torch.float32)
        out = torch.lgamma(values)
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all().item())
    except Exception as exc:
        print(f"[scVI] Native CUDA torch.lgamma smoke test failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False


def install_cuda_safe_lgamma_patch():
    if getattr(scvi_nb, "_myonis_cuda_safe_lgamma_patch", False):
        return

    def negative_binomial_log_prob(self, value):
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                    stacklevel=scvi_nb.settings.warnings_stacklevel,
                )

        lgamma_fn = scvi_nb.torch_lgamma_mps if self.on_mps else cuda_safe_lgamma_positive
        return scvi_nb.log_nb_positive(
            value,
            mu=self.mu,
            theta=self.theta,
            eps=self._eps,
            lgamma_fn=lgamma_fn,
        )

    def zero_inflated_negative_binomial_log_prob(self, value):
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
                stacklevel=scvi_nb.settings.warnings_stacklevel,
            )

        lgamma_fn = scvi_nb.torch_lgamma_mps if self.on_mps else cuda_safe_lgamma_positive
        return scvi_nb.log_zinb_positive(
            value,
            self.mu,
            self.theta,
            self.zi_logits,
            eps=1e-08,
            lgamma_fn=lgamma_fn,
        )

    def negative_binomial_mixture_log_prob(self, value):
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
                stacklevel=scvi_nb.settings.warnings_stacklevel,
            )

        lgamma_fn = scvi_nb.torch_lgamma_mps if self.on_mps else cuda_safe_lgamma_positive
        return scvi_nb.log_mixture_nb(
            value,
            self.mu1,
            self.mu2,
            self.theta1,
            self.theta2,
            self.mixture_logits,
            eps=1e-08,
            lgamma_fn=lgamma_fn,
        )

    scvi_nb.NegativeBinomial.log_prob = negative_binomial_log_prob
    scvi_nb.ZeroInflatedNegativeBinomial.log_prob = zero_inflated_negative_binomial_log_prob
    scvi_nb.NegativeBinomialMixture.log_prob = negative_binomial_mixture_log_prob
    scvi_nb._myonis_cuda_safe_lgamma_patch = True
    print("[scVI] Installed CUDA-safe lgamma fallback for scVI NB/ZINB log-probabilities", flush=True)


def configure_cuda_lgamma(mode):
    if mode == "stirling":
        install_cuda_safe_lgamma_patch()
        return "stirling"

    if mode == "torch":
        if torch.cuda.is_available() and not cuda_lgamma_smoke_test():
            raise RuntimeError(
                "Native CUDA torch.lgamma failed, and --cuda-lgamma-mode=torch was requested. "
                "Use --cuda-lgamma-mode auto or stirling, or install a PyTorch build with a working CUDA lgamma kernel."
            )
        print("[scVI] Native CUDA torch.lgamma is enabled", flush=True)
        return "torch"

    if torch.cuda.is_available() and not cuda_lgamma_smoke_test():
        install_cuda_safe_lgamma_patch()
        return "stirling"

    print("[scVI] Native CUDA torch.lgamma smoke test passed", flush=True)
    return "torch"


def check_gpu_or_raise():
    cuda_version = getattr(torch.version, "cuda", None)
    device_count = torch.cuda.device_count()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        "[scVI] GPU requested. "
        f"torch={torch.__version__}, torch_cuda={cuda_version}, "
        f"cuda_available={torch.cuda.is_available()}, device_count={device_count}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}",
        flush=True,
    )
    if not torch.cuda.is_available() or device_count < 1:
        raise RuntimeError(
            "GPU training was requested with --use-gpu, but PyTorch cannot see a CUDA GPU. "
            "Check that the job is running inside a GPU Slurm allocation and that the "
            "environment has a CUDA-enabled PyTorch build. Run: "
            "python -c \"import torch; print(torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available(), torch.cuda.device_count())\""
        )


def write_history(model, outdir):
    history = getattr(model, "history", None)
    if not history:
        return

    history_dir = outdir / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    for key, value in history.items():
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key)).strip("_") or "metric"
        path = history_dir / f"{safe_key}.csv"
        if isinstance(value, pd.DataFrame):
            value.to_csv(path, index=True)
        else:
            pd.DataFrame({safe_key: np.asarray(value).reshape(-1)}).to_csv(path, index=False)
    print(f"[scVI] Training history saved under {history_dir}", flush=True)


def utc_timestamp(timestamp):
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def read_metric_series(path, metric_name):
    path = Path(path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if metric_name not in df.columns:
        return None

    values = pd.to_numeric(df[metric_name], errors="coerce")
    if values.notna().sum() == 0:
        return None

    epoch = pd.to_numeric(df["epoch"], errors="coerce") if "epoch" in df.columns else pd.Series(np.arange(len(df)))
    return pd.DataFrame({"epoch": epoch, metric_name: values}).dropna(subset=[metric_name])


def summarize_metric(path, metric_name, prefix):
    series = read_metric_series(path, metric_name)
    if series is None or series.empty:
        return {}

    values = series[metric_name].to_numpy(dtype=float)
    epochs = series["epoch"].to_numpy(dtype=float)
    min_idx = int(np.nanargmin(values))
    return {
        f"final_{prefix}": float(values[-1]),
        f"min_{prefix}": float(values[min_idx]),
        f"min_{prefix}_epoch": float(epochs[min_idx]),
    }


def collect_training_metric_summary(outdir):
    history_dir = Path(outdir) / "training_history"
    summary = {}

    train_metrics = {
        "train_loss": "train_loss",
        "elbo_train": "elbo_train",
        "reconstruction_loss_train": "reconstruction_loss_train",
        "kl_local_train": "kl_local_train",
    }
    for filename, metric_name in train_metrics.items():
        summary.update(summarize_metric(history_dir / f"{filename}.csv", metric_name, metric_name))

    validation_path = history_dir / "validation_metrics.csv"
    validation_metrics = [
        "validation_loss",
        "elbo_validation",
        "reconstruction_loss_validation",
        "kl_local_validation",
    ]
    for metric_name in validation_metrics:
        summary.update(summarize_metric(validation_path, metric_name, metric_name))

    train_loss = read_metric_series(history_dir / "train_loss.csv", "train_loss")
    validation_loss = read_metric_series(validation_path, "validation_loss")
    if train_loss is not None and validation_loss is not None and not validation_loss.empty:
        best_val_idx = validation_loss["validation_loss"].astype(float).idxmin()
        best_epoch = validation_loss.loc[best_val_idx, "epoch"]
        train_at_epoch = train_loss.loc[train_loss["epoch"] == best_epoch, "train_loss"]
        if not train_at_epoch.empty:
            summary["train_validation_loss_gap_at_best_validation"] = float(
                validation_loss.loc[best_val_idx, "validation_loss"] - train_at_epoch.iloc[0]
            )

    return summary


def swept_hyperparameters(args):
    return {
        "n_latent": int(args.n_latent),
        "n_hidden": int(args.n_hidden),
        "n_layers": int(args.n_layers),
        "gene_likelihood": args.gene_likelihood,
    }


def sweep_metadata(args):
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return {
        "sweep_name": args.sweep_name,
        "run_id": args.run_id,
        "manifest_path": args.manifest_path,
        "SLURM_JOB_ID": slurm_job_id,
        "SLURM_ARRAY_TASK_ID": slurm_array_task_id,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "swept_hyperparameters": swept_hyperparameters(args),
    }


def write_sweep_run_summary(args, outdir, status, start_time, inference_output_path=None, error=None):
    end_time = time.time()
    outdir = Path(outdir)
    summary = {
        "status": status,
        "error": error,
        "start_time_utc": utc_timestamp(start_time),
        "end_time_utc": utc_timestamp(end_time),
        "runtime_seconds": float(end_time - start_time),
        "model_dir": str(outdir),
        "config_path": str(outdir / CONFIG_NAME),
        "history_dir": str(outdir / "training_history"),
        "latent_h5ad_path": str(inference_output_path) if inference_output_path is not None else None,
        "n_latent": int(args.n_latent),
        "n_hidden": int(args.n_hidden),
        "n_layers": int(args.n_layers),
        "gene_likelihood": args.gene_likelihood,
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "precision": args.precision,
    }
    summary.update(sweep_metadata(args))
    summary.update(collect_training_metric_summary(outdir))

    path = outdir / "sweep_run_summary.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(summary), handle, indent=2, sort_keys=True)
    print(f"[scVI] Sweep run summary saved to {path}", flush=True)
    return path


def save_run_snapshot(outdir):
    snapshot_dir = outdir / "run_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    script_snapshot = snapshot_dir / script_path.name
    shutil.copy2(script_path, script_snapshot)

    shell_path = script_path.with_suffix(".sh")
    shell_snapshot = None
    if shell_path.exists():
        shell_snapshot = snapshot_dir / shell_path.name
        shutil.copy2(shell_path, shell_snapshot)

    launcher_snapshot = None
    launcher_path = strip_control_chars(os.environ.get("SCVI_LAUNCHER_SCRIPT"), "SCVI_LAUNCHER_SCRIPT")
    if launcher_path and Path(launcher_path).exists():
        launcher_path = Path(launcher_path).resolve()
        launcher_snapshot = snapshot_dir / launcher_path.name
        if launcher_snapshot != shell_snapshot:
            shutil.copy2(launcher_path, launcher_snapshot)

    manifest_snapshot = None
    manifest_path = strip_control_chars(os.environ.get("SCVI_SWEEP_MANIFEST"), "SCVI_SWEEP_MANIFEST")
    if manifest_path and Path(manifest_path).exists():
        manifest_path = Path(manifest_path).resolve()
        manifest_snapshot = snapshot_dir / manifest_path.name
        shutil.copy2(manifest_path, manifest_snapshot)

    command_path = snapshot_dir / "command.txt"
    command_path.write_text(" ".join(map(str, sys.argv)) + "\n", encoding="utf-8")

    print(f"[scVI] Run snapshot saved to {snapshot_dir}", flush=True)
    return {
        "snapshot_dir": str(snapshot_dir),
        "scvi_main_py": str(script_snapshot),
        "scvi_main_sh": str(shell_snapshot) if shell_snapshot is not None else None,
        "launcher_script": str(launcher_snapshot) if launcher_snapshot is not None else None,
        "sweep_manifest": str(manifest_snapshot) if manifest_snapshot is not None else None,
        "command": str(command_path),
    }


def copy_obsm(adata):
    copied = {}
    for key in adata.obsm.keys():
        value = adata.obsm[key]
        try:
            copied[key] = value.copy()
        except AttributeError:
            copied[key] = copy.deepcopy(value)
    return copied


def make_json_safe(value):
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


def build_latent_obs(adata, counts_layer, covariates):
    obs = adata.obs.copy()
    total_counts = row_sums(get_matrix(adata, counts_layer)).astype(np.float32)
    obs["total_counts"] = total_counts
    if "n_counts" in covariates and "n_counts" not in obs.columns:
        obs["n_counts"] = total_counts
    return obs


def resolve_inference_output_path(outdir, inference_output_path):
    if inference_output_path:
        path = Path(strip_control_chars(inference_output_path, "--inference-output-path"))
    else:
        path = Path(outdir) / f"{Path(outdir).name}.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_training_latent_h5ad(
    model,
    adata,
    latent_obs,
    latent_obsm,
    outdir,
    args,
    input_mode,
    continuous_keys,
    categorical_keys,
    preprocessing_summary,
):
    out_path = resolve_inference_output_path(outdir, args.inference_output_path)
    if not latent_obs.index.equals(adata.obs_names):
        raise ValueError("Latent output obs index no longer matches the training AnnData row order.")

    print("[scVI] Encoding training data to latent space", flush=True)
    latent = model.get_latent_representation(adata=adata)
    latent = np.asarray(latent, dtype=np.float32)
    latent_var = pd.DataFrame(
        index=pd.Index([f"scvi_latent_{i}" for i in range(latent.shape[1])], name="latent_dimension")
    )
    latent_adata = ad.AnnData(X=latent, obs=latent_obs.copy(), var=latent_var)

    for key, value in latent_obsm.items():
        latent_adata.obsm[key] = value

    latent_adata.uns["scvi_latent"] = {
        "model_dir": str(Path(outdir)),
        "source_h5ad": [str(Path(path)) for path in args.h5ad_paths],
        "input_mode": input_mode,
        "is_myonucleus": bool(args.is_myonucleus),
        "covariates_to_remove": flatten_covariates(args.covariates_to_remove),
        "continuous_covariate_keys": continuous_keys,
        "categorical_covariate_keys": categorical_keys,
        "continuous_covariate_transform": args.continuous_covariate_transform,
        "counts_layer": "" if args.counts_layer is None else str(args.counts_layer),
        "n_latent": int(latent.shape[1]),
        "var_join": args.var_join,
        "min_counts": args.min_counts,
        "min_cells": args.min_cells,
        "control_probe_pattern": args.control_probe_pattern,
        "min_nuclei": args.min_nuclei,
        "preprocessing_summary_format": "json",
        "preprocessing_summary_json": json.dumps(
            make_json_safe(preprocessing_summary),
            indent=2,
            sort_keys=True,
        ),
    }

    print(f"[scVI] Saving post-training latent h5ad to {out_path}", flush=True)
    latent_adata.write_h5ad(out_path)
    print(f"[scVI] Done. Latent shape: {latent_adata.shape}", flush=True)
    return out_path


def main():
    args = sanitize_runtime_args(parse_args())
    start_time = time.time()
    outdir = Path(args.model_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    configure_runtime(args)
    cuda_lgamma_backend = "cpu"
    if args.use_gpu:
        check_gpu_or_raise()
        cuda_lgamma_backend = configure_cuda_lgamma(args.cuda_lgamma_mode)

    covariates = flatten_covariates(args.covariates_to_remove)
    adata, input_mode, summaries = load_h5ads(
        args.h5ad_paths,
        requested_mode=args.input_mode,
        is_myonucleus=args.is_myonucleus,
        var_join=args.var_join,
        filter_edge_nuclei=args.filter_edge_nuclei,
    )
    adata, filter_summary = apply_standard_filters(
        adata,
        counts_layer=args.counts_layer,
        min_counts=args.min_counts,
        min_cells=args.min_cells,
        control_probe_pattern=args.control_probe_pattern,
    )

    nuclei_sources = []
    nuclei_filter_summary = None
    nuclei_edge_summary = None
    myotube_assignment_summary = None
    if input_mode == "myotubes" and args.min_nuclei and args.min_nuclei > 0:
        nuclei_adata, nuclei_sources = load_paired_nuclei_h5ads(
            args.h5ad_paths,
            args.nuclei_h5ad_paths,
            args.var_join,
        )
        nuclei_adata, nuclei_edge_summary = filter_nuclei_edges_for_matching(nuclei_adata)
        nuclei_adata, nuclei_filter_summary = apply_standard_filters(
            nuclei_adata,
            counts_layer=args.counts_layer,
            min_counts=args.min_counts,
            min_cells=args.min_cells,
            control_probe_pattern=args.control_probe_pattern,
        )
        nuclei_counts, nuclei_source, myotube_assignment_summary = compute_myotube_nucleus_counts_from_nuclei(
            adata,
            nuclei_adata,
            ", ".join(nuclei_sources),
        )
        adata, _ = apply_min_nuclei_filter(adata, nuclei_counts, nuclei_source, args.min_nuclei)

    if not args.skip_count_validation:
        validate_count_matrix(adata, args.counts_layer)
    if not args.no_cast_counts_float32:
        cast_count_matrix_to_float32(adata, args.counts_layer)

    latent_obs = build_latent_obs(adata, args.counts_layer, covariates)
    latent_obsm = copy_obsm(adata)

    continuous_keys, categorical_keys, covariate_transform_params = prepare_covariates(
        adata,
        covariates,
        args.counts_layer,
        continuous_covariate_transform=args.continuous_covariate_transform,
    )

    print(
        f"[scVI] Training matrix: n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
        f"mode={input_mode}, continuous_covariates={continuous_keys}, "
        f"categorical_covariates={categorical_keys}, gene_likelihood={args.gene_likelihood}, "
        f"cuda_lgamma_backend={cuda_lgamma_backend}",
        flush=True,
    )

    SCVI.setup_anndata(
        adata,
        layer=args.counts_layer,
        continuous_covariate_keys=continuous_keys or None,
        categorical_covariate_keys=categorical_keys or None,
    )

    model = SCVI(
        adata,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        gene_likelihood=args.gene_likelihood,
        dispersion="gene",
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
        dropout_rate=args.dropout_rate,
    )

    train_kwargs = {
        "max_epochs": args.max_epochs,
        "plan_kwargs": {"weight_decay": args.weight_decay, "lr": args.lr},
        "accelerator": "gpu" if args.use_gpu else "cpu",
        "devices": "auto" if args.use_gpu else 1,
        "precision": args.precision,
        "validation_size": args.validation_size,
    }
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size
    if args.validation_size and args.validation_size > 0:
        train_kwargs["callbacks"] = [
            ValidationLossPrinter(
                every_n_epochs=args.validation_loss_print_interval,
                output_path=outdir / "training_history" / "validation_metrics.csv",
                print_metrics=not args.no_print_validation_loss,
            )
        ]

    outdir.mkdir(parents=True, exist_ok=True)
    run_snapshot = save_run_snapshot(outdir)

    model.train(**train_kwargs)

    model.save(outdir, overwrite=True, save_anndata=not args.no_save_anndata)
    print(f"[scVI] Model saved to {outdir.resolve()}", flush=True)

    write_history(model, outdir)

    preprocessing_summary = {
        "input_summaries": summaries,
        "filter_summary": filter_summary,
        "nuclei_sources": nuclei_sources,
        "nuclei_edge_summary": nuclei_edge_summary,
        "nuclei_filter_summary": nuclei_filter_summary,
        "myotube_assignment_summary": myotube_assignment_summary,
    }
    inference_output_path = None
    if args.inference:
        inference_output_path = write_training_latent_h5ad(
            model=model,
            adata=adata,
            latent_obs=latent_obs,
            latent_obsm=latent_obsm,
            outdir=outdir,
            args=args,
            input_mode=input_mode,
            continuous_keys=continuous_keys,
            categorical_keys=categorical_keys,
            preprocessing_summary=preprocessing_summary,
        )

    config = {
        **sweep_metadata(args),
        "h5ad_paths": [str(Path(path)) for path in args.h5ad_paths],
        "input_mode": input_mode,
        "is_myonucleus": bool(args.is_myonucleus),
        "counts_layer": args.counts_layer,
        "covariates_to_remove": covariates,
        "continuous_covariate_keys": continuous_keys,
        "categorical_covariate_keys": categorical_keys,
        "continuous_covariate_transform": args.continuous_covariate_transform,
        "covariate_transform_params": covariate_transform_params,
        "var_join": args.var_join,
        "min_counts": args.min_counts,
        "min_cells": args.min_cells,
        "control_probe_pattern": args.control_probe_pattern,
        "min_nuclei": args.min_nuclei,
        "filter_edge_nuclei": bool(args.filter_edge_nuclei),
        "gene_likelihood": args.gene_likelihood,
        "cuda_lgamma_mode": args.cuda_lgamma_mode,
        "cuda_lgamma_backend": cuda_lgamma_backend,
        "validation_size": args.validation_size,
        "validation_loss_print_interval": args.validation_loss_print_interval,
        "print_validation_loss": not args.no_print_validation_loss,
        "inference": bool(args.inference),
        "inference_output_path": str(inference_output_path) if inference_output_path is not None else None,
        "run_snapshot": run_snapshot,
        "nuclei_h5ad_paths": nuclei_sources or [str(Path(path)) for path in (args.nuclei_h5ad_paths or [])],
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_latent": int(args.n_latent),
        "n_hidden": int(args.n_hidden),
        "n_layers": int(args.n_layers),
        "dropout_rate": float(args.dropout_rate),
        "max_epochs": int(args.max_epochs),
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "precision": args.precision,
        "input_summaries": summaries,
        "filter_summary": filter_summary,
        "nuclei_edge_summary": nuclei_edge_summary,
        "nuclei_filter_summary": nuclei_filter_summary,
        "myotube_assignment_summary": myotube_assignment_summary,
    }
    with open(outdir / CONFIG_NAME, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"[scVI] Config saved to {outdir / CONFIG_NAME}", flush=True)

    write_sweep_run_summary(
        args,
        outdir,
        status="completed",
        start_time=start_time,
        inference_output_path=inference_output_path,
    )


if __name__ == "__main__":
    main()
