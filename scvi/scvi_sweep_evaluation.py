#!/usr/bin/env python3
"""Evaluate scVI sweep run folders and write sweep-level metric tables."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


CONFIG_NAME = "myonis_scvi_config.json"
SUMMARY_NAME = "sweep_run_summary.json"
MASTER_NAME = "sweep_evaluation_master.csv"

DEFECT_SCORE_PRIORITY = [
    "nuclear_aberration_assigned_nuclei_total_per_nuclear_area",
    "nuclear_aberration_mean_sigmoid",
    "defect_score",
]
RIDGE_COVARIATES = [
    "n_counts",
    "total_counts",
    "total_counts_for_plot",
    "area_px2",
    "n_nuclei",
]
CELL_LINE_COLUMNS = ["Cell Line", "cell_line", "cell_line_label"]
SLIDE_COLUMNS = ["Slide Name", "slide", "slide_name"]
BASE_COLUMNS = [
    "run_id",
    "run_name",
    "model_dir",
    "sweep_name",
    "input_mode",
    "n_latent",
    "n_hidden",
    "n_layers",
    "gene_likelihood",
    "seed",
    "max_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "precision",
    "n_obs",
    "n_vars",
    "min_counts",
    "min_cells",
    "min_nuclei",
    "filter_edge_nuclei",
    "covariates_to_remove",
    "status",
    "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate completed scVI sweep runs.")
    parser.add_argument("--sweep-dir", required=True, help="Sweep folder containing run_* directories.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation outputs. Default: <sweep-dir>/evaluation.",
    )
    parser.add_argument(
        "--defect-score-col",
        default="auto",
        help="Obs column to use for defect score, or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--cell-lines",
        default=None,
        help="Optional comma-separated Cell Line values to keep before classifier/regression evaluations.",
    )
    parser.add_argument("--n-qbins", type=int, default=8)
    parser.add_argument("--low-bin", type=int, default=0)
    parser.add_argument("--high-bin", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--classification-myonuclei-only",
        action="store_true",
        help=(
            "For nuclei latent h5ad files, restrict defect q-binning and classification "
            "to obs['myotube_id'] >= 0 before computing AUROC/F1/etc."
        ),
    )
    return parser.parse_args()


def split_csv_values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    return values or None


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        warnings.warn(f"Could not read JSON {path}: {exc}")
        return {}


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value or "value"


def first_present(mapping_list: list[dict], key: str, default=None):
    for mapping in mapping_list:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def serialize_cell_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(make_json_safe(value), sort_keys=True)
    return value


def discover_run_dirs(sweep_dir: Path) -> list[Path]:
    run_dirs = [path for path in sweep_dir.iterdir() if path.is_dir() and path.name != "evaluation"]

    def key(path: Path):
        match = re.search(r"run_(\d+)", path.name)
        return (int(match.group(1)) if match else math.inf, path.name)

    return sorted(run_dirs, key=key)


def run_descriptors(run_dir: Path, config: dict, summary: dict) -> dict:
    sources = [summary, config]
    descriptors = {
        "run_id": first_present(sources, "run_id", run_dir.name),
        "run_name": run_dir.name,
        "model_dir": first_present(sources, "model_dir", str(run_dir)),
        "sweep_name": first_present(sources, "sweep_name", run_dir.parent.name),
        "input_mode": first_present(sources, "input_mode"),
        "n_latent": first_present(sources, "n_latent"),
        "n_hidden": first_present(sources, "n_hidden"),
        "n_layers": first_present(sources, "n_layers"),
        "gene_likelihood": first_present(sources, "gene_likelihood"),
        "seed": first_present(sources, "seed"),
        "max_epochs": first_present(sources, "max_epochs"),
        "batch_size": first_present(sources, "batch_size"),
        "lr": first_present(sources, "lr"),
        "weight_decay": first_present(sources, "weight_decay"),
        "precision": first_present(sources, "precision"),
        "n_obs": first_present(sources, "n_obs"),
        "n_vars": first_present(sources, "n_vars"),
        "min_counts": first_present(sources, "min_counts"),
        "min_cells": first_present(sources, "min_cells"),
        "min_nuclei": first_present(sources, "min_nuclei"),
        "filter_edge_nuclei": first_present(sources, "filter_edge_nuclei"),
        "covariates_to_remove": first_present(sources, "covariates_to_remove"),
        "status": first_present(sources, "status"),
        "runtime_seconds": first_present(sources, "runtime_seconds"),
    }
    return {key: serialize_cell_value(value) for key, value in descriptors.items()}


def resolve_latent_path(run_dir: Path, config: dict, summary: dict) -> Path | None:
    candidates = []
    for value in (
        config.get("inference_output_path"),
        summary.get("latent_h5ad_path"),
        run_dir / f"{run_dir.name}.h5ad",
    ):
        if value:
            candidates.append(Path(value))

    candidates.extend(sorted(path for path in run_dir.glob("*.h5ad") if path.name != "adata.h5ad"))
    candidates.extend(sorted(run_dir.glob("*.h5ad")))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def read_metric_series(path: Path, metric_name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    if metric_name in df.columns:
        value_col = metric_name
    else:
        numeric_cols = [
            col
            for col in df.columns
            if col != "epoch" and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0
        ]
        if not numeric_cols:
            return None
        value_col = numeric_cols[-1]

    if "epoch" in df.columns:
        epoch = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        first_col = df.columns[0]
        first_numeric = pd.to_numeric(df[first_col], errors="coerce")
        if first_numeric.notna().sum() == len(df):
            epoch = first_numeric
        else:
            epoch = pd.Series(np.arange(len(df)), index=df.index)

    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.DataFrame({"epoch": epoch, metric_name: values}).dropna(subset=[metric_name])
    return out if not out.empty else None


def metric_summary(series: pd.DataFrame, metric_name: str) -> dict:
    values = series[metric_name].to_numpy(dtype=float)
    epochs = series["epoch"].to_numpy(dtype=float)
    min_idx = int(np.nanargmin(values))
    return {
        "n_epochs": int(len(values)),
        "final_value": float(values[-1]),
        "min_value": float(values[min_idx]),
        "min_epoch": float(epochs[min_idx]),
    }


def collect_loss_metrics(run_dir: Path, base: dict) -> tuple[list[dict], dict]:
    history_dir = run_dir / "training_history"
    rows = []
    wide = {}

    metric_specs = [
        ("train_loss.csv", "train_loss", "train"),
        ("elbo_train.csv", "elbo_train", "train"),
        ("reconstruction_loss_train.csv", "reconstruction_loss_train", "train"),
        ("kl_local_train.csv", "kl_local_train", "train"),
        ("kl_global_train.csv", "kl_global_train", "train"),
        ("validation_metrics.csv", "validation_loss", "validation"),
        ("validation_metrics.csv", "elbo_validation", "validation"),
        ("validation_metrics.csv", "reconstruction_loss_validation", "validation"),
        ("validation_metrics.csv", "kl_local_validation", "validation"),
    ]

    series_by_metric = {}
    for filename, metric_name, split in metric_specs:
        series = read_metric_series(history_dir / filename, metric_name)
        if series is None:
            continue
        series_by_metric[metric_name] = series
        summary = metric_summary(series, metric_name)
        rows.append(
            {
                **base,
                "split": split,
                "metric_name": metric_name,
                **summary,
            }
        )
        wide[f"final_{metric_name}"] = summary["final_value"]
        wide[f"min_{metric_name}"] = summary["min_value"]
        wide[f"min_{metric_name}_epoch"] = summary["min_epoch"]
        wide[f"n_epochs_{metric_name}"] = summary["n_epochs"]

    train_loss = series_by_metric.get("train_loss")
    validation_loss = series_by_metric.get("validation_loss")
    if train_loss is not None and validation_loss is not None:
        best_idx = validation_loss["validation_loss"].astype(float).idxmin()
        best_epoch = validation_loss.loc[best_idx, "epoch"]
        train_at_epoch = train_loss.loc[train_loss["epoch"] == best_epoch, "train_loss"]
        if not train_at_epoch.empty:
            wide["train_validation_loss_gap_at_best_validation"] = float(
                validation_loss.loc[best_idx, "validation_loss"] - train_at_epoch.iloc[0]
            )

    return rows, wide


def as_dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def finite_numeric_series(obs: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(obs[col], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def safe_pearson(y_true, y_pred) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred).statistic)


def safe_spearman(y_true, y_pred) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred).statistic)


def classifier_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def find_first_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in obs.columns:
            return col
    return None


def apply_cell_line_filter(adata: ad.AnnData, cell_lines: list[str] | None) -> tuple[ad.AnnData, dict]:
    if not cell_lines:
        return adata, {"cell_line_filter_col": None, "cell_line_filter_values": None}

    col = find_first_column(adata.obs, CELL_LINE_COLUMNS)
    if col is None:
        raise ValueError(f"Requested --cell-lines but no cell-line column found among {CELL_LINE_COLUMNS}")

    keep = adata.obs[col].astype(str).isin(cell_lines).to_numpy()
    if keep.sum() == 0:
        raise ValueError(f"--cell-lines kept 0 rows using obs[{col!r}] and values {cell_lines}")
    return adata[keep].copy(), {"cell_line_filter_col": col, "cell_line_filter_values": ",".join(cell_lines)}


def evaluate_slide_classifier(adata: ad.AnnData, base: dict, seed: int, test_size: float) -> tuple[list[dict], dict]:
    slide_col = find_first_column(adata.obs, SLIDE_COLUMNS)
    if slide_col is None:
        return [], {"slide_status": "skipped_no_slide_column"}

    y_raw = adata.obs[slide_col].astype("string")
    mask = y_raw.notna().to_numpy()
    if mask.sum() < 4:
        return [], {"slide_status": "skipped_too_few_samples"}

    y_raw = y_raw.loc[mask].astype(str).to_numpy()
    counts = pd.Series(y_raw).value_counts()
    valid_classes = counts[counts >= 2].index
    keep = np.isin(y_raw, valid_classes)
    y_raw = y_raw[keep]
    if len(valid_classes) < 2 or len(y_raw) < 4:
        return [], {"slide_status": "skipped_too_few_slide_classes"}

    x = as_dense_float32(adata.X)[mask][keep]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = classifier_scores(model, x_test)

    auroc = np.nan
    try:
        if len(encoder.classes_) == 2:
            auroc = float(roc_auc_score(y_test, score))
        else:
            auroc = float(roc_auc_score(y_test, score, multi_class="ovr", average="macro"))
    except Exception:
        auroc = np.nan

    metrics = {
        "slide_status": "evaluated",
        "slide_col": slide_col,
        "slide_n_samples": int(len(y)),
        "slide_n_classes": int(len(encoder.classes_)),
        "slide_logreg_accuracy": float(accuracy_score(y_test, pred)),
        "slide_logreg_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "slide_logreg_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "slide_logreg_auroc": auroc,
    }
    row = {**base, **metrics}
    return [row], metrics


def evaluate_ridge_predictability(
    adata: ad.AnnData,
    base: dict,
    seed: int,
    test_size: float,
) -> tuple[list[dict], list[dict], dict]:
    x_all = as_dense_float32(adata.X)
    ridge_rows = []
    axis_rows = []
    wide = {}
    alphas = np.logspace(-3, 3, 13)

    for col in RIDGE_COVARIATES:
        if col not in adata.obs.columns:
            continue
        y_series = finite_numeric_series(adata.obs, col)
        mask = y_series.notna().to_numpy()
        if mask.sum() < 8:
            continue

        y = y_series.loc[mask].to_numpy(dtype=float)
        if np.nanstd(y) == 0:
            continue
        x = x_all[mask]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=seed,
        )
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas)),
            ]
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mse = float(mean_squared_error(y_test, pred))
        metrics = {
            "covariate": col,
            "n_samples": int(mask.sum()),
            "r2": float(r2_score(y_test, pred)),
            "pearson": safe_pearson(y_test, pred),
            "spearman": safe_spearman(y_test, pred),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mse)),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
        }
        ridge_rows.append({**base, **metrics})

        safe_col = sanitize_name(col)
        for key, value in metrics.items():
            if key not in {"covariate"}:
                wide[f"ridge_{safe_col}_{key}"] = value

        correlations = []
        for axis in range(x.shape[1]):
            corr = safe_spearman(y, x[:, axis])
            abs_corr = float(abs(corr)) if np.isfinite(corr) else np.nan
            axis_row = {
                **base,
                "covariate": col,
                "axis": int(axis),
                "spearman": corr,
                "abs_spearman": abs_corr,
            }
            axis_rows.append(axis_row)
            correlations.append((axis, corr, abs_corr))

        valid = [(axis, corr, abs_corr) for axis, corr, abs_corr in correlations if np.isfinite(abs_corr)]
        if valid:
            top_axis, top_corr, top_abs = max(valid, key=lambda item: item[2])
            wide[f"axis_{safe_col}_top_axis"] = int(top_axis)
            wide[f"axis_{safe_col}_top_spearman"] = float(top_corr)
            wide[f"axis_{safe_col}_max_abs_spearman"] = float(top_abs)
            wide[f"axis_{safe_col}_median_abs_spearman"] = float(np.nanmedian([item[2] for item in valid]))

    return ridge_rows, axis_rows, wide


def resolve_defect_column(obs: pd.DataFrame, requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in obs.columns else None
    for col in DEFECT_SCORE_PRIORITY:
        if col in obs.columns:
            return col
    return None


def myonucleus_eval_mask(obs: pd.DataFrame) -> np.ndarray | None:
    if "myotube_id" not in obs.columns:
        return None
    values = pd.to_numeric(obs["myotube_id"], errors="coerce").fillna(-1)
    return values.to_numpy() >= 0


def evaluate_myonucleus_classification(
    adata: ad.AnnData,
    base: dict,
    seed: int,
    test_size: float,
) -> tuple[list[dict], dict]:
    mask = myonucleus_eval_mask(adata.obs)
    if mask is None:
        return [], {"myonucleus_classification_status": "skipped_no_myotube_id"}

    y = mask.astype(int)
    non_myonucleus_n = int((y == 0).sum())
    myonucleus_n = int((y == 1).sum())
    summary = {
        "myonucleus_classification_n_samples": int(len(y)),
        "myonucleus_classification_n_non_myonucleus": non_myonucleus_n,
        "myonucleus_classification_n_myonucleus": myonucleus_n,
    }
    if min(non_myonucleus_n, myonucleus_n) < 2:
        return [], {
            **summary,
            "myonucleus_classification_status": "skipped_too_few_samples_in_one_class",
        }

    x = as_dense_float32(adata.X)
    n_test = max(2, int(math.ceil(len(y) * test_size)))
    n_test = min(n_test, len(y) - 2)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=n_test,
        random_state=seed,
        stratify=y,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = classifier_scores(model, x_test)

    metrics = {
        **summary,
        "myonucleus_classification_status": "evaluated",
        "myonucleus_classification_n_train": int(len(y_train)),
        "myonucleus_classification_n_test": int(len(y_test)),
        "myonucleus_classification_AUROC": float(roc_auc_score(y_test, score)),
        "myonucleus_classification_accuracy": float(accuracy_score(y_test, pred)),
        "myonucleus_classification_f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    return [{**base, **metrics}], metrics


def make_defect_models(seed: int, n_jobs: int, n_features: int) -> dict[str, object]:
    gamma = 1.0 / max(1, int(n_features))
    return {
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "KNN": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=n_jobs)),
            ]
        ),
        "RBF-SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("rbf", RBFSampler(gamma=gamma, n_components=512, random_state=seed)),
                ("model", LinearSVC(class_weight="balanced", max_iter=10000, random_state=seed)),
            ]
        ),
    }


def evaluate_defect_classification(
    adata: ad.AnnData,
    base: dict,
    requested_col: str,
    n_qbins: int,
    low_bin: int,
    high_bin: int,
    seed: int,
    test_size: float,
    n_jobs: int,
    myonuclei_only: bool = False,
) -> tuple[list[dict], dict]:
    myonucleus_filter_summary = {
        "defect_myonuclei_only": bool(myonuclei_only),
        "defect_n_before_myonucleus_filter": int(adata.n_obs),
        "defect_n_after_myonucleus_filter": int(adata.n_obs),
    }
    if myonuclei_only:
        mask = myonucleus_eval_mask(adata.obs)
        if mask is None:
            return [], {
                **myonucleus_filter_summary,
                "defect_status": "skipped_no_myotube_id_for_myonucleus_filter",
            }
        myonucleus_filter_summary["defect_n_after_myonucleus_filter"] = int(mask.sum())
        if mask.sum() < n_qbins * 2:
            return [], {
                **myonucleus_filter_summary,
                "defect_status": "skipped_too_few_myonuclei",
            }
        adata = adata[mask].copy()

    defect_col = resolve_defect_column(adata.obs, requested_col)
    if defect_col is None:
        return [], {**myonucleus_filter_summary, "defect_status": "skipped_no_defect_column"}

    score = finite_numeric_series(adata.obs, defect_col)
    valid = score.notna().to_numpy()
    if valid.sum() < n_qbins * 2:
        return [], {
            **myonucleus_filter_summary,
            "defect_status": "skipped_too_few_scores",
            "defect_score_col": defect_col,
        }

    score_valid = score.loc[valid]
    try:
        qbin = pd.qcut(
            score_valid.rank(method="first"),
            q=n_qbins,
            labels=False,
        ).astype(int)
    except Exception as exc:
        return [], {
            **myonucleus_filter_summary,
            "defect_status": "skipped_qbin_failed",
            "defect_score_col": defect_col,
            "defect_skip_reason": str(exc),
        }

    selected = qbin.isin([low_bin, high_bin])
    if selected.sum() < 4:
        return [], {
            **myonucleus_filter_summary,
            "defect_status": "skipped_empty_low_high_bins",
            "defect_score_col": defect_col,
        }

    y = (qbin.loc[selected].to_numpy(dtype=int) == high_bin).astype(int)
    x = as_dense_float32(adata.X)[valid][selected.to_numpy()]
    low_count = int((y == 0).sum())
    high_count = int((y == 1).sum())
    n_per_class = min(low_count, high_count)
    if n_per_class < 2:
        return [], {
            **myonucleus_filter_summary,
            "defect_status": "skipped_too_few_low_high_samples",
            "defect_score_col": defect_col,
            "defect_low_n_raw": low_count,
            "defect_high_n_raw": high_count,
        }

    rng = np.random.default_rng(seed)
    low_idx = np.flatnonzero(y == 0)
    high_idx = np.flatnonzero(y == 1)
    keep_idx = np.concatenate(
        [
            rng.choice(low_idx, size=n_per_class, replace=False),
            rng.choice(high_idx, size=n_per_class, replace=False),
        ]
    )
    rng.shuffle(keep_idx)
    x = x[keep_idx]
    y = y[keep_idx]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    rows = []
    wide = {
        **myonucleus_filter_summary,
        "defect_status": "evaluated",
        "defect_score_col": defect_col,
        "defect_n_qbins": int(n_qbins),
        "defect_low_bin": int(low_bin),
        "defect_high_bin": int(high_bin),
        "defect_low_n_raw": low_count,
        "defect_high_n_raw": high_count,
        "defect_n_per_class_balanced": int(n_per_class),
        "defect_n_train": int(len(y_train)),
        "defect_n_test": int(len(y_test)),
    }

    best = None
    for model_name, model in make_defect_models(seed, n_jobs, x.shape[1]).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score_pred = classifier_scores(model, x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        metrics = {
            "model": model_name,
            "model_impl": "RBFSampler+LinearSVC" if model_name == "RBF-SVM" else model_name,
            "defect_score_col": defect_col,
            "myonuclei_only": bool(myonuclei_only),
            "n_before_myonucleus_filter": myonucleus_filter_summary["defect_n_before_myonucleus_filter"],
            "n_after_myonucleus_filter": myonucleus_filter_summary["defect_n_after_myonucleus_filter"],
            "n_qbins": int(n_qbins),
            "low_bin": int(low_bin),
            "high_bin": int(high_bin),
            "low_n_raw": low_count,
            "high_n_raw": high_count,
            "n_per_class_balanced": int(n_per_class),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "auroc": float(roc_auc_score(y_test, score_pred)),
            "auprc": float(average_precision_score(y_test, score_pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        rows.append({**base, **metrics})

        safe_model = sanitize_name(model_name)
        for key, value in metrics.items():
            if key not in {"model", "model_impl", "defect_score_col"}:
                wide[f"defect_{safe_model}_{key}"] = value

        if model_name == "LogisticRegression":
            wide["defect_primary_model"] = model_name
            for key in ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]:
                wide[f"defect_primary_{key}"] = metrics[key]

        if best is None or metrics["auroc"] > best["auroc"]:
            best = metrics

    if best is not None:
        wide["defect_best_model"] = best["model"]
        for key in ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]:
            wide[f"defect_best_{key}"] = best[key]

    return rows, wide


def evaluate_run(run_dir: Path, args: argparse.Namespace, cell_lines: list[str] | None) -> dict:
    config = read_json(run_dir / CONFIG_NAME)
    summary = read_json(run_dir / SUMMARY_NAME)
    base = run_descriptors(run_dir, config, summary)
    master = {**base, "evaluation_status": "started", "skip_reason": None}

    loss_rows, loss_wide = collect_loss_metrics(run_dir, base)
    master.update(loss_wide)

    latent_path = resolve_latent_path(run_dir, config, summary)
    if latent_path is None:
        master.update({"evaluation_status": "skipped", "skip_reason": "missing_latent_h5ad"})
        return {
            "master": master,
            "loss_rows": loss_rows,
            "slide_rows": [],
            "myonucleus_rows": [],
            "ridge_rows": [],
            "axis_rows": [],
            "defect_rows": [],
            "skipped": [{**base, "skip_reason": "missing_latent_h5ad"}],
        }

    try:
        adata = ad.read_h5ad(latent_path)
        adata, filter_info = apply_cell_line_filter(adata, cell_lines)
    except Exception as exc:
        reason = f"latent_load_or_filter_failed: {exc}"
        master.update({"evaluation_status": "skipped", "skip_reason": reason})
        return {
            "master": master,
            "loss_rows": loss_rows,
            "slide_rows": [],
            "myonucleus_rows": [],
            "ridge_rows": [],
            "axis_rows": [],
            "defect_rows": [],
            "skipped": [{**base, "skip_reason": reason}],
        }

    master["latent_h5ad_path"] = str(latent_path)
    master["eval_n_obs"] = int(adata.n_obs)
    master["eval_n_latent"] = int(adata.n_vars)
    master.update(filter_info)

    try:
        slide_rows, slide_wide = evaluate_slide_classifier(adata, base, args.seed, args.test_size)
        master.update(slide_wide)
    except Exception as exc:
        master.update({"slide_status": "failed", "slide_error": str(exc)})
        slide_rows = []

    myonucleus_rows = []
    if str(base.get("input_mode", "")).lower() == "nuclei":
        try:
            myonucleus_rows, myonucleus_wide = evaluate_myonucleus_classification(
                adata,
                base,
                args.seed,
                args.test_size,
            )
            master.update(myonucleus_wide)
        except Exception as exc:
            master.update(
                {
                    "myonucleus_classification_status": "failed",
                    "myonucleus_classification_error": str(exc),
                }
            )

    try:
        ridge_rows, axis_rows, ridge_wide = evaluate_ridge_predictability(
            adata,
            base,
            args.seed,
            args.test_size,
        )
        master.update(ridge_wide)
    except Exception as exc:
        master.update({"ridge_status": "failed", "ridge_error": str(exc)})
        ridge_rows = []
        axis_rows = []

    try:
        defect_rows, defect_wide = evaluate_defect_classification(
            adata,
            base,
            args.defect_score_col,
            args.n_qbins,
            args.low_bin,
            args.high_bin,
            args.seed,
            args.test_size,
            args.n_jobs,
            args.classification_myonuclei_only,
        )
        master.update(defect_wide)
    except Exception as exc:
        master.update({"defect_status": "failed", "defect_error": str(exc)})
        defect_rows = []

    master["evaluation_status"] = "evaluated"
    return {
        "master": master,
        "loss_rows": loss_rows,
        "slide_rows": slide_rows,
        "myonucleus_rows": myonucleus_rows,
        "ridge_rows": ridge_rows,
        "axis_rows": axis_rows,
        "defect_rows": defect_rows,
        "skipped": [],
    }


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        if columns:
            ordered = columns + [col for col in df.columns if col not in columns]
            df = df.reindex(columns=ordered)
    else:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def plot_metric_by_hyperparams(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    if metric not in df.columns or df[metric].notna().sum() == 0:
        return
    plot_df = df.copy()
    for col in ["n_latent", "n_hidden", "n_layers"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric, "n_latent"])
    if plot_df.empty:
        return

    sns.set_style("whitegrid")
    grid = sns.relplot(
        data=plot_df,
        x="n_latent",
        y=metric,
        hue="n_hidden" if "n_hidden" in plot_df.columns else None,
        style="n_layers" if "n_layers" in plot_df.columns else None,
        col="gene_likelihood" if "gene_likelihood" in plot_df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
    )
    grid.fig.suptitle(title, y=1.05)
    grid.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)


def plot_ridge_covariates(df: pd.DataFrame, out_path: Path):
    metrics = [
        "ridge_n_counts_r2",
        "ridge_total_counts_r2",
        "ridge_total_counts_for_plot_r2",
        "ridge_area_px2_r2",
    ]
    existing = [col for col in metrics if col in df.columns and df[col].notna().sum() > 0]
    if not existing:
        return
    id_vars = [col for col in ["run_id", "n_latent", "n_hidden", "n_layers", "gene_likelihood"] if col in df.columns]
    plot_df = df[id_vars + existing].melt(
        id_vars=id_vars,
        value_vars=existing,
        var_name="metric",
        value_name="r2",
    )
    plot_df["r2"] = pd.to_numeric(plot_df["r2"], errors="coerce")
    plot_df["n_latent"] = pd.to_numeric(plot_df["n_latent"], errors="coerce")
    plot_df = plot_df.dropna(subset=["r2", "n_latent"])
    if plot_df.empty:
        return
    plot_df["metric"] = plot_df["metric"].str.replace("^ridge_", "", regex=True).str.replace("_r2$", "", regex=True)

    sns.set_style("whitegrid")
    grid = sns.relplot(
        data=plot_df,
        x="n_latent",
        y="r2",
        hue="metric",
        style="n_layers" if "n_layers" in plot_df.columns else None,
        col="gene_likelihood" if "gene_likelihood" in plot_df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
    )
    grid.fig.suptitle("Ridge covariate predictability from scVI latent", y=1.05)
    grid.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)


def write_plots(master_df: pd.DataFrame, outdir: Path):
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_by_hyperparams(
        master_df,
        "defect_best_auroc",
        plot_dir / "defect_best_auroc_by_hyperparams.png",
        "Defect classification AUROC",
    )
    plot_metric_by_hyperparams(
        master_df,
        "min_validation_loss",
        plot_dir / "validation_loss_by_hyperparams.png",
        "Minimum validation loss",
    )
    plot_metric_by_hyperparams(
        master_df,
        "slide_logreg_accuracy",
        plot_dir / "slide_classifier_accuracy_by_hyperparams.png",
        "Slide classifier accuracy",
    )
    plot_ridge_covariates(master_df, plot_dir / "ridge_counts_area_r2_by_hyperparams.png")


def main():
    args = parse_args()
    start = time.time()
    sweep_dir = Path(args.sweep_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else sweep_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")

    cell_lines = split_csv_values(args.cell_lines)
    run_dirs = discover_run_dirs(sweep_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {sweep_dir}")

    all_master = []
    all_loss = []
    all_slide = []
    all_myonucleus = []
    all_ridge = []
    all_axis = []
    all_defect = []
    all_skipped = []

    print(f"[eval] Sweep dir: {sweep_dir}", flush=True)
    print(f"[eval] Output dir: {output_dir}", flush=True)
    print(f"[eval] Found {len(run_dirs)} run directories", flush=True)

    for run_dir in tqdm(run_dirs, desc="Evaluating scVI runs"):
        result = evaluate_run(run_dir, args, cell_lines)
        all_master.append(result["master"])
        all_loss.extend(result["loss_rows"])
        all_slide.extend(result["slide_rows"])
        all_myonucleus.extend(result["myonucleus_rows"])
        all_ridge.extend(result["ridge_rows"])
        all_axis.extend(result["axis_rows"])
        all_defect.extend(result["defect_rows"])
        all_skipped.extend(result["skipped"])

    master_df = pd.DataFrame(all_master)
    if "run_id" in master_df.columns:
        master_df = master_df.sort_values("run_id", kind="stable")
    master_df.to_csv(output_dir / MASTER_NAME, index=False)
    master_df.to_csv(output_dir / "evaluation_summary.csv", index=False)

    write_csv(
        output_dir / "loss_metrics.csv",
        all_loss,
        BASE_COLUMNS + ["split", "metric_name", "n_epochs", "final_value", "min_value", "min_epoch"],
    )
    write_csv(
        output_dir / "slide_classifier_metrics.csv",
        all_slide,
        BASE_COLUMNS
        + [
            "slide_status",
            "slide_col",
            "slide_n_samples",
            "slide_n_classes",
            "slide_logreg_accuracy",
            "slide_logreg_balanced_accuracy",
            "slide_logreg_macro_f1",
            "slide_logreg_auroc",
        ],
    )
    write_csv(
        output_dir / "myonucleus_classifier_metrics.csv",
        all_myonucleus,
        BASE_COLUMNS
        + [
            "myonucleus_classification_status",
            "myonucleus_classification_n_samples",
            "myonucleus_classification_n_non_myonucleus",
            "myonucleus_classification_n_myonucleus",
            "myonucleus_classification_n_train",
            "myonucleus_classification_n_test",
            "myonucleus_classification_AUROC",
            "myonucleus_classification_accuracy",
            "myonucleus_classification_f1",
        ],
    )
    write_csv(
        output_dir / "ridge_predictability_metrics.csv",
        all_ridge,
        BASE_COLUMNS + ["covariate", "n_samples", "r2", "pearson", "spearman", "mae", "rmse", "ridge_alpha"],
    )
    write_csv(
        output_dir / "axis_covariate_correlations.csv",
        all_axis,
        BASE_COLUMNS + ["covariate", "axis", "spearman", "abs_spearman"],
    )
    write_csv(
        output_dir / "defect_classifier_metrics.csv",
        all_defect,
        BASE_COLUMNS
        + [
            "model",
            "model_impl",
            "defect_score_col",
            "myonuclei_only",
            "n_before_myonucleus_filter",
            "n_after_myonucleus_filter",
            "n_qbins",
            "low_bin",
            "high_bin",
            "low_n_raw",
            "high_n_raw",
            "n_per_class_balanced",
            "n_train",
            "n_test",
            "auroc",
            "auprc",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    write_csv(output_dir / "skipped_runs.csv", all_skipped, BASE_COLUMNS + ["skip_reason"])

    config = {
        "sweep_dir": str(sweep_dir),
        "output_dir": str(output_dir),
        "n_run_dirs": len(run_dirs),
        "n_master_rows": int(master_df.shape[0]),
        "defect_score_col": args.defect_score_col,
        "cell_lines": cell_lines,
        "n_qbins": args.n_qbins,
        "low_bin": args.low_bin,
        "high_bin": args.high_bin,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "skip_plots": bool(args.skip_plots),
        "classification_myonuclei_only": bool(args.classification_myonuclei_only),
        "runtime_seconds": time.time() - start,
        "outputs": {
            "master": str(output_dir / MASTER_NAME),
            "summary": str(output_dir / "evaluation_summary.csv"),
            "loss_metrics": str(output_dir / "loss_metrics.csv"),
            "slide_classifier_metrics": str(output_dir / "slide_classifier_metrics.csv"),
            "myonucleus_classifier_metrics": str(output_dir / "myonucleus_classifier_metrics.csv"),
            "ridge_predictability_metrics": str(output_dir / "ridge_predictability_metrics.csv"),
            "axis_covariate_correlations": str(output_dir / "axis_covariate_correlations.csv"),
            "defect_classifier_metrics": str(output_dir / "defect_classifier_metrics.csv"),
            "skipped_runs": str(output_dir / "skipped_runs.csv"),
        },
    }
    with open(output_dir / "evaluation_config.json", "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(config), handle, indent=2, sort_keys=True)

    if not args.skip_plots:
        write_plots(master_df, output_dir)

    print(f"[eval] Wrote master table: {output_dir / MASTER_NAME}", flush=True)
    print(f"[eval] Done in {time.time() - start:.1f} seconds", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
