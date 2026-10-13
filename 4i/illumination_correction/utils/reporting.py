"""Incremental metric tables and compact cross-well summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


WELL_METRICS = (
    "edge_energy",
    "edge_energy_field_mean",
    "edge_energy_field_sd",
    "edge_energy_field_min",
    "edge_energy_field_max",
    "p_positive",
    "p_negative",
    "p_magnitude",
    "p_positive_x_axis",
    "p_positive_y_axis",
    "boundary_discontinuity",
    "selected_seam_disparity_score",
)


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=1 if finite.size > 1 else 0))


def summarize_metrics(
    marker_name: str,
    display_label: str,
    per_field_rows: Sequence[dict],
    per_well_rows: Sequence[dict],
    harmony_model: dict,
) -> list[dict]:
    """Return one compact mean/stdev row per correction method."""
    methods = ("Original", "BaSiC", "Harmony-like")
    summary = []
    for method in methods:
        wells = [row for row in per_well_rows if row["method"] == method]
        fields = [row for row in per_field_rows if row["method"] == method]
        row = {
            "marker": marker_name,
            "display_label": display_label,
            "method": method,
            "visualization_well_count": len(wells),
            "field_observation_count": len(fields),
        }
        field_mean, field_sd = _mean_sd([item["edge_energy"] for item in fields])
        row["per_field_edge_energy_mean"] = field_mean
        row["per_field_edge_energy_std"] = field_sd
        for metric in WELL_METRICS:
            mean, sd = _mean_sd([item[metric] for item in wells])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sd
        row["edge_threshold"] = wells[0]["edge_threshold"] if wells else float("nan")
        row["metric_binning"] = wells[0]["metric_binning"] if wells else float("nan")
        if method == "Harmony-like":
            row["background_split_half_range_error_percent"] = harmony_model[
                "background_split_half_range_error_percent"
            ]
            row["foreground_split_half_range_error_percent"] = harmony_model[
                "foreground_split_half_range_error_percent"
            ]
        else:
            row["background_split_half_range_error_percent"] = float("nan")
            row["foreground_split_half_range_error_percent"] = float("nan")
        summary.append(row)
    return summary


def save_model_diagnostics(metrics_dir: Path, harmony_model: dict) -> None:
    """Save Harmony profiles plus scalar fit diagnostics."""
    np.savez_compressed(
        metrics_dir / "harmony_profiles.npz",
        background_profile=harmony_model["background_profile"],
        foreground_profile=harmony_model["foreground_profile"],
        background_profile_small=harmony_model["background_profile_small"],
        foreground_profile_small=harmony_model["foreground_profile_small"],
    )
    scalar_diagnostics = {
        key: value
        for key, value in harmony_model.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with (metrics_dir / "harmony_model_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(scalar_diagnostics, stream, indent=2, allow_nan=True)


def save_basic_diagnostics(metrics_dir: Path, diagnostics: dict) -> None:
    """Save the BaSiC fit and finite-output validation report."""
    with (metrics_dir / "basic_model_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(diagnostics, stream, indent=2, allow_nan=True)
