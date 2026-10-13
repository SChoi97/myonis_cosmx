"""Whole-well Fourier registration, robust field consensus, and crop geometry."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from .layout import FIELD_ID_GRID, field_box


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    sample = values[::8, ::8]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, [p_low, p_high])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = np.clip(values, low, high)
    normalized = (normalized - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 0
    return normalized.astype(np.float32)


def normalized_correlation(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_values = robust_normalize(fixed).ravel().astype(np.float64)
    moving_values = robust_normalize(moving).ravel().astype(np.float64)
    fixed_values -= fixed_values.mean()
    moving_values -= moving_values.mean()
    denominator = np.linalg.norm(fixed_values) * np.linalg.norm(moving_values)
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(np.dot(fixed_values, moving_values) / denominator)


def estimate_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    downsample: int = 1,
    upsample_factor: int = 20,
) -> dict:
    if fixed.shape != moving.shape or fixed.ndim != 2:
        raise ValueError(f"Expected equal 2-D images; got {fixed.shape}, {moving.shape}.")
    fixed_fft = robust_normalize(fixed[::downsample, ::downsample])
    moving_fft = robust_normalize(moving[::downsample, ::downsample])
    shift_small, error, phase = phase_cross_correlation(
        fixed_fft,
        moving_fft,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    shift = np.asarray(shift_small, dtype=float) * downsample
    return {
        "shift_y": float(shift[0]),
        "shift_x": float(shift[1]),
        "shift_magnitude": float(np.hypot(*shift)),
        "phase_error": float(error),
        "phase_difference": float(phase),
        "fft_shape": list(fixed_fft.shape),
    }


def apply_translation(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    return ndi.shift(
        np.asarray(image, dtype=np.float32),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def valid_bounds_for_shift(
    shape: tuple[int, int], shift_y: float, shift_x: float
) -> tuple[int, int, int, int]:
    height, width = shape

    def axis_bounds(length: int, shift: float) -> tuple[int, int]:
        start = max(0, int(math.ceil(shift)))
        end = min(length, int(math.floor((length - 1) + shift)) + 1)
        return start, max(start, end)

    y0, y1 = axis_bounds(height, shift_y)
    x0, x1 = axis_bounds(width, shift_x)
    return y0, y1, x0, x1


def bounds_area(bounds: tuple[int, int, int, int]) -> int:
    y0, y1, x0, x1 = bounds
    return max(0, y1 - y0) * max(0, x1 - x0)


def intersect_bounds(
    bounds: Iterable[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    bounds = tuple(bounds)
    if not bounds:
        raise ValueError("At least one bounds tuple is required.")
    y0 = max(item[0] for item in bounds)
    y1 = min(item[1] for item in bounds)
    x0 = max(item[2] for item in bounds)
    x1 = min(item[3] for item in bounds)
    return y0, max(y0, y1), x0, max(x0, x1)


def centered_patch_crop(
    bounds: tuple[int, int, int, int], patch_size: int
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = bounds
    target_height = ((y1 - y0) // patch_size) * patch_size
    target_width = ((x1 - x0) // patch_size) * patch_size
    if target_height == 0 or target_width == 0:
        return y0, y0, x0, x0
    y_margin = (y1 - y0 - target_height) // 2
    x_margin = (x1 - x0 - target_width) // 2
    return (
        y0 + y_margin,
        y0 + y_margin + target_height,
        x0 + x_margin,
        x0 + x_margin + target_width,
    )


def global_qc(
    fixed: np.ndarray,
    moving: np.ndarray,
    result: dict,
    *,
    max_shift_px: float,
    min_overlap_fraction: float,
    min_correlation: float,
) -> dict:
    registered = apply_translation(moving, result["shift_y"], result["shift_x"])
    bounds = valid_bounds_for_shift(
        fixed.shape, result["shift_y"], result["shift_x"]
    )
    y0, y1, x0, x1 = bounds
    overlap_fraction = bounds_area(bounds) / float(fixed.size)
    correlation = (
        normalized_correlation(fixed[y0:y1, x0:x1], registered[y0:y1, x0:x1])
        if bounds_area(bounds)
        else float("nan")
    )
    reasons = []
    if not np.isfinite(result["shift_magnitude"]):
        reasons.append("nonfinite_shift")
    if max(abs(result["shift_y"]), abs(result["shift_x"])) > max_shift_px:
        reasons.append("shift_exceeds_limit")
    if overlap_fraction < min_overlap_fraction:
        reasons.append("insufficient_overlap")
    if not np.isfinite(correlation) or correlation < min_correlation:
        reasons.append("low_global_correlation")
    return {
        **result,
        "overlap_fraction": overlap_fraction,
        "post_registration_correlation": correlation,
        "valid_bounds": list(bounds),
        "global_qc_status": "pass" if not reasons else "fail",
        "global_qc_reasons": ";".join(reasons),
        "registered": registered,
    }


def field_candidates(
    fixed_fields: np.ndarray,
    moving_fields: np.ndarray,
    *,
    upsample_factor: int,
    min_normalized_std: float,
) -> list[dict]:
    candidates = []
    for index in range(36):
        fixed = robust_normalize(fixed_fields[index])
        moving = robust_normalize(moving_fields[index])
        informative = min(float(np.std(fixed)), float(np.std(moving))) >= min_normalized_std
        row = {"field_id": index + 1, "informative": informative}
        if informative:
            row.update(
                estimate_translation(
                    fixed_fields[index], moving_fields[index], upsample_factor=upsample_factor
                )
            )
        candidates.append(row)
    return candidates


def robust_consensus_shift(
    candidates: list[dict], *, tolerance_px: float, minimum_fields: int
) -> dict | None:
    available = [
        row
        for row in candidates
        if row.get("informative")
        and np.isfinite(row.get("shift_y", np.nan))
        and np.isfinite(row.get("shift_x", np.nan))
    ]
    if len(available) < minimum_fields:
        return None
    values = np.asarray([[row["shift_y"], row["shift_x"]] for row in available])
    center = np.median(values, axis=0)
    distances = np.linalg.norm(values - center, axis=1)
    inlier_mask = distances <= tolerance_px
    if int(np.count_nonzero(inlier_mask)) < minimum_fields:
        return None
    inliers = values[inlier_mask]
    shift = np.median(inliers, axis=0)
    return {
        "shift_y": float(shift[0]),
        "shift_x": float(shift[1]),
        "shift_magnitude": float(np.hypot(*shift)),
        "phase_error": float("nan"),
        "phase_difference": float("nan"),
        "fft_shape": None,
        "consensus_field_count": int(inliers.shape[0]),
        "candidate_field_count": len(available),
    }


def validate_registered_fields(
    fixed_mosaic: np.ndarray,
    registered_mosaic: np.ndarray,
    *,
    upsample_factor: int,
    max_residual_px: float,
    min_correlation: float,
    min_normalized_std: float,
) -> list[dict]:
    rows = []
    for field_id in range(1, 37):
        y0, y1, x0, x1 = field_box(field_id)
        fixed = fixed_mosaic[y0:y1, x0:x1]
        moving = registered_mosaic[y0:y1, x0:x1]
        fixed_norm = robust_normalize(fixed)
        moving_norm = robust_normalize(moving)
        normalized_std = min(float(np.std(fixed_norm)), float(np.std(moving_norm)))
        grid_position = np.argwhere(FIELD_ID_GRID == field_id)[0]
        row = {
            "field_id": field_id,
            "grid_row": int(grid_position[0]) + 1,
            "grid_column": int(grid_position[1]) + 1,
            "normalized_std": normalized_std,
        }
        if normalized_std < min_normalized_std:
            row.update(
                {
                    "status": "not_evaluable",
                    "correlation": float("nan"),
                    "residual_shift_y": float("nan"),
                    "residual_shift_x": float("nan"),
                    "residual_magnitude": float("nan"),
                    "failure_reasons": "low_information",
                }
            )
            rows.append(row)
            continue
        residual = estimate_translation(fixed, moving, upsample_factor=upsample_factor)
        correlation = normalized_correlation(fixed, moving)
        reasons = []
        if residual["shift_magnitude"] > max_residual_px:
            reasons.append("large_local_residual")
        if not np.isfinite(correlation) or correlation < min_correlation:
            reasons.append("low_field_correlation")
        row.update(
            {
                "status": "fail" if reasons else "pass",
                "correlation": correlation,
                "residual_shift_y": residual["shift_y"],
                "residual_shift_x": residual["shift_x"],
                "residual_magnitude": residual["shift_magnitude"],
                "failure_reasons": ";".join(reasons),
            }
        )
        rows.append(row)
    evaluable = [row for row in rows if row["status"] != "not_evaluable"]
    if evaluable:
        median = np.median(
            [[row["residual_shift_y"], row["residual_shift_x"]] for row in evaluable],
            axis=0,
        )
        for row in rows:
            if row["status"] == "not_evaluable":
                row["residual_consensus_disagreement"] = float("nan")
            else:
                row["residual_consensus_disagreement"] = float(
                    np.hypot(
                        row["residual_shift_y"] - median[0],
                        row["residual_shift_x"] - median[1],
                    )
                )
    return rows

