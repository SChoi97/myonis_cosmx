"""Paper-faithful approximation of Harmony Advanced flatfield correction.

This is an independent implementation based on Kask et al. (2016), not
Revvity/PerkinElmer source code. Every selected FOV is retained; segmentation
selects reliable pixels within each image rather than rejecting whole FOVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.transform import resize


def _mean_normalize(profile: np.ndarray, minimum: float = 0.05) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    finite_positive = profile[np.isfinite(profile) & (profile > 0)]
    if finite_positive.size == 0:
        raise ValueError("Profile contains no finite positive values.")
    fill = float(np.mean(finite_positive))
    profile = np.where(np.isfinite(profile), profile, fill)
    profile = np.maximum(profile, minimum * fill)
    profile /= np.mean(profile)
    return profile.astype(np.float32)


def _grid_edges(length: int, nominal_block_size: int) -> np.ndarray:
    block_count = min(17, max(1, int(round(length / nominal_block_size))))
    return np.linspace(0, length, block_count + 1, dtype=int)


def segment_profile_regions(
    image: np.ndarray,
    *,
    nominal_block_size: int = 64,
    local_window: int = 9,
    noise_mad_multiplier: float = 3.0,
    foreground_sigma_multiplier: float = 5.0,
) -> dict:
    """Segment low-noise background and block-local foreground pixels."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; received {image.shape}.")
    local_mean = ndi.uniform_filter(image, size=local_window, mode="reflect")
    trend = ndi.gaussian_filter(image, sigma=max(local_window / 3.0, 1.0))
    residual = image - trend
    residual_mean = ndi.uniform_filter(residual, size=local_window, mode="reflect")
    residual_second = ndi.uniform_filter(
        residual * residual, size=local_window, mode="reflect"
    )
    local_variance = np.maximum(
        residual_second - residual_mean * residual_mean, 0
    )
    noise_image = local_variance / np.maximum(local_mean, 1.0)

    low_intensity = image <= np.percentile(image, 60.0)
    candidates = noise_image[low_intensity & np.isfinite(noise_image)]
    if candidates.size == 0:
        candidates = noise_image[np.isfinite(noise_image)]
    noise_median = float(np.median(candidates))
    noise_mad = float(np.median(np.abs(candidates - noise_median)))
    noise_sigma = max(1.4826 * noise_mad, np.finfo(np.float32).eps)
    low_noise = np.isfinite(noise_image) & (
        noise_image <= noise_median + noise_mad_multiplier * noise_sigma
    )

    safe = image[low_noise & low_intensity]
    if safe.size == 0:
        safe = image[low_intensity]
    intensity_median = float(np.median(safe))
    intensity_mad = float(np.median(np.abs(safe - intensity_median)))
    intensity_limit = intensity_median + 3.0 * max(1.4826 * intensity_mad, 1.0)
    background_mask = low_noise & (image <= intensity_limit)
    background_mask = ndi.binary_erosion(
        background_mask, iterations=1, border_value=0
    )

    background_noise = noise_image[background_mask]
    noise_constant = (
        float(np.median(background_noise))
        if background_noise.size
        else noise_median
    )
    noise_constant = max(noise_constant, np.finfo(np.float32).eps)
    y_edges = _grid_edges(image.shape[0], nominal_block_size)
    x_edges = _grid_edges(image.shape[1], nominal_block_size)
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    background_bins = np.full(shape, np.nan, dtype=np.float32)
    foreground_bins = np.full(shape, np.nan, dtype=np.float32)
    foreground_mask = np.zeros(image.shape, dtype=bool)

    for row, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for column, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            block = image[y0:y1, x0:x1]
            block_background = background_mask[y0:y1, x0:x1]
            if np.any(block_background):
                background_bins[row, column] = float(
                    np.mean(block[block_background])
                )
            block_mean = float(np.mean(block))
            expected_sd = np.sqrt(max(noise_constant * block_mean, 0.0))
            threshold = block_mean + foreground_sigma_multiplier * expected_sd
            block_foreground = np.isfinite(block) & (block > threshold)
            foreground_mask[y0:y1, x0:x1] = block_foreground
            if np.any(block_foreground):
                foreground_bins[row, column] = float(
                    np.mean(block[block_foreground])
                )
    return {
        "background_bins": background_bins,
        "foreground_bins": foreground_bins,
        "background_mask": background_mask,
        "foreground_mask": foreground_mask,
        "background_fraction": float(np.mean(background_mask)),
        "foreground_fraction": float(np.mean(foreground_mask)),
        "grid_shape": shape,
    }


def _second_difference(length: int) -> sparse.csr_matrix:
    if length < 3:
        return sparse.csr_matrix((0, length))
    return sparse.diags(
        (np.ones(length - 2), -2 * np.ones(length - 2), np.ones(length - 2)),
        (0, 1, 2),
        shape=(length - 2, length),
        format="csr",
    )


def _curvature_penalty(shape: tuple[int, int]) -> sparse.csr_matrix:
    rows, columns = shape
    dyy = sparse.kron(_second_difference(rows), sparse.eye(columns), format="csr")
    dxx = sparse.kron(sparse.eye(rows), _second_difference(columns), format="csr")
    return dyy.T @ dyy + dxx.T @ dxx


def _regularized_surface(
    observed: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("No valid observations remain for profile fitting.")
    fill = float(np.nanmean(observed[valid]))
    observed = np.where(valid, observed, fill)
    weights = np.where(valid, weights, 0.0)
    weights /= max(float(np.max(weights)), 1.0)
    vector_weights = weights.ravel()
    system = (
        sparse.diags(vector_weights, format="csr")
        + regularization * _curvature_penalty(observed.shape)
        + 1e-8 * sparse.eye(observed.size, format="csr")
    )
    fitted = spsolve(system, vector_weights * observed.ravel()).reshape(
        observed.shape
    )
    return _mean_normalize(fitted)


def _rank1_fit(
    images: np.ndarray,
    *,
    regularization: float,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(images, dtype=np.float64)
    valid = np.isfinite(data) & (data > 0)
    profile = np.ones(data.shape[1:], dtype=np.float64)
    scales = np.full(data.shape[0], np.nan, dtype=np.float64)
    for _ in range(iterations):
        for index in range(data.shape[0]):
            mask = valid[index]
            if np.any(mask):
                denominator = np.sum(profile[mask] ** 2)
                scales[index] = np.sum(data[index][mask] * profile[mask]) / max(
                    denominator, 1e-12
                )
        normalized = data / np.maximum(scales[:, None, None], 1e-12)
        observed = np.nanmean(np.where(valid, normalized, np.nan), axis=0)
        profile = _regularized_surface(
            observed, np.sum(valid, axis=0), regularization
        )
    return (
        profile.astype(np.float32),
        scales.astype(np.float32),
        np.sum(valid, axis=0),
    )


def _group_average(images: np.ndarray, group_size: int) -> np.ndarray:
    groups = []
    for start in range(0, images.shape[0], group_size):
        group = images[start : start + group_size]
        count = np.sum(np.isfinite(group), axis=0)
        summed = np.nansum(group, axis=0)
        groups.append(
            np.divide(
                summed,
                count,
                out=np.full(summed.shape, np.nan, dtype=np.float64),
                where=count > 0,
            )
        )
    return np.stack(groups).astype(np.float32)


def _split_half_error(images: np.ndarray, regularization: float) -> float:
    if images.shape[0] < 4:
        return float("nan")
    first, _, _ = _rank1_fit(
        images[0::2], regularization=regularization, iterations=8
    )
    second, _, _ = _rank1_fit(
        images[1::2], regularization=regularization, iterations=8
    )
    return float(50.0 * np.ptp(first - second))


def fit_profiles(
    training_paths: Sequence[Path],
    *,
    target_shape: tuple[int, int],
    label: str,
    random_seed: int = 20260819,
    nominal_block_size: int = 64,
    local_window: int = 9,
    group_size: int = 128,
    background_regularization: float = 0.2,
    foreground_regularization: float = 20.0,
) -> dict:
    """Fit additive background and multiplicative foreground profiles."""
    paths = sorted(Path(path) for path in training_paths)
    if not paths:
        raise ValueError(f"No Harmony-like training paths supplied for {label}.")
    rng = np.random.default_rng(random_seed)
    paths = [paths[index] for index in rng.permutation(len(paths))]
    background_bins = []
    foreground_bins = []
    fallback_backgrounds = []
    background_fractions = []
    foreground_fractions = []
    expected_shape = None
    print(f"[{label}] Harmony-like: segmenting {len(paths)} shared FOVs...")
    for number, path in enumerate(paths, start=1):
        image = np.asarray(iio.imread(path))
        segmented = segment_profile_regions(
            image,
            nominal_block_size=nominal_block_size,
            local_window=local_window,
        )
        if expected_shape is None:
            expected_shape = segmented["grid_shape"]
        elif segmented["grid_shape"] != expected_shape:
            raise ValueError(
                f"Inconsistent Harmony grid shapes at {path}: "
                f"{expected_shape} versus {segmented['grid_shape']}."
            )
        background_bins.append(segmented["background_bins"])
        foreground_bins.append(segmented["foreground_bins"])
        fallback_backgrounds.append(float(np.percentile(image, 10.0)))
        background_fractions.append(segmented["background_fraction"])
        foreground_fractions.append(segmented["foreground_fraction"])
        if number % 100 == 0 or number == len(paths):
            print(f"  Harmony-like segmented {number}/{len(paths)}")

    background_bins_array = np.stack(background_bins)
    foreground_bins_array = np.stack(foreground_bins)
    background_small, background_scales, background_counts = _rank1_fit(
        background_bins_array, regularization=background_regularization
    )
    fallback = np.asarray(fallback_backgrounds, dtype=np.float32)
    background_scales = np.where(
        np.isfinite(background_scales) & (background_scales > 0),
        background_scales,
        fallback,
    )
    foreground_signal = (
        foreground_bins_array
        - background_scales[:, None, None] * background_small[None, :, :]
    )
    foreground_signal = np.where(foreground_signal > 0, foreground_signal, np.nan)
    grouped = _group_average(foreground_signal, group_size)
    foreground_small, _, foreground_counts = _rank1_fit(
        grouped, regularization=foreground_regularization
    )
    background_error = _split_half_error(
        background_bins_array, background_regularization
    )
    foreground_error = _split_half_error(grouped, foreground_regularization)
    background_profile = _mean_normalize(
        resize(
            background_small,
            target_shape,
            order=3,
            preserve_range=True,
            anti_aliasing=False,
        )
    )
    foreground_profile = _mean_normalize(
        resize(
            foreground_small,
            target_shape,
            order=3,
            preserve_range=True,
            anti_aliasing=False,
        )
    )
    return {
        "background_profile": background_profile,
        "foreground_profile": foreground_profile,
        "background_profile_small": background_small,
        "foreground_profile_small": foreground_small,
        "background_counts": background_counts,
        "foreground_counts": foreground_counts,
        "background_split_half_range_error_percent": background_error,
        "foreground_split_half_range_error_percent": foreground_error,
        "background_fraction_median": float(np.median(background_fractions)),
        "foreground_fraction_median": float(np.median(foreground_fractions)),
        "available_path_count": len(paths),
        "sampled_path_count": len(paths),
        "retained_path_count": len(paths),
        "training_paths": paths,
        "group_count": int(grouped.shape[0]),
        "background_regularization": float(background_regularization),
        "foreground_regularization": float(foreground_regularization),
        "nominal_block_size": int(nominal_block_size),
        "local_window": int(local_window),
        "group_size": int(group_size),
    }


def _background_amplitude(
    image: np.ndarray,
    background_profile: np.ndarray,
    *,
    nominal_block_size: int,
    local_window: int,
) -> float:
    segmented = segment_profile_regions(
        image,
        nominal_block_size=nominal_block_size,
        local_window=local_window,
    )
    mask = segmented["background_mask"]
    image = np.asarray(image, dtype=np.float32)
    if np.count_nonzero(mask) >= 100:
        return float(np.mean(image[mask] / np.maximum(background_profile[mask], 1e-6)))
    return float(
        np.percentile(image / np.maximum(background_profile, 1e-6), 10.0)
    )


def apply_correction(
    field_stack: np.ndarray,
    model: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply additive-background then multiplicative-profile correction."""
    images = np.asarray(field_stack)
    background = np.asarray(model["background_profile"], dtype=np.float32)
    foreground = np.maximum(
        np.asarray(model["foreground_profile"], dtype=np.float32), 1e-6
    )
    if images.shape[1:] != background.shape:
        raise ValueError(
            f"Field/profile mismatch: {images.shape[1:]} versus {background.shape}."
        )
    corrected = np.empty(images.shape, dtype=np.float32)
    amplitudes = np.empty(images.shape[0], dtype=np.float32)
    for index, image in enumerate(images):
        image_float = np.asarray(image, dtype=np.float32)
        amplitude = _background_amplitude(
            image_float,
            background,
            nominal_block_size=model["nominal_block_size"],
            local_window=model["local_window"],
        )
        corrected[index] = (
            image_float - amplitude * background
        ) / foreground + amplitude
        corrected[index] = np.clip(corrected[index], 0, None)
        amplitudes[index] = amplitude
    return corrected, amplitudes
