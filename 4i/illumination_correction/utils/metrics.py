"""Independent implementation of the raw no-reference EVEN metrics."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.signal import peak_prominences

from .layout import FIELD_ID_GRID, GRID_SHAPE


@dataclass(frozen=True)
class EvenSettings:
    edge_threshold: float = 3.0 / 8.0
    binning: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.edge_threshold < 1:
            raise ValueError("EVEN edge threshold must be between 0 and 1.")
        if self.binning < 1:
            raise ValueError("EVEN binning must be positive.")


def radial_edge_mask(
    shape: tuple[int, int], threshold: float = 3.0 / 8.0
) -> np.ndarray:
    height, width = shape
    radius = int(min(shape) * threshold)
    y = np.arange(height) - height // 2
    x = np.arange(width) - width // 2
    xx, yy = np.meshgrid(x, y)
    return xx * xx + yy * yy > radius * radius


def edge_energy_ratio(
    image: np.ndarray,
    *,
    threshold: float = 3.0 / 8.0,
    edge_mask: np.ndarray | None = None,
) -> float:
    image = np.asarray(image, dtype=np.float64)
    if edge_mask is None:
        edge_mask = radial_edge_mask(image.shape, threshold)
    total = float(np.sum(image, dtype=np.float64))
    if not np.isfinite(total) or total == 0:
        return float("nan")
    return float(np.sum(image[edge_mask], dtype=np.float64)) / total


def mean_bin(image: np.ndarray, factor: int) -> np.ndarray:
    image = np.asarray(image)
    if factor == 1:
        return image.astype(np.float32, copy=False)
    height, width = image.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Image shape {image.shape} is not divisible by binning {factor}."
        )
    return (
        image.reshape(height // factor, factor, width // factor, factor)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )


def normalized_power_spectrum(image: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
    total = float(np.sum(spectrum, dtype=np.float64))
    return spectrum / total if np.isfinite(total) and total else np.zeros(image.shape)


def periodic_peak_positions(length: int, tile_period: int) -> np.ndarray:
    if tile_period % 2 == 0:
        return np.linspace(0, length, tile_period, endpoint=False).astype(int)
    spacing = length // tile_period
    return np.asarray(
        [spacing * (0.5 + index) for index in range(tile_period)], dtype=int
    )


def _prominence_sum(profile: np.ndarray, positions: np.ndarray) -> float:
    positions = positions[(positions >= 0) & (positions < profile.size)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = peak_prominences(np.asarray(profile, dtype=float), positions)[0]
    return float(np.sum(values, dtype=np.float64))


def prominence_metrics(
    mosaic: np.ndarray, *, tile_shape: tuple[int, int]
) -> dict[str, float]:
    power = normalized_power_spectrum(np.asarray(mosaic, dtype=np.float32))
    tile_height, tile_width = tile_shape
    grid_y = power.shape[0] // tile_height
    grid_x = power.shape[1] // tile_width
    center_y, center_x = power.shape[0] // 2, power.shape[1] // 2
    y0, y1 = center_y - grid_y // 2 + 1, center_y + grid_y // 2
    x0, x1 = center_x - grid_x // 2 + 1, center_x + grid_x // 2
    power[max(0, y0) : y1, max(0, x0) : x1] = 0
    peaks_y = periodic_peak_positions(power.shape[0], tile_height)
    peaks_x = periodic_peak_positions(power.shape[1], tile_width)
    profile_y = np.sum(power, axis=1)
    profile_x = np.sum(power, axis=0)
    positive_y = _prominence_sum(profile_y, peaks_y)
    positive_x = _prominence_sum(profile_x, peaks_x)
    negative_y = _prominence_sum(-profile_y, peaks_y)
    negative_x = _prominence_sum(-profile_x, peaks_x)
    positive = float(np.mean((positive_y, positive_x)))
    negative = float(np.mean((negative_y, negative_x)))
    return {
        "p_positive": positive,
        "p_negative": negative,
        "p_magnitude": positive + negative,
        "p_positive_x_axis": _prominence_sum(power[center_y, :], peaks_x),
        "p_positive_y_axis": _prominence_sum(power[:, center_x], peaks_y),
    }


def _field_positions() -> dict[int, tuple[int, int]]:
    return {
        int(FIELD_ID_GRID[row, column]): (row + 1, column + 1)
        for row in range(FIELD_ID_GRID.shape[0])
        for column in range(FIELD_ID_GRID.shape[1])
    }


def evaluate_method(
    fields: np.ndarray,
    mosaic: np.ndarray,
    *,
    method: str,
    settings: EvenSettings,
) -> tuple[dict, list[dict]]:
    """Return full-well metrics and the 36 per-field E_edge values."""
    fields = np.asarray(fields)
    if fields.ndim != 3 or fields.shape[0] != int(np.prod(GRID_SHAPE)):
        raise ValueError(f"Expected (36, y, x); received {fields.shape}.")
    mask = radial_edge_mask(fields.shape[1:], settings.edge_threshold)
    positions = _field_positions()
    per_field = []
    values = []
    for index, image in enumerate(fields):
        field_id = index + 1
        value = edge_energy_ratio(
            image, threshold=settings.edge_threshold, edge_mask=mask
        )
        values.append(value)
        grid_row, grid_column = positions[field_id]
        per_field.append(
            {
                "method": method,
                "field_id": field_id,
                "display_grid_row": grid_row,
                "display_grid_column": grid_column,
                "edge_energy": value,
            }
        )
    aggregate = edge_energy_ratio(
        np.sum(fields, axis=0, dtype=np.float64),
        threshold=settings.edge_threshold,
        edge_mask=mask,
    )
    binned = mean_bin(mosaic, settings.binning)
    prominence = prominence_metrics(
        binned,
        tile_shape=(
            fields.shape[1] // settings.binning,
            fields.shape[2] // settings.binning,
        ),
    )
    array = np.asarray(values, dtype=float)
    return (
        {
            "method": method,
            "field_count": fields.shape[0],
            "edge_energy": aggregate,
            "edge_energy_field_mean": float(np.nanmean(array)),
            "edge_energy_field_sd": float(np.nanstd(array)),
            "edge_energy_field_min": float(np.nanmin(array)),
            "edge_energy_field_max": float(np.nanmax(array)),
            **prominence,
            "edge_threshold": settings.edge_threshold,
            "metric_binning": settings.binning,
        },
        per_field,
    )


def format_metrics(row: dict) -> str:
    return (
        f"E_edge={row['edge_energy']:.6f} | "
        f"P+={row['p_positive']:.6e} | "
        f"P-={row['p_negative']:.6e} | "
        f"P±={row['p_magnitude']:.6e}"
    )
