"""Three-panel full-well, top-right 2 × 2, and seam figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .layout import GRID_SHAPE, field_index_grid


def shared_limits(images, sample_step: int = 8) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(image)[::sample_step, ::sample_step].ravel() for image in images]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [0.5, 99.8])
    return (float(low), float(high if high > low else low + 1.0))


def _boundary_jump(
    image: np.ndarray,
    *,
    orientation: str,
    center_y: int,
    center_x: int,
    along_start: int,
    along_stop: int,
    score_band: int,
) -> float:
    if orientation == "vertical":
        first = image[along_start:along_stop, center_x - score_band : center_x]
        second = image[along_start:along_stop, center_x : center_x + score_band]
        first_profile = np.median(first, axis=1)
        second_profile = np.median(second, axis=1)
    else:
        first = image[center_y - score_band : center_y, along_start:along_stop]
        second = image[center_y : center_y + score_band, along_start:along_stop]
        first_profile = np.median(first, axis=0)
        second_profile = np.median(second, axis=0)
    local = np.concatenate((first.ravel(), second.ravel()))
    local = local[np.isfinite(local)]
    if local.size == 0:
        return float("nan")
    low, high = np.percentile(local, [5.0, 95.0])
    scale = max(float(high - low), np.finfo(np.float32).eps)
    return float(np.median(np.abs(first_profile - second_profile)) / scale)


def select_maximum_disparity_seam(
    mosaics: Mapping[str, np.ndarray],
    *,
    along_half_width: int,
    across_half_width: int,
    score_band: int,
) -> dict:
    """Score all 60 internal borders and select the clearest method difference."""
    shape = next(iter(mosaics.values())).shape
    if any(mosaic.shape != shape for mosaic in mosaics.values()):
        raise ValueError("All method mosaics must have the same shape.")
    rows, columns = GRID_SHAPE
    tile_height, tile_width = shape[0] // rows, shape[1] // columns
    if not 1 <= score_band <= min(tile_height, tile_width) // 4:
        raise ValueError(f"Invalid seam score band {score_band}.")
    candidates = []
    for row in range(rows):
        y0, y1 = row * tile_height, (row + 1) * tile_height
        for boundary in range(1, columns):
            center_y = int((row + 0.5) * tile_height)
            center_x = boundary * tile_width
            jumps = {
                method: _boundary_jump(
                    image,
                    orientation="vertical",
                    center_y=center_y,
                    center_x=center_x,
                    along_start=y0,
                    along_stop=y1,
                    score_band=score_band,
                )
                for method, image in mosaics.items()
            }
            candidates.append(("vertical", row, boundary, center_y, center_x, jumps))
    for boundary in range(1, rows):
        for column in range(columns):
            center_y = boundary * tile_height
            center_x = int((column + 0.5) * tile_width)
            x0, x1 = column * tile_width, (column + 1) * tile_width
            jumps = {
                method: _boundary_jump(
                    image,
                    orientation="horizontal",
                    center_y=center_y,
                    center_x=center_x,
                    along_start=x0,
                    along_stop=x1,
                    score_band=score_band,
                )
                for method, image in mosaics.items()
            }
            candidates.append(
                ("horizontal", boundary, column, center_y, center_x, jumps)
            )

    def score(candidate) -> float:
        jumps = candidate[-1]
        finite = np.asarray(
            [value for value in jumps.values() if np.isfinite(value)], dtype=float
        )
        if finite.size < 2:
            return -np.inf
        spread = float(np.ptp(finite))
        original_harmony = abs(
            jumps.get("Original", finite[0]) - jumps.get("Harmony-like", finite[-1])
        )
        return original_harmony + 0.25 * spread

    selected = max(candidates, key=score)
    orientation, first, second, center_y, center_x, jumps = selected
    layout = field_index_grid()
    if orientation == "vertical":
        row, boundary = first, second
        field_pair = (
            int(layout[row, boundary - 1] + 1),
            int(layout[row, boundary] + 1),
        )
        bounds = (
            max(0, center_y - along_half_width),
            min(shape[0], center_y + along_half_width),
            max(0, center_x - across_half_width),
            min(shape[1], center_x + across_half_width),
        )
        seam_position = center_x - bounds[2]
        description = f"fields {field_pair[0]} left / {field_pair[1]} right"
    else:
        boundary, column = first, second
        field_pair = (
            int(layout[boundary - 1, column] + 1),
            int(layout[boundary, column] + 1),
        )
        bounds = (
            max(0, center_y - across_half_width),
            min(shape[0], center_y + across_half_width),
            max(0, center_x - along_half_width),
            min(shape[1], center_x + along_half_width),
        )
        seam_position = center_y - bounds[0]
        description = f"fields {field_pair[0]} top / {field_pair[1]} bottom"
    return {
        "orientation": orientation,
        "field_pair": field_pair,
        "description": description,
        "bounds": bounds,
        "seam_position": seam_position,
        "method_discontinuity": jumps,
        "disparity_score": score(selected),
        "score_band": score_band,
    }


def _titles(metric_rows: list[dict]) -> dict[str, str]:
    return {
        row["method"]: (
            f"{row['method']}\nE_edge={row['edge_energy']:.5f} | "
            f"P+={row['p_positive']:.3e}\nP-={row['p_negative']:.3e} | "
            f"P±={row['p_magnitude']:.3e}"
        )
        for row in metric_rows
    }


def _save_well_panels(
    mosaics: Mapping[str, np.ndarray],
    *,
    title: str,
    view_label: str,
    display_limits: tuple[float, float],
    metric_rows: list[dict],
    output_path: Path,
    display_downsample: int,
    dpi: int,
) -> None:
    low, high = display_limits
    titles = _titles(metric_rows)
    fig, axes = plt.subplots(
        1, len(mosaics), figsize=(6.2 * len(mosaics), 6.5), squeeze=False
    )
    for ax, (method, mosaic) in zip(axes[0], mosaics.items()):
        ax.imshow(
            mosaic[::display_downsample, ::display_downsample],
            cmap="gray",
            vmin=low,
            vmax=high,
        )
        ax.set_title(titles[method], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title}\n{view_label} | shared display limits", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_full_well(
    mosaics: Mapping[str, np.ndarray],
    *,
    title: str,
    metric_rows: list[dict],
    output_path: Path,
    display_downsample: int,
    dpi: int,
) -> None:
    _save_well_panels(
        mosaics,
        title=title,
        view_label="Full 6 × 6 well",
        display_limits=shared_limits(mosaics.values()),
        metric_rows=metric_rows,
        output_path=output_path,
        display_downsample=display_downsample,
        dpi=dpi,
    )


def save_zoom_2x2(
    mosaics: Mapping[str, np.ndarray],
    *,
    title: str,
    metric_rows: list[dict],
    output_path: Path,
    display_downsample: int,
    dpi: int,
) -> None:
    """Show the top-right four fields using the full-well display settings."""
    rows, columns = GRID_SHAPE
    crops = {}
    for method, mosaic in mosaics.items():
        tile_height, tile_width = mosaic.shape[0] // rows, mosaic.shape[1] // columns
        crops[method] = mosaic[: 2 * tile_height, (columns - 2) * tile_width :]
    _save_well_panels(
        crops,
        title=title,
        view_label="Top-right 2 × 2 fields (6, 7 / 9, 8)",
        display_limits=shared_limits(mosaics.values()),
        metric_rows=metric_rows,
        output_path=output_path,
        display_downsample=display_downsample,
        dpi=dpi,
    )


def save_zoom(
    mosaics: Mapping[str, np.ndarray],
    *,
    title: str,
    metric_rows: list[dict],
    output_path: Path,
    along_half_width: int,
    across_half_width: int,
    score_band: int,
    dpi: int,
) -> dict:
    seam = select_maximum_disparity_seam(
        mosaics,
        along_half_width=along_half_width,
        across_half_width=across_half_width,
        score_band=score_band,
    )
    y0, y1, x0, x1 = seam["bounds"]
    crops = {method: image[y0:y1, x0:x1] for method, image in mosaics.items()}
    low, high = shared_limits(crops.values(), sample_step=2)
    titles = _titles(metric_rows)
    fig, axes = plt.subplots(
        1, len(crops), figsize=(6.2 * len(crops), 5.5), squeeze=False
    )
    for ax, (method, crop) in zip(axes[0], crops.items()):
        ax.imshow(crop, cmap="gray", vmin=low, vmax=high)
        if seam["orientation"] == "vertical":
            ax.axvline(seam["seam_position"] - 0.5, color="cyan", linewidth=0.9)
        else:
            ax.axhline(seam["seam_position"] - 0.5, color="cyan", linewidth=0.9)
        ax.set_title(
            f"{titles[method]}\nBoundary={seam['method_discontinuity'][method]:.4f}",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle(
        f"{title}\nMaximum-disparity edge: {seam['description']} | "
        f"score={seam['disparity_score']:.4f}",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return seam
