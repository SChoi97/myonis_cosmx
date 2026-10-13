"""Static registration QC figures used by HTML and PDF reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layout import FIELD_ID_GRID
from .registration import robust_normalize


def _overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    green = robust_normalize(fixed)
    red = robust_normalize(moving)
    image = np.zeros((*green.shape, 3), dtype=np.float32)
    image[..., 0] = red
    image[..., 1] = green
    return image


def save_registration_overlay(
    path: str | Path,
    fixed: np.ndarray,
    moving: np.ndarray,
    registered: np.ndarray,
    *,
    title: str,
    subtitle: str,
    downsample: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, downsample)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(_overlay(fixed[::step, ::step], moving[::step, ::step]))
    axes[0].set_title("Before")
    axes[1].imshow(_overlay(fixed[::step, ::step], registered[::step, ::step]))
    axes[1].set_title("After")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{title}\n{subtitle}\nReference = green | moving = red | overlap = yellow")
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_field_qc_heatmaps(path: str | Path, rows: list[dict], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    measurements = sorted({int(row["measurement"]) for row in rows})
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    cmap = ListedColormap(["#49a078", "#b8b8b8", "#d1495b"])
    for axis, measurement in zip(axes.flat, measurements):
        subset = [row for row in rows if int(row["measurement"]) == measurement]
        status_grid = np.ones((6, 6), dtype=int)
        dy = np.zeros((6, 6), dtype=float)
        dx = np.zeros((6, 6), dtype=float)
        for row in subset:
            grid_row = int(row["grid_row"]) - 1
            grid_column = int(row["grid_column"]) - 1
            status_grid[grid_row, grid_column] = {
                "pass": 0,
                "not_evaluable": 1,
                "fail": 2,
            }[row["status"]]
            dy[grid_row, grid_column] = float(
                row.get("registration_shift_y", row.get("residual_shift_y")) or 0
            )
            dx[grid_row, grid_column] = float(
                row.get("registration_shift_x", row.get("residual_shift_x")) or 0
            )
        axis.imshow(status_grid, cmap=cmap, vmin=0, vmax=2)
        yy, xx = np.mgrid[0:6, 0:6]
        if any("registration_shift_y" in row for row in subset):
            maximum_vector = float(np.nanmax(np.hypot(dx, dy)))
            vector_scale = max(15.0, maximum_vector / 0.8) if np.isfinite(maximum_vector) else 15.0
        else:
            vector_scale = 15.0
        axis.quiver(
            xx,
            yy,
            dx,
            dy,
            color="black",
            angles="xy",
            scale_units="xy",
            scale=vector_scale,
        )
        for grid_row in range(6):
            for grid_column in range(6):
                axis.text(
                    grid_column,
                    grid_row,
                    str(int(FIELD_ID_GRID[grid_row, grid_column])),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if status_grid[grid_row, grid_column] == 2 else "black",
                )
        axis.set_title(f"Measurement {measurement}")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(measurements) :]:
        axis.axis("off")
    vector_label = (
        "per-field applied translation"
        if any("registration_shift_y" in row for row in rows)
        else "local residual"
    )
    figure.suptitle(
        title
        + f"\nGreen=pass | grey=not evaluable | red=failed; arrows={vector_label}"
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_patch_level_region_heatmaps(
    path: str | Path, regions: list[dict], *, title: str
) -> None:
    """Summarize common overlap, retained patch count, and region status by field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = sorted({str(row["cohort_id"]) for row in regions})
    figure, axes = plt.subplots(len(cohorts), 3, figsize=(12, 4 * len(cohorts)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for cohort_index, cohort_id in enumerate(cohorts):
        subset = [row for row in regions if str(row["cohort_id"]) == cohort_id]
        status = np.ones((6, 6), dtype=int)
        overlap = np.full((6, 6), np.nan, dtype=float)
        patches = np.zeros((6, 6), dtype=float)
        for row in subset:
            grid_row = int(row["grid_row"]) - 1
            grid_column = int(row["grid_column"]) - 1
            status[grid_row, grid_column] = 0 if row["status"] == "complete" else 1
            overlap[grid_row, grid_column] = float(row["common_overlap_fraction"])
            patches[grid_row, grid_column] = float(row["patch_count_retained"])
        cohort_axes = axes[cohort_index]
        cohort_axes[0].imshow(status, cmap=ListedColormap(["#49a078", "#d1495b"]), vmin=0, vmax=1)
        cohort_axes[1].imshow(overlap, cmap="viridis", vmin=0, vmax=1)
        cohort_axes[2].imshow(patches, cmap="Blues", vmin=0)
        for axis, values, label in zip(
            cohort_axes,
            (status, overlap, patches),
            ("status", "common overlap", "retained 512 patches"),
        ):
            for grid_row in range(6):
                for grid_column in range(6):
                    field_id = int(FIELD_ID_GRID[grid_row, grid_column])
                    value = values[grid_row, grid_column]
                    shown = field_id if label == "status" else (
                        f"{value:.2f}" if label == "common overlap" else f"{int(value)}"
                    )
                    axis.text(grid_column, grid_row, shown, ha="center", va="center", fontsize=8)
            axis.set_title(f"{cohort_id}: {label}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_crop_outline(
    path: str | Path,
    reference: np.ndarray,
    common_bounds: tuple[int, int, int, int],
    crop_bounds: tuple[int, int, int, int],
    *,
    title: str,
    downsample: int,
) -> None:
    """Show the common-valid rectangle and patch-divisible final crop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, downsample)
    shown = robust_normalize(reference)[::step, ::step]
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(shown, cmap="gray")
    for bounds, color, label in (
        (common_bounds, "cyan", "maximal common valid overlap"),
        (crop_bounds, "yellow", "centered patch-divisible crop"),
    ):
        y0, y1, x0, x1 = bounds
        axis.add_patch(
            Rectangle(
                (x0 / step, y0 / step),
                (x1 - x0) / step,
                (y1 - y0) / step,
                fill=False,
                edgecolor=color,
                linewidth=2,
                label=label,
            )
        )
    axis.legend(loc="upper right", facecolor="black", labelcolor="white")
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
