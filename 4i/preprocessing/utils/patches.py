"""Patch planning, field-to-patch quarantine, and traceable naming."""

from __future__ import annotations

from pathlib import Path

from .layout import field_box
from .registry import biological_coordinates, PlateSpec


def rectangles_intersect(a, b) -> bool:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return max(ay0, by0) < min(ay1, by1) and max(ax0, bx0) < min(ax1, bx1)


def patch_plan(crop_bounds: tuple[int, int, int, int], patch_size: int) -> list[dict]:
    y0, y1, x0, x1 = crop_bounds
    if (y1 - y0) % patch_size or (x1 - x0) % patch_size:
        raise ValueError("Crop bounds must be divisible by patch_size.")
    rows = (y1 - y0) // patch_size
    columns = (x1 - x0) // patch_size
    plan = []
    index = 1
    for patch_row in range(rows):
        for patch_column in range(columns):
            ref_y = y0 + patch_row * patch_size
            ref_x = x0 + patch_column * patch_size
            plan.append(
                {
                    "patch_id": f"patch_{index:04d}",
                    "patch_index": index,
                    "patch_row": patch_row + 1,
                    "patch_column": patch_column + 1,
                    "reference_y": ref_y,
                    "reference_x": ref_x,
                    "local_y": patch_row * patch_size,
                    "local_x": patch_column * patch_size,
                    "bounds": (ref_y, ref_y + patch_size, ref_x, ref_x + patch_size),
                }
            )
            index += 1
    return plan


def quarantined_patch_ids(plan: list[dict], failed_field_ids: set[int]) -> set[str]:
    failed_boxes = [field_box(field_id) for field_id in sorted(failed_field_ids)]
    return {
        patch["patch_id"]
        for patch in plan
        if any(rectangles_intersect(patch["bounds"], box) for box in failed_boxes)
    }


def patch_filename(
    *,
    plate: PlateSpec,
    row: int,
    column: int,
    acquisition,
    patch: dict,
) -> str:
    biological = biological_coordinates(plate, row, column)
    fluor = acquisition.fluorophore.lower()
    marker = "_".join(
        part for part in acquisition.acquisition_id.split("_")[3:] if part
    ) or acquisition.marker.lower()
    return (
        f"plate-{plate.name}__rep-{biological['repeat']:02d}__"
        f"well-r{row:02d}c{column:02d}__measurement-{acquisition.measurement:02d}__"
        f"source-ch{acquisition.source_channel:02d}__fluor-{fluor}__marker-{marker}__"
        f"{patch['patch_id']}__pr{patch['patch_row']:02d}-pc{patch['patch_column']:02d}__"
        f"refy-{patch['reference_y']:06d}-refx-{patch['reference_x']:06d}.ome.tif"
    )


def patch_level_filename(
    *,
    plate: PlateSpec,
    row: int,
    column: int,
    acquisition,
    patch: dict,
) -> str:
    """Traceable filename for a patch cut inside one registered source field."""
    biological = biological_coordinates(plate, row, column)
    fluor = acquisition.fluorophore.lower()
    marker = "_".join(
        part for part in acquisition.acquisition_id.split("_")[3:] if part
    ) or acquisition.marker.lower()
    cohort = str(patch["cohort_id"]).replace("_", "-")
    return (
        f"plate-{plate.name}__rep-{biological['repeat']:02d}__"
        f"well-r{row:02d}c{column:02d}__field-f{int(patch['field_id']):02d}__"
        f"cohort-{cohort}__measurement-{acquisition.measurement:02d}__"
        f"source-ch{acquisition.source_channel:02d}__fluor-{fluor}__marker-{marker}__"
        f"{patch['patch_id']}__pr{patch['patch_row']:02d}-pc{patch['patch_column']:02d}__"
        f"fieldrefy-{patch['field_reference_y']:04d}-fieldrefx-{patch['field_reference_x']:04d}__"
        f"refy-{patch['reference_y']:06d}-refx-{patch['reference_x']:06d}.ome.tif"
    )
