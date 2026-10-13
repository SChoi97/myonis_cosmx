"""Verified Phenix field layout and mosaic helpers."""

from __future__ import annotations

import re

import numpy as np


GRID_SHAPE = (6, 6)
FIELD_PATTERN = re.compile(
    r"^r(?P<row>\d+)_c(?P<column>\d+)_f(?P<field>\d+)_"
    r"t(?P<timepoint>\d+)_ch(?P<channel>\d+)\.tiff?$",
    flags=re.IGNORECASE,
)

# Manually verified Phenix acquisition order, displayed top to bottom.
FIELD_ID_GRID = np.array(
    [
        [2, 3, 4, 5, 6, 7],
        [13, 12, 11, 10, 9, 8],
        [14, 15, 16, 17, 18, 19],
        [25, 24, 23, 22, 21, 20],
        [26, 27, 28, 29, 30, 31],
        [1, 36, 35, 34, 33, 32],
    ],
    dtype=int,
)


def field_index_grid() -> np.ndarray:
    """Return zero-based field indices in displayed top-to-bottom positions."""
    return FIELD_ID_GRID - 1


def tile_fields(field_stack: np.ndarray) -> np.ndarray:
    """Tile fields 1..36 into the verified 6 × 6 Phenix snake layout."""
    fields = np.asarray(field_stack)
    if fields.ndim != 3:
        raise ValueError(f"Expected (field, y, x); received {fields.shape}.")
    if fields.shape[0] != int(np.prod(GRID_SHAPE)):
        raise ValueError(f"Expected 36 fields; received {fields.shape[0]}.")
    grid = fields[field_index_grid()]
    return grid.transpose(0, 2, 1, 3).reshape(
        GRID_SHAPE[0] * fields.shape[1],
        GRID_SHAPE[1] * fields.shape[2],
    )
