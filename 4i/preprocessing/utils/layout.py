"""Verified Phenix filename parsing and 6 x 6 field layout."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np


GRID_SHAPE = (6, 6)
FIELD_SHAPE = (1080, 1080)
FIELD_ID_GRID = np.array(
    [
        [2, 3, 4, 5, 6, 7],
        [13, 12, 11, 10, 9, 8],
        [14, 15, 16, 17, 18, 19],
        [25, 24, 23, 22, 21, 20],
        [26, 27, 28, 29, 30, 31],
        [1, 36, 35, 34, 33, 32],
    ],
    dtype=np.int16,
)
FIELD_PATTERN = re.compile(
    r"^r(?P<row>\d+)_c(?P<column>\d+)_f(?P<field>\d+)_"
    r"t(?P<timepoint>\d+)_ch(?P<channel>\d+)\.tiff?$",
    re.IGNORECASE,
)


def parse_field_path(path: str | Path) -> dict[str, int]:
    path = Path(path)
    match = FIELD_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Not a supported Phenix field filename: {path.name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def index_channel_directory(directory: str | Path) -> dict[tuple[int, int, int], Path]:
    directory = Path(directory)
    indexed: dict[tuple[int, int, int], Path] = {}
    channels = set()
    for path in sorted(directory.glob("*.tif*")):
        try:
            values = parse_field_path(path)
        except ValueError:
            continue
        key = (values["row"], values["column"], values["field"])
        if key in indexed:
            raise ValueError(f"Duplicate field coordinate {key} in {directory}.")
        indexed[key] = path
        channels.add(values["channel"])
    if not indexed:
        raise FileNotFoundError(f"No Phenix TIFFs found in {directory}.")
    if len(channels) != 1:
        raise ValueError(f"Expected one source channel in {directory}; found {channels}.")
    return indexed


def complete_wells(
    indexed: dict[tuple[int, int, int], Path]
) -> tuple[tuple[int, int], ...]:
    fields: dict[tuple[int, int], set[int]] = {}
    for row, column, field in indexed:
        fields.setdefault((row, column), set()).add(field)
    required = set(range(1, 37))
    return tuple(sorted(well for well, present in fields.items() if present == required))


def paths_for_well(
    indexed: dict[tuple[int, int, int], Path], row: int, column: int
) -> list[Path]:
    paths = [indexed.get((row, column, field)) for field in range(1, 37)]
    if any(path is None for path in paths):
        missing = [field for field, path in enumerate(paths, 1) if path is None]
        raise FileNotFoundError(f"r{row:02d}_c{column:02d} missing fields {missing}.")
    return [path for path in paths if path is not None]


def tile_fields(field_stack: np.ndarray) -> np.ndarray:
    fields = np.asarray(field_stack)
    if fields.ndim != 3 or fields.shape[0] != 36:
        raise ValueError(f"Expected (36, y, x); received {fields.shape}.")
    grid = fields[FIELD_ID_GRID - 1]
    return grid.transpose(0, 2, 1, 3).reshape(
        GRID_SHAPE[0] * fields.shape[1], GRID_SHAPE[1] * fields.shape[2]
    )


def field_box(field_id: int, field_shape: tuple[int, int] = FIELD_SHAPE) -> tuple[int, int, int, int]:
    position = np.argwhere(FIELD_ID_GRID == field_id)
    if position.shape != (1, 2):
        raise ValueError(f"field_id must be in 1..36; received {field_id}.")
    grid_row, grid_column = map(int, position[0])
    height, width = field_shape
    return (
        grid_row * height,
        (grid_row + 1) * height,
        grid_column * width,
        (grid_column + 1) * width,
    )


def assert_expected_wells(wells: Iterable[tuple[int, int]]) -> None:
    wells = set(wells)
    expected = {(row, column) for row in range(2, 8) for column in range(2, 12)}
    if wells != expected:
        missing = sorted(expected - wells)
        unexpected = sorted(wells - expected)
        raise ValueError(
            f"Expected 60 complete wells; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}."
        )

