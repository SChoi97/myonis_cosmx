"""Compressed Zarr v2 and OME-TIFF storage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import os
import tempfile

import numpy as np


def _zarr_modules():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as error:
        raise ImportError(
            "zarr<3 and numcodecs are required for preprocessing storage."
        ) from error
    return zarr, Blosc


def compressor(level: int = 5):
    _, Blosc = _zarr_modules()
    return Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)


def corrected_store_path(run_dir: str | Path, acquisition_id: str, plate_name: str) -> Path:
    return (
        Path(run_dir)
        / "corrected"
        / acquisition_id
        / f"{plate_name}.ome.zarr"
    )


def create_corrected_store(
    path: str | Path,
    *,
    wells: Sequence[tuple[int, int]],
    field_shape: tuple[int, int],
    acquisition_metadata: Mapping,
    chunk_shape: tuple[int, int],
    compression_level: int,
):
    zarr, _ = _zarr_modules()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_version=2)
    height, width = field_shape
    images = group.create_dataset(
        "0",
        shape=(len(wells), 36, height, width),
        chunks=(1, 1, min(chunk_shape[0], height), min(chunk_shape[1], width)),
        dtype="f4",
        compressor=compressor(compression_level),
        fill_value=np.nan,
        overwrite=True,
    )
    group.create_dataset("well_row", data=np.asarray([well[0] for well in wells], dtype="i2"))
    group.create_dataset("well_column", data=np.asarray([well[1] for well in wells], dtype="i2"))
    group.create_dataset("field_id", data=np.arange(1, 37, dtype="i2"))
    group.attrs.update(
        {
            "ome_ngff_version": "0.4",
            "name": acquisition_metadata["acquisition_id"],
            "multiscales": [
                {
                    "version": "0.4",
                    "name": acquisition_metadata["acquisition_id"],
                    "axes": [
                        {"name": "well"},
                        {"name": "field"},
                        {"name": "y", "type": "space", "unit": "pixel"},
                        {"name": "x", "type": "space", "unit": "pixel"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ],
            "field_layout": "phenix-field1-bottom-left-snake",
            **dict(acquisition_metadata),
        }
    )
    return group, images


def open_corrected_store(path: str | Path, mode: str = "r"):
    zarr, _ = _zarr_modules()
    return zarr.open_group(str(path), mode=mode)


def well_index(group, row: int, column: int) -> int:
    rows = np.asarray(group["well_row"][:])
    columns = np.asarray(group["well_column"][:])
    matches = np.flatnonzero((rows == row) & (columns == column))
    if matches.size != 1:
        raise KeyError(f"Store has no unique r{row:02d}_c{column:02d}.")
    return int(matches[0])


def read_corrected_well(path: str | Path, row: int, column: int) -> np.ndarray:
    group = open_corrected_store(path)
    return np.asarray(group["0"][well_index(group, row, column)], dtype=np.float32)


def cropped_store_path(run_dir: str | Path, plate_name: str, row: int, column: int) -> Path:
    return (
        Path(run_dir)
        / "registered_crops"
        / plate_name
        / f"r{row:02d}_c{column:02d}.ome.zarr"
    )


def create_cropped_store(path: str | Path, metadata: Mapping):
    zarr, _ = _zarr_modules()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_version=2)
    group.attrs.update({"ome_ngff_version": "0.4", **dict(metadata)})
    return group


def open_cropped_store(path: str | Path, mode: str = "r"):
    zarr, _ = _zarr_modules()
    return zarr.open_group(str(path), mode=mode)


def write_crop(
    group,
    acquisition_id: str,
    image: np.ndarray,
    *,
    metadata: Mapping,
    chunk_shape: tuple[int, int],
    compression_level: int,
) -> None:
    subgroup = group.require_group(acquisition_id)
    height, width = image.shape
    subgroup.create_dataset(
        "0",
        data=np.asarray(image, dtype=np.float32),
        chunks=(min(chunk_shape[0], height), min(chunk_shape[1], width)),
        dtype="f4",
        compressor=compressor(compression_level),
        overwrite=True,
    )
    subgroup.attrs.update(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": acquisition_id,
                    "axes": [
                        {"name": "y", "type": "space", "unit": "pixel"},
                        {"name": "x", "type": "space", "unit": "pixel"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ],
            **dict(metadata),
        }
    )


def read_crop(path: str | Path, acquisition_id: str) -> np.ndarray:
    zarr, _ = _zarr_modules()
    group = zarr.open_group(str(path), mode="r")
    return np.asarray(group[acquisition_id]["0"][:], dtype=np.float32)


def write_patch_level_crop(
    group,
    cohort_id: str,
    field_id: int,
    acquisition_id: str,
    image: np.ndarray,
    *,
    metadata: Mapping,
    chunk_shape: tuple[int, int],
    compression_level: int,
) -> None:
    """Write one variable-sized registered field crop into a well store."""
    subgroup = (
        group.require_group("regions")
        .require_group(cohort_id)
        .require_group(f"field_{field_id:02d}")
        .require_group(acquisition_id)
    )
    height, width = image.shape
    subgroup.create_dataset(
        "0",
        data=np.asarray(image, dtype=np.float32),
        chunks=(min(chunk_shape[0], height), min(chunk_shape[1], width)),
        dtype="f4",
        compressor=compressor(compression_level),
        overwrite=True,
    )
    subgroup.attrs.update(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": acquisition_id,
                    "axes": [
                        {"name": "y", "type": "space", "unit": "pixel"},
                        {"name": "x", "type": "space", "unit": "pixel"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ],
            **dict(metadata),
        }
    )


def read_patch_level_crop(
    path: str | Path, cohort_id: str, field_id: int, acquisition_id: str
) -> np.ndarray:
    group = open_cropped_store(path, mode="r")
    return read_patch_level_crop_group(
        group, cohort_id, field_id, acquisition_id
    )


def read_patch_level_crop_group(
    group, cohort_id: str, field_id: int, acquisition_id: str
) -> np.ndarray:
    return np.asarray(
        group["regions"][cohort_id][f"field_{field_id:02d}"][acquisition_id]["0"][:],
        dtype=np.float32,
    )


def write_float_ome_tiff(
    path: str | Path, image: np.ndarray, *, metadata: Mapping, compression: str
) -> None:
    try:
        import tifffile
    except ImportError as error:
        raise ImportError("tifffile is required to write quantitative patches.") from error
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".ome.tif", dir=path.parent
    )
    os.close(descriptor)
    try:
        tifffile.imwrite(
            temporary_name,
            np.asarray(image, dtype=np.float32),
            ome=True,
            metadata={
                "axes": "YX",
                "Name": str(metadata.get("acquisition_id", path.stem)),
            },
            compression=compression,
        )
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
