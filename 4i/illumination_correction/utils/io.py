"""Index paired Phenix channel directories without loading image pixels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import imageio.v3 as iio
import numpy as np

from .layout import FIELD_PATTERN


LOCAL_PREFIX = Path("/Volumes/lab-tedescos")
CLUSTER_PREFIX = Path("/nemo/lab/tedescos")
FieldKey = tuple[int, int, int]
WellKey = tuple[int, int]


@dataclass(frozen=True)
class MarkerSpec:
    """One marker represented by the same channel on two plates."""

    name: str
    display_label: str
    plate_paths: tuple[Path, Path]


@dataclass
class ChannelIndex:
    """Filename-derived coordinate index for one marker and plate."""

    marker_name: str
    plate_slot: int
    plate_name: str
    directory: Path
    channel: int
    by_key: dict[FieldKey, Path]
    complete_wells: set[WellKey]


def safe_name(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.lower().split("_") if part)


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a data folder on either Nemo or the mounted lab volume."""
    supplied = Path(path).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    supplied_text = str(supplied)
    alternatives = []
    if supplied_text.startswith(str(LOCAL_PREFIX) + "/"):
        alternatives.append(
            Path(str(CLUSTER_PREFIX) + supplied_text[len(str(LOCAL_PREFIX)) :])
        )
    if supplied_text.startswith(str(CLUSTER_PREFIX) + "/"):
        alternatives.append(
            Path(str(LOCAL_PREFIX) + supplied_text[len(str(CLUSTER_PREFIX)) :])
        )
    for alternative in alternatives:
        if alternative.is_dir():
            return alternative.resolve()
    tried = [supplied, *alternatives]
    raise FileNotFoundError(
        "Channel directory was not found. Tried:\n  "
        + "\n  ".join(str(candidate) for candidate in tried)
    )


def _measurement_name(channel_directory: Path) -> str:
    measurement_directory = channel_directory.parents[1].name
    match = re.search(r"measurement[ _-]*(\d+)", measurement_directory, re.I)
    if match is None:
        raise ValueError(
            f"Could not infer a measurement number from {measurement_directory!r}."
        )
    return f"measurement_{int(match.group(1))}"


def infer_marker_name(channel_directory: str | Path) -> str:
    """Return names such as ``measurement_1_channel_1_dapi``."""
    directory = Path(channel_directory)
    return safe_name(f"{_measurement_name(directory)}_{directory.name}")


def infer_plate_name(channel_directory: Path) -> str:
    measurement_directory = channel_directory.parents[1].name
    match = re.search(r"^(.*?)[ _-]*measurement[ _-]*\d+", measurement_directory, re.I)
    prefix = match.group(1) if match else measurement_directory
    return safe_name(prefix)


def marker_spec(
    plate_1_path: str | Path,
    plate_2_path: str | Path,
    *,
    name: str | None = None,
    display_label: str | None = None,
) -> MarkerSpec:
    """Resolve and validate a pair of plate paths for one marker."""
    paths = (resolve_data_path(plate_1_path), resolve_data_path(plate_2_path))
    inferred = tuple(infer_marker_name(path) for path in paths)
    if name is None and inferred[0] != inferred[1]:
        raise ValueError(
            "The paired paths infer different marker names: "
            f"{inferred[0]!r} and {inferred[1]!r}. Supply an explicit name."
        )
    resolved_name = safe_name(name) if name else inferred[0]
    if not resolved_name:
        raise ValueError("Marker name cannot be empty.")
    return MarkerSpec(
        name=resolved_name,
        display_label=display_label or resolved_name,
        plate_paths=paths,
    )


def read_marker_config(path: str | Path) -> list[MarkerSpec]:
    """Read a TSV/CSV with columns name, plate_1_path, and plate_2_path."""
    config_path = Path(path).expanduser().resolve()
    delimiter = "\t" if config_path.suffix.lower() in {".tsv", ".txt"} else ","
    with config_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    required = {"plate_1_path", "plate_2_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"{config_path} must contain plate_1_path and plate_2_path columns."
        )
    return [
        marker_spec(
            row["plate_1_path"],
            row["plate_2_path"],
            name=(row.get("name") or None),
            display_label=(row.get("display_label") or None),
        )
        for row in rows
    ]


def index_channel(spec: MarkerSpec, plate_slot: int) -> ChannelIndex:
    """Index all (row, column, field) TIFF coordinates for one plate."""
    directory = spec.plate_paths[plate_slot - 1]
    by_key: dict[FieldKey, Path] = {}
    channels = set()
    well_fields: dict[WellKey, set[int]] = {}
    for path in sorted(directory.glob("*.tif*")):
        match = FIELD_PATTERN.match(path.name)
        if match is None:
            continue
        values = {key: int(value) for key, value in match.groupdict().items()}
        key = (values["row"], values["column"], values["field"])
        if key in by_key:
            raise ValueError(
                f"Duplicate coordinate {key} in {directory}: "
                f"{by_key[key].name} and {path.name}."
            )
        by_key[key] = path
        channels.add(values["channel"])
        well_fields.setdefault(key[:2], set()).add(key[2])
    if not by_key:
        raise FileNotFoundError(f"No matching Phenix TIFFs found in {directory}.")
    if len(channels) != 1:
        raise ValueError(f"Expected one channel in {directory}; found {channels}.")
    required_fields = set(range(1, 37))
    complete_wells = {
        well for well, fields in well_fields.items() if fields == required_fields
    }
    return ChannelIndex(
        marker_name=spec.name,
        plate_slot=plate_slot,
        plate_name=infer_plate_name(directory),
        directory=directory,
        channel=next(iter(channels)),
        by_key=by_key,
        complete_wells=complete_wells,
    )


def index_markers(
    markers: Sequence[MarkerSpec],
) -> dict[str, tuple[ChannelIndex, ChannelIndex]]:
    """Index all marker paths and reject duplicate output names."""
    names = [marker.name for marker in markers]
    if len(set(names)) != len(names):
        raise ValueError(f"Marker output names must be unique; received {names}.")
    indexed = {}
    for marker_number, marker in enumerate(markers, start=1):
        print(f"Indexing marker {marker_number}/{len(markers)}: {marker.name}")
        indexed[marker.name] = (
            index_channel(marker, 1),
            index_channel(marker, 2),
        )
    return indexed


def load_paths(paths: Sequence[Path]) -> np.ndarray:
    """Load a same-shaped list of 2-D TIFFs into one field stack."""
    if not paths:
        raise ValueError("Cannot load an empty image path list.")
    images = [np.asarray(iio.imread(path)) for path in paths]
    invalid = [str(path) for path, image in zip(paths, images) if image.ndim != 2]
    if invalid:
        raise ValueError(f"Expected 2-D images; invalid files: {invalid[:5]}.")
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"Images have inconsistent shapes: {sorted(shapes)}.")
    return np.stack(images)


def paths_for_well(index: ChannelIndex, well: WellKey) -> list[Path]:
    """Return field paths 1..36 for one complete well."""
    paths = [index.by_key.get((*well, field)) for field in range(1, 37)]
    if any(path is None for path in paths):
        missing = [field for field, path in enumerate(paths, start=1) if path is None]
        raise FileNotFoundError(
            f"Incomplete well r{well[0]:02d}_c{well[1]:02d} in "
            f"{index.directory}; missing fields {missing}."
        )
    return [path for path in paths if path is not None]
