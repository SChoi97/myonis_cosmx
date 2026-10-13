"""Canonical acquisition and plate registry for the August 2026 4i run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable


LOCAL_DATA_ROOT = Path(
    "/Volumes/lab-tedescos/home/users/chois1/HTS/phoenix_plus/4i_experiment"
)
CLUSTER_DATA_ROOT = Path(
    "/nemo/lab/tedescos/home/users/chois1/HTS/phoenix_plus/4i_experiment"
)


@dataclass(frozen=True)
class PlateSpec:
    slot: int
    name: str
    repeat_left: int
    repeat_right: int


PLATES = (
    PlateSpec(1, "sc_rep5-6", 5, 6),
    PlateSpec(2, "sc_rep7-8", 7, 8),
)


@dataclass(frozen=True)
class AcquisitionSpec:
    acquisition_id: str
    evaluation_name: str
    measurement: int
    channel_folder: str
    source_channel: int
    fluorophore: str
    marker: str
    is_dapi: bool = False

    def plate_directory(self, plate: PlateSpec, data_root: Path) -> Path:
        return (
            data_root
            / f"{plate.name}_measurement_{self.measurement}"
            / "max_projection"
            / self.channel_folder
        )

    def as_row(self) -> dict:
        return asdict(self)


NON_DAPI = (
    (1, "channel_2_alexa_488", 2, "488", "eMYHC"),
    (1, "channel_3_alexa_555_isabel", 3, "555", "Lamin A/C"),
    (1, "channel_4_alexa_647", 4, "647", "53BP1"),
    (2, "channel_2_alexa_488", 2, "488", "RNA Pol II"),
    (2, "channel_3_alexa_555_isabel", 3, "555", "MyHC"),
    (2, "channel_4_alexa_647", 4, "647", "ATF3"),
    (3, "channel_2_alexa_488", 2, "488", "p53 DO-1"),
    (3, "channel_3_alexa_555_isabel", 3, "555", "MyoG"),
    (3, "channel_4_alexa_647", 4, "647", "BAF"),
    (4, "channel_2_alexa_488", 2, "488", "YAP1"),
    (4, "channel_3_alexa_555_isabel", 3, "555", "H3K27ac"),
    (4, "channel_4_alexa_647", 4, "647", "H3K27me3"),
    (5, "channel_2_alexa_488", 2, "488", "a-tubulin"),
    (5, "channel_3_alexa_555_isabel", 3, "555", "SC35"),
    (5, "channel_4_alexa_647", 4, "647", "c-CASP3"),
    (6, "channel_2_alexa_647", 2, "647", "cGAS"),
    (7, "channel_2_alexa_647", 2, "647", "p-MLKL"),
    (8, "channel_2_alexa_647", 2, "647", "PTEN"),
    # Measurement 9 Alexa 488 is intentionally excluded; its Alexa 647 remains.
    (9, "channel_3_alexa_647", 3, "647", "p-4EBP1"),
    (10, "channel_2_alexa_488", 2, "488", "CKAP4"),
    (10, "channel_3_alexa_555_isabel", 3, "555", "GM130"),
    (10, "channel_4_alexa_647", 4, "647", "c-JUN"),
    (11, "channel_2_alexa_488", 2, "488", "HSP60"),
    (11, "channel_3_alexa_555_isabel", 3, "555", "b-Actin"),
    (11, "channel_4_alexa_647", 4, "647", "CTGF"),
    (12, "channel_2_alexa_488", 2, "488", "ATF6"),
    (12, "channel_3_alexa_555_isabel", 3, "555", "p-S6RP"),
    (12, "channel_4_alexa_647", 4, "647", "PEX14"),
)


def safe_name(value: str) -> str:
    return "_".join(
        part
        for part in re.sub(r"[^A-Za-z0-9]+", "_", value).lower().split("_")
        if part
    )


def _make_spec(
    measurement: int,
    channel_folder: str,
    source_channel: int,
    fluorophore: str,
    marker: str,
    *,
    is_dapi: bool = False,
) -> AcquisitionSpec:
    marker_slug = safe_name(marker)
    suffix = "dapi" if is_dapi else f"{fluorophore}_{marker_slug}"
    return AcquisitionSpec(
        acquisition_id=f"measurement_{measurement:02d}_{suffix}",
        evaluation_name=f"measurement_{measurement}_{channel_folder}",
        measurement=measurement,
        channel_folder=channel_folder,
        source_channel=source_channel,
        fluorophore=fluorophore,
        marker=marker,
        is_dapi=is_dapi,
    )


def acquisition_registry() -> tuple[AcquisitionSpec, ...]:
    by_measurement: dict[int, list[AcquisitionSpec]] = {i: [] for i in range(1, 13)}
    for measurement in range(1, 13):
        by_measurement[measurement].append(
            _make_spec(
                measurement,
                "channel_1_dapi",
                1,
                "DAPI",
                "DAPI",
                is_dapi=True,
            )
        )
    for values in NON_DAPI:
        by_measurement[values[0]].append(_make_spec(*values))
    return tuple(
        acquisition
        for measurement in range(1, 13)
        for acquisition in by_measurement[measurement]
    )


ACQUISITIONS = acquisition_registry()


def resolve_data_root(supplied: str | Path | None = None) -> Path:
    candidates = []
    if supplied:
        candidates.append(Path(supplied).expanduser())
    candidates.extend((CLUSTER_DATA_ROOT, LOCAL_DATA_ROOT))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return Path(supplied).expanduser() if supplied else CLUSTER_DATA_ROOT


def acquisition_by_id(acquisition_id: str) -> AcquisitionSpec:
    matches = [item for item in ACQUISITIONS if item.acquisition_id == acquisition_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown acquisition_id {acquisition_id!r}.")
    return matches[0]


def dapi_for_measurement(measurement: int) -> AcquisitionSpec:
    return next(
        item
        for item in ACQUISITIONS
        if item.measurement == measurement and item.is_dapi
    )


def acquisitions_for_measurement(measurement: int) -> tuple[AcquisitionSpec, ...]:
    return tuple(item for item in ACQUISITIONS if item.measurement == measurement)


def well_tasks() -> tuple[tuple[PlateSpec, int, int], ...]:
    return tuple(
        (plate, row, column)
        for plate in PLATES
        for row in range(2, 8)
        for column in range(2, 12)
    )


def configured_well_tasks(config) -> tuple[tuple[PlateSpec, int, int], ...]:
    """Return all production wells or the exact four-well test cohort."""
    if not getattr(config, "test_mode", False):
        return well_tasks()
    plates = {plate.name: plate for plate in PLATES}
    return tuple(
        (plates[item.plate_name], item.row, item.column)
        for item in config.test_wells
    )


def wells_for_plate(config, plate: PlateSpec) -> tuple[tuple[int, int], ...]:
    """Return the configured coordinates belonging to one physical plate."""
    return tuple(
        (row, column)
        for selected_plate, row, column in configured_well_tasks(config)
        if selected_plate.name == plate.name
    )


def biological_coordinates(
    plate: PlateSpec, row: int, column: int
) -> dict[str, int]:
    if not 2 <= row <= 7 or not 2 <= column <= 11:
        raise ValueError(f"Unexpected occupied well r{row:02d}_c{column:02d}.")
    left = column <= 6
    return {
        "repeat": plate.repeat_left if left else plate.repeat_right,
        "cell_line_row": row - 1,
        "well_within_repeat": column - 1 if left else column - 6,
    }


def ensure_unique(items: Iterable[AcquisitionSpec] = ACQUISITIONS) -> None:
    items = tuple(items)
    ids = [item.acquisition_id for item in items]
    evaluation_names = [item.evaluation_name for item in items]
    if len(ids) != len(set(ids)) or len(evaluation_names) != len(set(evaluation_names)):
        raise ValueError("Acquisition IDs and evaluation names must be unique.")


ensure_unique()
