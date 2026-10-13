#!/usr/bin/env python3
"""Copy representative raw max-projection TIFFs for manual clip-limit review.

The script selects the 28 unique marker acquisitions in the canonical August
2026 4i registry plus DAPI from measurement 12.  ``N`` images are sampled per
acquisition, approximately equally between the two plates and from as many
distinct occupied wells as possible.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = SCRIPT_DIR.parent / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from utils.registry import ACQUISITIONS, PLATES  # noqa: E402


MEASUREMENT_DIRECTORY_PATTERN = re.compile(
    r"^(?P<plate>.+)_measurement_(?P<measurement>\d+)$", re.IGNORECASE
)
FIELD_FILENAME_PATTERN = re.compile(
    r"^r(?P<row>\d+)_c(?P<column>\d+)_f(?P<field>\d+)_"
    r"t(?P<timepoint>\d+)_ch(?P<channel>\d+)\.tiff?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceImage:
    path: Path
    plate: str
    measurement: int
    row: int
    column: int
    field: int
    timepoint: int
    channel: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample raw TIFF max projections for every unique 4i marker and "
            "measurement-12 DAPI."
        )
    )
    parser.add_argument(
        "--max-projection-folders",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "The 24 max_projection directories (measurements 1-12 for both "
            "plates). Each directory must contain channel_* subdirectories."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="New or empty output directory in which marker folders are made.",
    )
    parser.add_argument(
        "--samples-per-marker",
        type=int,
        default=20,
        help="Total TIFFs copied per marker across both plates (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260821,
        help="Random seed for reproducible sampling (default: 20260821).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without creating or copying files.",
    )
    return parser


def selected_acquisitions():
    """Return 28 non-DAPI markers and DAPI from measurement 12."""
    selected = tuple(
        item
        for item in ACQUISITIONS
        if (not item.is_dapi) or item.measurement == 12
    )
    non_dapi = tuple(item for item in selected if not item.is_dapi)
    dapi = tuple(item for item in selected if item.is_dapi)
    marker_names = [item.marker.casefold() for item in non_dapi]
    if len(non_dapi) != 28 or len(marker_names) != len(set(marker_names)):
        raise RuntimeError(
            "The canonical registry must contain exactly 28 unique non-DAPI markers."
        )
    if len(dapi) != 1 or dapi[0].measurement != 12:
        raise RuntimeError("Expected exactly one selected DAPI: measurement 12.")
    return selected


def parse_max_projection_folder(path: Path) -> tuple[str, int]:
    path = path.expanduser().absolute()
    if not path.is_dir():
        raise FileNotFoundError(f"Max-projection folder does not exist: {path}")
    if path.name != "max_projection":
        raise ValueError(
            f"Expected a directory named 'max_projection'; received {path}"
        )
    match = MEASUREMENT_DIRECTORY_PATTERN.match(path.parent.name)
    if match is None:
        raise ValueError(
            "Could not infer plate and measurement from parent directory "
            f"{path.parent.name!r}. Expected '<plate>_measurement_<number>'."
        )
    return match.group("plate"), int(match.group("measurement"))


def index_max_projection_folders(
    folders: Iterable[Path],
) -> dict[tuple[str, int], Path]:
    indexed: dict[tuple[str, int], Path] = {}
    for supplied in folders:
        folder = supplied.expanduser().absolute()
        key = parse_max_projection_folder(folder)
        if key in indexed and indexed[key] != folder:
            raise ValueError(
                f"Duplicate max_projection input for {key}: "
                f"{indexed[key]} and {folder}"
            )
        indexed[key] = folder
    return indexed


def validate_required_folders(
    indexed_roots: dict[tuple[str, int], Path], acquisitions
) -> None:
    expected_plates = {plate.name for plate in PLATES}
    supplied_plates = {plate for plate, _ in indexed_roots}
    if supplied_plates != expected_plates:
        raise ValueError(
            f"Expected plates {sorted(expected_plates)}; "
            f"received {sorted(supplied_plates)}."
        )

    missing_roots: list[str] = []
    missing_channels: list[str] = []
    needed_measurements = sorted({item.measurement for item in acquisitions})
    for plate in PLATES:
        for measurement in needed_measurements:
            key = (plate.name, measurement)
            if key not in indexed_roots:
                missing_roots.append(f"{plate.name} measurement {measurement}")
    for acquisition in acquisitions:
        for plate in PLATES:
            root = indexed_roots.get((plate.name, acquisition.measurement))
            if root is None:
                continue
            channel_dir = root / acquisition.channel_folder
            if not channel_dir.is_dir():
                missing_channels.append(str(channel_dir))
    if missing_roots or missing_channels:
        details = []
        if missing_roots:
            details.append("missing max_projection roots: " + ", ".join(missing_roots))
        if missing_channels:
            details.append("missing channel folders: " + ", ".join(missing_channels))
        raise FileNotFoundError("; ".join(details))


def parse_source_image(path: Path, plate: str, measurement: int) -> SourceImage:
    match = FIELD_FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported Phenix TIFF filename: {path.name}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    return SourceImage(
        # channel_dir is already absolute. Avoid resolving every TIFF through
        # the network filesystem; that adds a costly stat/readlink round trip.
        path=path,
        plate=plate,
        measurement=measurement,
        row=values["row"],
        column=values["column"],
        field=values["field"],
        timepoint=values["timepoint"],
        channel=values["channel"],
    )


def acquisition_seed(global_seed: int, acquisition_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{acquisition_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def balanced_plate_counts(total: int, rng: random.Random) -> dict[str, int]:
    if total < len(PLATES):
        raise ValueError(
            f"--samples-per-marker must be at least {len(PLATES)} so both plates "
            "are represented."
        )
    base, remainder = divmod(total, len(PLATES))
    names = [plate.name for plate in PLATES]
    rng.shuffle(names)
    return {
        plate.name: base + (1 if plate.name in names[:remainder] else 0)
        for plate in PLATES
    }


def expected_field_path(
    channel_dir: Path,
    row: int,
    column: int,
    field: int,
    source_channel: int,
) -> Path | None:
    """Resolve a canonical max-projection filename without listing the folder."""
    stem = (
        f"r{row:02d}_c{column:02d}_f{field:02d}_t01_"
        f"ch{source_channel:02d}"
    )
    for suffix in (".tiff", ".tif"):
        candidate = channel_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def sample_spread_across_wells(
    channel_dir: Path,
    plate: str,
    measurement: int,
    source_channel: int,
    count: int,
    rng: random.Random,
) -> list[SourceImage]:
    """Sample existing TIFFs while maximizing distinct occupied wells first."""
    # These are the 60 occupied wells in each plate of this 4i experiment.
    wells = [(row, column) for row in range(2, 8) for column in range(2, 12)]
    rng.shuffle(wells)
    available_fields = {well: list(range(1, 37)) for well in wells}
    for fields in available_fields.values():
        rng.shuffle(fields)

    selected: list[SourceImage] = []
    while len(selected) < count:
        eligible = [well for well in wells if available_fields[well]]
        if not eligible:
            raise ValueError(
                f"Requested {count} images but only {len(selected)} unique TIFFs "
                f"were found in {channel_dir}."
            )
        made_progress = False
        for row, column in eligible:
            source_path = None
            while available_fields[(row, column)] and source_path is None:
                field = available_fields[(row, column)].pop()
                source_path = expected_field_path(
                    channel_dir, row, column, field, source_channel
                )
            if source_path is None:
                continue
            selected.append(parse_source_image(source_path, plate, measurement))
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise FileNotFoundError(
                "No canonical Phenix TIFF filenames were found in "
                f"{channel_dir}. Expected names such as "
                f"r02_c02_f01_t01_ch{source_channel:02d}.tiff."
            )
        rng.shuffle(wells)
    return selected


def destination_name(sample_index: int, image: SourceImage) -> str:
    return (
        f"sample-{sample_index:03d}__plate-{image.plate}__"
        f"measurement-{image.measurement:02d}__{image.path.name}"
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_marker < 2:
        raise ValueError("--samples-per-marker must be at least 2.")

    acquisitions = selected_acquisitions()
    roots = index_max_projection_folders(args.max_projection_folders)
    validate_required_folders(roots, acquisitions)

    save_dir = args.save_dir.expanduser().absolute()
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be new or empty to avoid mixing runs: {save_dir}"
        )

    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []

    print(
        f"Planning {args.samples_per_marker} TIFFs for each of "
        f"{len(acquisitions)} acquisitions (28 markers + measurement-12 DAPI).",
        flush=True,
    )
    for acquisition in acquisitions:
        rng = random.Random(acquisition_seed(args.seed, acquisition.acquisition_id))
        per_plate = balanced_plate_counts(args.samples_per_marker, rng)
        sampled: list[SourceImage] = []

        for plate in PLATES:
            channel_dir = (
                roots[(plate.name, acquisition.measurement)]
                / acquisition.channel_folder
            )
            sampled.extend(
                sample_spread_across_wells(
                    channel_dir=channel_dir,
                    plate=plate.name,
                    measurement=acquisition.measurement,
                    source_channel=acquisition.source_channel,
                    count=per_plate[plate.name],
                    rng=rng,
                )
            )

        rng.shuffle(sampled)
        destination_dir = save_dir / acquisition.acquisition_id
        for sample_index, image in enumerate(sampled, start=1):
            destination = destination_dir / destination_name(sample_index, image)
            manifest_rows.append(
                {
                    "acquisition_id": acquisition.acquisition_id,
                    "marker": acquisition.marker,
                    "fluorophore": acquisition.fluorophore,
                    "source_channel": acquisition.source_channel,
                    "is_dapi": acquisition.is_dapi,
                    "sample_index": sample_index,
                    "plate": image.plate,
                    "measurement": image.measurement,
                    "well_row": image.row,
                    "well_column": image.column,
                    "field": image.field,
                    "timepoint": image.timepoint,
                    "source_path": str(image.path),
                    "destination_path": str(destination),
                }
            )

        distinct_wells = {(item.plate, item.row, item.column) for item in sampled}
        summary_rows.append(
            {
                "acquisition_id": acquisition.acquisition_id,
                "marker": acquisition.marker,
                "measurement": acquisition.measurement,
                "fluorophore": acquisition.fluorophore,
                "source_channel": acquisition.source_channel,
                "images_sampled": len(sampled),
                "distinct_wells_sampled": len(distinct_wells),
                **{
                    f"{plate.name}_images": sum(
                        image.plate == plate.name for image in sampled
                    )
                    for plate in PLATES
                },
            }
        )
        counts_text = ", ".join(
            f"{plate.name}={per_plate[plate.name]}" for plate in PLATES
        )
        print(
            f"  {acquisition.acquisition_id}: {len(sampled)} images "
            f"({counts_text}); {len(distinct_wells)} distinct plate/wells",
            flush=True,
        )

    if args.dry_run:
        print(
            f"Dry run complete: would create {len(acquisitions)} folders and copy "
            f"{len(manifest_rows)} TIFFs to {save_dir}."
        )
        return 0

    save_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest_rows:
        destination = Path(row["destination_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source_path"], destination)

    manifest_fields = [
        "acquisition_id",
        "marker",
        "fluorophore",
        "source_channel",
        "is_dapi",
        "sample_index",
        "plate",
        "measurement",
        "well_row",
        "well_column",
        "field",
        "timepoint",
        "source_path",
        "destination_path",
    ]
    summary_fields = list(summary_rows[0])
    write_csv(save_dir / "sampling_manifest.csv", manifest_rows, manifest_fields)
    write_csv(save_dir / "sampling_summary.csv", summary_rows, summary_fields)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "samples_per_marker": args.samples_per_marker,
        "seed": args.seed,
        "selection": "28 unique non-DAPI markers plus measurement-12 DAPI",
        "plate_names": [plate.name for plate in PLATES],
        "max_projection_folders": [str(path) for path in roots.values()],
        "output_directory": str(save_dir),
        "acquisitions": [item.as_row() for item in acquisitions],
        "files_copied": len(manifest_rows),
    }
    (save_dir / "sampling_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Complete: copied {len(manifest_rows)} TIFFs into "
        f"{len(acquisitions)} folders under {save_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
