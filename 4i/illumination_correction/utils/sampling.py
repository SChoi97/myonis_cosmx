"""Shared paired-plate sampling for reproducible cross-marker evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .io import ChannelIndex, FieldKey, WellKey


PlateFieldKey = tuple[int, int, int, int]
PlateWellKey = tuple[int, int, int]


@dataclass(frozen=True)
class SamplingPlan:
    training_fields: tuple[PlateFieldKey, ...]
    visualization_wells: tuple[PlateWellKey, ...]
    seed: int


def _balanced_sample(
    candidates_by_plate: Mapping[int, Sequence[tuple]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple]:
    if count < 1:
        raise ValueError("Sample count must be positive.")
    plates = sorted(candidates_by_plate)
    if not plates:
        raise ValueError("No plate candidates were supplied.")
    base, remainder = divmod(count, len(plates))
    order = list(rng.permutation(plates))
    allocations = {plate: base for plate in plates}
    for plate in order[:remainder]:
        allocations[int(plate)] += 1
    sampled = []
    for plate in plates:
        candidates = list(candidates_by_plate[plate])
        needed = allocations[plate]
        if len(candidates) < needed:
            raise ValueError(
                f"Plate {plate} has {len(candidates)} shared candidates but "
                f"needs {needed}."
            )
        chosen = rng.choice(len(candidates), size=needed, replace=False)
        sampled.extend(candidates[int(index)] for index in chosen)
    rng.shuffle(sampled)
    return sampled


def build_shared_sampling_plan(
    indexed_markers: Mapping[str, tuple[ChannelIndex, ChannelIndex]],
    *,
    training_field_count: int,
    visualization_well_count: int,
    seed: int,
) -> SamplingPlan:
    """Sample only coordinates present for every marker on the same plate slot."""
    if not indexed_markers:
        raise ValueError("At least one indexed marker is required.")
    common_fields: dict[int, list[PlateFieldKey]] = {}
    common_wells: dict[int, list[PlateWellKey]] = {}
    for slot in (1, 2):
        slot_indices = [pair[slot - 1] for pair in indexed_markers.values()]
        plate_names = {index.plate_name for index in slot_indices}
        if len(plate_names) != 1:
            raise ValueError(
                f"Plate slot {slot} is inconsistent across markers: {plate_names}. "
                "Pass the same biological plate in the same argument position."
            )
        shared_field_keys = set(slot_indices[0].by_key)
        shared_well_keys = set(slot_indices[0].complete_wells)
        for index in slot_indices[1:]:
            shared_field_keys.intersection_update(index.by_key)
            shared_well_keys.intersection_update(index.complete_wells)
        common_fields[slot] = [
            (slot, row, column, field)
            for row, column, field in sorted(shared_field_keys)
        ]
        common_wells[slot] = [
            (slot, row, column) for row, column in sorted(shared_well_keys)
        ]
        print(
            f"Plate slot {slot} ({next(iter(plate_names))}): "
            f"{len(common_fields[slot])} shared fields, "
            f"{len(common_wells[slot])} shared complete wells."
        )
    rng = np.random.default_rng(seed)
    training = _balanced_sample(common_fields, training_field_count, rng)
    visualization = _balanced_sample(common_wells, visualization_well_count, rng)
    return SamplingPlan(
        training_fields=tuple(training),
        visualization_wells=tuple(visualization),
        seed=seed,
    )


def training_paths_for_marker(
    marker_indices: tuple[ChannelIndex, ChannelIndex],
    plan: SamplingPlan,
) -> list[Path]:
    return [
        marker_indices[slot - 1].by_key[(row, column, field)]
        for slot, row, column, field in plan.training_fields
    ]


def load_sampling_manifest(path: str | Path) -> SamplingPlan:
    """Load a plan prepared once for use by independent marker jobs."""
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Sampling manifest is empty: {manifest_path}.")
    rows.sort(key=lambda row: (row["purpose"], int(row["sequence"])))
    training = []
    visualization = []
    seeds = set()
    for row in rows:
        seeds.add(int(row["random_seed"]))
        if row["purpose"] == "model_training":
            training.append(
                (
                    int(row["plate_slot"]),
                    int(row["row"]),
                    int(row["column"]),
                    int(row["field"]),
                )
            )
        elif row["purpose"] == "visualization_well":
            visualization.append(
                (
                    int(row["plate_slot"]),
                    int(row["row"]),
                    int(row["column"]),
                )
            )
    if len(seeds) != 1 or not training or not visualization:
        raise ValueError(
            f"Invalid sampling manifest {manifest_path}: seeds={seeds}, "
            f"training={len(training)}, visualization={len(visualization)}."
        )
    return SamplingPlan(
        training_fields=tuple(training),
        visualization_wells=tuple(visualization),
        seed=next(iter(seeds)),
    )


def validate_plan_for_marker(
    marker_indices: tuple[ChannelIndex, ChannelIndex],
    plan: SamplingPlan,
) -> None:
    """Verify a prepared shared plan against one independently indexed marker."""
    missing_fields = [
        coordinate
        for coordinate in plan.training_fields
        if coordinate[1:] not in marker_indices[coordinate[0] - 1].by_key
    ]
    missing_wells = [
        coordinate
        for coordinate in plan.visualization_wells
        if coordinate[1:] not in marker_indices[coordinate[0] - 1].complete_wells
    ]
    if missing_fields or missing_wells:
        raise ValueError(
            f"Prepared plan is incompatible with marker {marker_indices[0].marker_name}: "
            f"{len(missing_fields)} missing training fields and "
            f"{len(missing_wells)} incomplete visualization wells."
        )


def write_sampling_manifest(
    path: Path,
    plan: SamplingPlan,
    indexed_markers: Mapping[str, tuple[ChannelIndex, ChannelIndex]],
) -> None:
    """Write the exact shared coordinates used throughout the run."""
    first_pair = next(iter(indexed_markers.values()))
    rows = []
    for sequence, (slot, row, column, field) in enumerate(
        plan.training_fields, start=1
    ):
        rows.append(
            {
                "purpose": "model_training",
                "sequence": sequence,
                "plate_slot": slot,
                "plate_name": first_pair[slot - 1].plate_name,
                "row": row,
                "column": column,
                "field": field,
                "random_seed": plan.seed,
            }
        )
    for sequence, (slot, row, column) in enumerate(
        plan.visualization_wells, start=1
    ):
        rows.append(
            {
                "purpose": "visualization_well",
                "sequence": sequence,
                "plate_slot": slot,
                "plate_name": first_pair[slot - 1].plate_name,
                "row": row,
                "column": column,
                "field": "all_1_to_36",
                "random_seed": plan.seed,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
