"""Independent source-field registration for the two pre/post-maintenance cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .layout import FIELD_ID_GRID, FIELD_SHAPE, field_box
from .patches import patch_plan
from .registration import (
    bounds_area,
    centered_patch_crop,
    estimate_translation,
    global_qc,
    intersect_bounds,
    robust_normalize,
    valid_bounds_for_shift,
)


@dataclass(frozen=True)
class RegistrationCohort:
    cohort_id: str
    measurements: tuple[int, ...]
    reference_measurement: int


PATCH_LEVEL_COHORTS = (
    RegistrationCohort("m01_08", tuple(range(1, 9)), 1),
    RegistrationCohort("m09_12", tuple(range(9, 13)), 9),
)


def cohort_for_measurement(measurement: int) -> RegistrationCohort:
    matches = [item for item in PATCH_LEVEL_COHORTS if measurement in item.measurements]
    if len(matches) != 1:
        raise ValueError(f"Measurement {measurement} does not belong to one patch-level cohort.")
    return matches[0]


def _grid_position(field_id: int) -> tuple[int, int]:
    position = np.argwhere(FIELD_ID_GRID == field_id)
    if position.shape != (1, 2):
        raise ValueError(f"field_id must be in 1..36; received {field_id}.")
    return int(position[0, 0]) + 1, int(position[0, 1]) + 1


def _low_information_result(
    normalized_std: float, field_shape: tuple[int, int]
) -> dict:
    return {
        "shift_y": 0.0,
        "shift_x": 0.0,
        "shift_magnitude": 0.0,
        "phase_error": float("nan"),
        "phase_difference": float("nan"),
        "fft_shape": list(field_shape),
        "overlap_fraction": 1.0,
        "post_registration_correlation": float("nan"),
        "valid_bounds": [0, field_shape[0], 0, field_shape[1]],
        "global_qc_status": "fail",
        "global_qc_reasons": "low_information",
        "shift_source": "patch_phase_correlation",
        "normalized_std": normalized_std,
    }


def register_patch_field(
    fixed: np.ndarray,
    moving: np.ndarray,
    settings,
    override: dict[str, str] | None = None,
    *,
    expected_shape: tuple[int, int] = FIELD_SHAPE,
) -> dict:
    """Register one corrected 1080x1080 source field without assembling a mosaic."""
    if fixed.shape != expected_shape or moving.shape != expected_shape:
        raise ValueError(
            f"Patch-level registration requires {expected_shape}; got {fixed.shape}, {moving.shape}."
        )
    normalized_std = min(
        float(np.std(robust_normalize(fixed))),
        float(np.std(robust_normalize(moving))),
    )
    if normalized_std < settings.min_field_normalized_std:
        checked = _low_information_result(normalized_std, expected_shape)
    else:
        estimate = estimate_translation(
            fixed,
            moving,
            downsample=settings.downsample,
            upsample_factor=settings.upsample_factor,
        )
        estimate["shift_source"] = "patch_phase_correlation"
        checked = global_qc(
            fixed,
            moving,
            estimate,
            max_shift_px=settings.patch_level_max_shift_px,
            min_overlap_fraction=settings.patch_level_min_overlap_fraction,
            min_correlation=settings.min_global_correlation,
        )
        checked["normalized_std"] = normalized_std

    initial = {
        f"initial_{key}": checked.get(key)
        for key in (
            "shift_y",
            "shift_x",
            "shift_magnitude",
            "phase_error",
            "phase_difference",
            "overlap_fraction",
            "post_registration_correlation",
            "global_qc_status",
            "global_qc_reasons",
        )
    }
    if override:
        action = override.get("action", "").strip().lower()
        if action == "exclude":
            checked["global_qc_status"] = "fail"
            checked["global_qc_reasons"] = "manual_exclusion"
            checked["override_action"] = action
        elif action == "replace":
            shift_y = float(override["shift_y"])
            shift_x = float(override["shift_x"])
            replacement = {
                "shift_y": shift_y,
                "shift_x": shift_x,
                "shift_magnitude": float(np.hypot(shift_y, shift_x)),
                "phase_error": float("nan"),
                "phase_difference": float("nan"),
                "fft_shape": None,
                "shift_source": "manual_patch_replacement",
            }
            checked = global_qc(
                fixed,
                moving,
                replacement,
                max_shift_px=settings.patch_level_max_shift_px,
                min_overlap_fraction=settings.patch_level_min_overlap_fraction,
                min_correlation=settings.min_global_correlation,
            )
            checked["normalized_std"] = normalized_std
            checked["override_action"] = action
        elif action == "accept":
            checked["global_qc_status"] = "pass"
            checked["global_qc_reasons"] = ""
            checked["override_action"] = action
        elif action:
            raise ValueError(f"Unknown override action {action!r}.")
    checked.update(initial)
    checked.pop("registered", None)
    return checked


def analyze_patch_level_dapi(
    dapi_fields: Mapping[int, np.ndarray],
    settings,
    patch_size: int,
    *,
    override_lookup: Callable[[int, int], dict[str, str] | None] | None = None,
    expected_field_shape: tuple[int, int] = FIELD_SHAPE,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return per-field shifts, QC rows, and cohort-specific crop regions."""
    expected = set(range(1, 13))
    if set(dapi_fields) != expected:
        raise ValueError(
            f"Patch-level DAPI input must contain measurements 1..12; "
            f"missing={sorted(expected - set(dapi_fields))}."
        )
    for measurement, values in dapi_fields.items():
        if values.shape != (36, *expected_field_shape):
            raise ValueError(
                f"Measurement {measurement} has invalid DAPI stack {values.shape}."
            )

    shift_rows: list[dict] = []
    qc_rows: list[dict] = []
    regions: list[dict] = []
    for cohort in PATCH_LEVEL_COHORTS:
        reference_stack = dapi_fields[cohort.reference_measurement]
        for field_id in range(1, 37):
            grid_row, grid_column = _grid_position(field_id)
            fixed = reference_stack[field_id - 1]
            reference_std = float(np.std(robust_normalize(fixed)))
            per_measurement: list[dict] = []
            for measurement in cohort.measurements:
                if measurement == cohort.reference_measurement:
                    status = (
                        "pass"
                        if reference_std >= settings.min_field_normalized_std
                        else "not_evaluable"
                    )
                    result = {
                        "shift_y": 0.0,
                        "shift_x": 0.0,
                        "shift_magnitude": 0.0,
                        "phase_error": 0.0,
                        "phase_difference": 0.0,
                        "fft_shape": list(expected_field_shape),
                        "overlap_fraction": 1.0,
                        "post_registration_correlation": 1.0,
                        "valid_bounds": [
                            0,
                            expected_field_shape[0],
                            0,
                            expected_field_shape[1],
                        ],
                        "global_qc_status": status,
                        "global_qc_reasons": "" if status == "pass" else "low_information",
                        "shift_source": "cohort_reference",
                        "normalized_std": reference_std,
                    }
                else:
                    override = (
                        override_lookup(measurement, field_id)
                        if override_lookup is not None
                        else None
                    )
                    result = register_patch_field(
                        fixed,
                        dapi_fields[measurement][field_id - 1],
                        settings,
                        override,
                        expected_shape=expected_field_shape,
                    )
                row = {
                    "registration_level": "patch_level",
                    "cohort_id": cohort.cohort_id,
                    "cohort_reference_measurement": cohort.reference_measurement,
                    "measurement": measurement,
                    "field_id": field_id,
                    "grid_row": grid_row,
                    "grid_column": grid_column,
                    **result,
                }
                shift_rows.append(row)
                per_measurement.append(row)
                qc_rows.append(
                    {
                        "registration_level": "patch_level",
                        "cohort_id": cohort.cohort_id,
                        "cohort_reference_measurement": cohort.reference_measurement,
                        "measurement": measurement,
                        "field_id": field_id,
                        "grid_row": grid_row,
                        "grid_column": grid_column,
                        "normalized_std": result.get("normalized_std"),
                        "status": (
                            "pass"
                            if result["global_qc_status"] == "pass"
                            else (
                                "not_evaluable"
                                if result.get("global_qc_reasons") == "low_information"
                                else "fail"
                            )
                        ),
                        "correlation": result.get("post_registration_correlation"),
                        "registration_shift_y": result.get("shift_y"),
                        "registration_shift_x": result.get("shift_x"),
                        "registration_shift_magnitude": result.get("shift_magnitude"),
                        "residual_shift_y": float("nan"),
                        "residual_shift_x": float("nan"),
                        "residual_magnitude": float("nan"),
                        "residual_consensus_disagreement": float("nan"),
                        "failure_reasons": result.get("global_qc_reasons", ""),
                    }
                )

            common_bounds = intersect_bounds(
                valid_bounds_for_shift(
                    expected_field_shape,
                    float(row["shift_y"]),
                    float(row["shift_x"]),
                )
                for row in per_measurement
            )
            common_overlap_fraction = bounds_area(common_bounds) / float(
                expected_field_shape[0] * expected_field_shape[1]
            )
            crop_bounds = centered_patch_crop(common_bounds, patch_size)
            crop_has_patch = bounds_area(crop_bounds) >= patch_size**2
            failed_measurements = [
                int(row["measurement"])
                for row in per_measurement
                if row["global_qc_status"] != "pass"
            ]
            reasons = []
            if failed_measurements:
                reasons.append("registration_qc_failure")
            if common_overlap_fraction < settings.patch_level_min_overlap_fraction:
                reasons.append("insufficient_common_overlap")
            if not crop_has_patch:
                reasons.append("no_complete_output_patch")
            count = len(patch_plan(crop_bounds, patch_size)) if crop_has_patch else 0
            regions.append(
                {
                    "registration_level": "patch_level",
                    "cohort_id": cohort.cohort_id,
                    "cohort_measurements": list(cohort.measurements),
                    "cohort_reference_measurement": cohort.reference_measurement,
                    "field_id": field_id,
                    "grid_row": grid_row,
                    "grid_column": grid_column,
                    "status": "quarantined" if reasons else "complete",
                    "failure_reasons": ";".join(reasons),
                    "failed_measurements": failed_measurements,
                    "common_bounds": list(common_bounds),
                    "common_overlap_fraction": common_overlap_fraction,
                    "crop_bounds": list(crop_bounds),
                    "crop_area_fraction": bounds_area(crop_bounds)
                    / float(expected_field_shape[0] * expected_field_shape[1]),
                    "patch_count_total": count,
                    "patch_count_retained": 0 if reasons else count,
                }
            )
    return shift_rows, qc_rows, regions


def region_patch_plan(region: dict, patch_size: int) -> list[dict]:
    """Create uniquely named 512-patch records within one field/cohort crop."""
    field_id = int(region["field_id"])
    field_y0, _, field_x0, _ = field_box(field_id)
    records = []
    for patch in patch_plan(tuple(region["crop_bounds"]), patch_size):
        field_reference_y = int(patch["reference_y"])
        field_reference_x = int(patch["reference_x"])
        records.append(
            {
                **patch,
                "patch_id": (
                    f"{region['cohort_id']}__field_{field_id:02d}__"
                    f"{patch['patch_id']}"
                ),
                "field_id": field_id,
                "cohort_id": region["cohort_id"],
                "cohort_reference_measurement": int(
                    region["cohort_reference_measurement"]
                ),
                "field_reference_y": field_reference_y,
                "field_reference_x": field_reference_x,
                "reference_y": field_y0 + field_reference_y,
                "reference_x": field_x0 + field_reference_x,
            }
        )
    return records
