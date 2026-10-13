from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from utils.registration import (
    centered_patch_crop,
    estimate_translation,
    intersect_bounds,
    robust_consensus_shift,
    valid_bounds_for_shift,
)
from utils.patch_level import PATCH_LEVEL_COHORTS, cohort_for_measurement, region_patch_plan
from utils.patch_level import analyze_patch_level_dapi
from utils.config import RegistrationSettings


def textured_image(size=256):
    rng = np.random.default_rng(42)
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(25):
        y, x = rng.integers(20, size - 20, size=2)
        image[y - 3 : y + 4, x - 5 : x + 6] += rng.uniform(1, 5)
    return ndi.gaussian_filter(image, 2)


def test_subpixel_translation_recovery():
    fixed = textured_image()
    imposed = (-3.25, 4.5)
    moving = ndi.shift(fixed, imposed, order=1, mode="constant", cval=0)
    result = estimate_translation(fixed, moving, upsample_factor=20)
    assert np.allclose(
        [result["shift_y"], result["shift_x"]],
        [-imposed[0], -imposed[1]],
        atol=0.25,
    )


def test_valid_intersection_and_centered_patch_crop():
    assert valid_bounds_for_shift((100, 120), 2.5, -3.25) == (3, 100, 0, 116)
    common = intersect_bounds(
        [(0, 100, 0, 120), (3, 100, 0, 116), (0, 98, 4, 120)]
    )
    assert common == (3, 98, 4, 116)
    assert centered_patch_crop(common, 32) == (18, 82, 12, 108)


def test_robust_field_consensus_ignores_outliers():
    candidates = [
        {"field_id": index, "informative": True, "shift_y": 10 + index * 0.01, "shift_x": -5}
        for index in range(1, 10)
    ]
    candidates.extend(
        [
            {"field_id": 10, "informative": True, "shift_y": 400, "shift_x": -300},
            {"field_id": 11, "informative": False},
        ]
    )
    result = robust_consensus_shift(candidates, tolerance_px=2, minimum_fields=6)
    assert result is not None
    assert abs(result["shift_y"] - 10.05) < 0.1
    assert abs(result["shift_x"] + 5) < 0.1
    assert result["consensus_field_count"] == 9


def test_patch_level_cohorts_split_at_microscope_maintenance():
    assert [item.cohort_id for item in PATCH_LEVEL_COHORTS] == ["m01_08", "m09_12"]
    assert cohort_for_measurement(1).reference_measurement == 1
    assert cohort_for_measurement(8).reference_measurement == 1
    assert cohort_for_measurement(9).reference_measurement == 9
    assert cohort_for_measurement(12).reference_measurement == 9


def test_patch_level_region_plan_is_field_and_cohort_traceable():
    region = {
        "cohort_id": "m09_12",
        "cohort_reference_measurement": 9,
        "field_id": 2,
        "crop_bounds": [28, 1052, 20, 1044],
    }
    plan = region_patch_plan(region, 512)
    assert len(plan) == 4
    assert plan[0]["patch_id"] == "m09_12__field_02__patch_0001"
    assert plan[0]["field_reference_y"] == 28
    assert plan[0]["field_reference_x"] == 20
    assert plan[0]["reference_y"] == 28
    assert plan[0]["reference_x"] == 20
    assert plan[-1]["field_reference_y"] == 540
    assert plan[-1]["field_reference_x"] == 532


def test_patch_level_registration_uses_two_independent_references():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(36, 64, 64)).astype(np.float32)
    base = ndi.gaussian_filter(base, sigma=(0, 1.2, 1.2))
    stacks = {}
    for measurement in range(1, 13):
        if measurement in {1, 9}:
            imposed = (0.0, 0.0)
        elif measurement <= 8:
            imposed = (2.0, -1.0)
        else:
            imposed = (-1.5, 2.0)
        stacks[measurement] = ndi.shift(
            base,
            shift=(0.0, *imposed),
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
    settings = RegistrationSettings(
        upsample_factor=5,
        min_global_correlation=0.10,
        patch_level_max_shift_px=16,
        patch_level_min_overlap_fraction=0.70,
    )
    shifts, qc, regions = analyze_patch_level_dapi(
        stacks,
        settings,
        16,
        expected_field_shape=(64, 64),
    )
    assert len(shifts) == 12 * 36
    assert len(qc) == 12 * 36
    assert len(regions) == 2 * 36
    assert all(item["status"] == "complete" for item in regions)
    lookup = {
        (item["measurement"], item["field_id"]): item for item in shifts
    }
    assert lookup[(1, 1)]["shift_y"] == 0
    assert lookup[(9, 1)]["shift_y"] == 0
    assert np.allclose(
        [lookup[(8, 1)]["shift_y"], lookup[(8, 1)]["shift_x"]],
        [-2.0, 1.0],
        atol=0.4,
    )
    assert np.allclose(
        [lookup[(12, 1)]["shift_y"], lookup[(12, 1)]["shift_x"]],
        [1.5, -2.0],
        atol=0.4,
    )
