from __future__ import annotations

import numpy as np

from utils.layout import FIELD_ID_GRID, field_box, tile_fields
from utils.config import PipelineConfig, with_execution_mode
from utils.registry import (
    ACQUISITIONS,
    biological_coordinates,
    configured_well_tasks,
    PLATES,
    well_tasks,
)


def test_registry_contains_12_dapi_and_28_markers():
    assert len(ACQUISITIONS) == 40
    assert sum(item.is_dapi for item in ACQUISITIONS) == 12
    assert len({item.acquisition_id for item in ACQUISITIONS}) == 40
    assert "measurement_09_488" not in " ".join(item.acquisition_id for item in ACQUISITIONS)


def test_verified_phenix_layout_and_tiling():
    assert FIELD_ID_GRID.tolist() == [
        [2, 3, 4, 5, 6, 7],
        [13, 12, 11, 10, 9, 8],
        [14, 15, 16, 17, 18, 19],
        [25, 24, 23, 22, 21, 20],
        [26, 27, 28, 29, 30, 31],
        [1, 36, 35, 34, 33, 32],
    ]
    fields = np.stack([np.full((2, 3), field_id) for field_id in range(1, 37)])
    mosaic = tile_fields(fields)
    assert mosaic.shape == (12, 18)
    assert mosaic[0, 0] == 2
    assert mosaic[-1, 0] == 1
    assert mosaic[-1, -1] == 32


def test_field_boxes_and_biological_coordinates():
    assert field_box(2) == (0, 1080, 0, 1080)
    assert field_box(1) == (5400, 6480, 0, 1080)
    assert biological_coordinates(PLATES[0], 4, 2) == {
        "repeat": 5,
        "cell_line_row": 3,
        "well_within_repeat": 1,
    }
    assert biological_coordinates(PLATES[1], 7, 11)["repeat"] == 8
    assert len(well_tasks()) == 120


def test_test_mode_selects_four_wells_across_both_plates_and_all_repeats():
    config = PipelineConfig(data_root="/data", evaluation_dir="/evaluation", test_mode=True)
    tasks = configured_well_tasks(config)
    assert len(tasks) == 4
    assert [(plate.name, row, column) for plate, row, column in tasks] == [
        ("sc_rep5-6", 4, 2),
        ("sc_rep5-6", 4, 7),
        ("sc_rep7-8", 4, 2),
        ("sc_rep7-8", 4, 7),
    ]
    repeats = {
        biological_coordinates(plate, row, column)["repeat"]
        for plate, row, column in tasks
    }
    assert repeats == {5, 6, 7, 8}
    production = PipelineConfig(data_root="/data", evaluation_dir="/evaluation")
    assert len(configured_well_tasks(production)) == 120


def test_execution_mode_changes_registration_without_changing_well_selection():
    base = PipelineConfig(data_root="/data", evaluation_dir="/evaluation")
    patch_test = with_execution_mode(
        base, test_mode=True, registration_level="patch_level"
    )
    assert patch_test.registration.level == "patch_level"
    assert patch_test.test_mode is True
    assert len(configured_well_tasks(patch_test)) == 4
    field_production = with_execution_mode(
        base, test_mode=False, registration_level="field_level"
    )
    assert field_production.registration.level == "field_level"
    assert len(configured_well_tasks(field_production)) == 120
