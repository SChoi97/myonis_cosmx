from __future__ import annotations

from pathlib import Path

import pytest

from utils.review_workbook import normalize_method, read_review_workbook


def test_method_normalization():
    assert normalize_method("basic") == "BaSiC"
    assert normalize_method("BaSiC") == "BaSiC"
    assert normalize_method("Harmony") == "Harmony-like"
    assert normalize_method("Harmony-like") == "Harmony-like"
    assert normalize_method("") == ""
    with pytest.raises(ValueError, match="BaSiC or Harmony-like"):
        normalize_method("unknown")


def test_review_workbook_derives_approval_from_choice(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    marker = workbook.active
    marker.title = "Marker Selections"
    marker.append(["acquisition_id", "chosen_method", "review_notes"])
    marker.append(["marker_one", "basic", "lowest seams"])
    dapi = workbook.create_sheet("DAPI Selections")
    dapi.append(["acquisition_id", "chosen_method", "review_notes"])
    dapi.append(["dapi_one", "Harmony", ""])
    path = tmp_path / "review.xlsx"
    workbook.save(path)

    rows = read_review_workbook(path)
    assert rows["marker_one"]["chosen_method"] == "BaSiC"
    assert rows["marker_one"]["review_status"] == "approved"
    assert rows["marker_one"]["review_notes"] == "lowest seams"
    assert rows["dapi_one"]["chosen_method"] == "Harmony-like"
