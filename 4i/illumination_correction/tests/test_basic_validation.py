"""BaSiC validation must retry or fail before NaNs reach metric calculation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from utils import corrections  # noqa: E402


def test_darkfield_nan_fit_retries_without_darkfield(monkeypatch):
    attempts = []

    class FakeBaSiC:
        def __init__(self, *, get_darkfield, sort_intensity, **kwargs):
            self.get_darkfield = get_darkfield
            self.sort_intensity = sort_intensity
            attempts.append(get_darkfield)

        def fit(self, images):
            value = np.nan if self.get_darkfield else 1.0
            self.flatfield = np.full(images.shape[1:], value, dtype=np.float32)
            self.darkfield = np.zeros(images.shape[1:], dtype=np.float32)
            self.baseline = None

        def transform(self, images):
            if self.get_darkfield:
                return np.full(images.shape, np.nan, dtype=np.float32)
            return np.asarray(images, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "basicpy", SimpleNamespace(BaSiC=FakeBaSiC))
    images = np.ones((8, 16, 16), dtype=np.uint16)
    _, diagnostics = corrections.fit_basic(
        images,
        label="synthetic",
        get_darkfield=True,
        sort_intensity=True,
    )
    assert attempts == [True, False]
    assert diagnostics["get_darkfield"] is False
    assert diagnostics["status"] == "validated_finite"


def test_nonfinite_transform_is_rejected():
    class InvalidModel:
        def transform(self, images):
            return np.full(images.shape, np.nan, dtype=np.float32)

    images = np.ones((36, 16, 16), dtype=np.float32)
    try:
        corrections.apply_basic(images, InvalidModel())
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("A non-finite BaSiC transform was not rejected.")
