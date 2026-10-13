from __future__ import annotations

import numpy as np
import pytest

from utils.correction import load_basic_model, model_load_mode


def test_basicpy_model_save_reload_equivalence(tmp_path):
    """Exercise BaSiCPy's supported serialization when it is installed."""
    basicpy = pytest.importorskip("basicpy")
    rng = np.random.default_rng(7)
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    illumination = 0.75 + 0.25 * np.exp(-(x * x + y * y) / 0.5)
    images = np.stack(
        [
            (200 + 30 * index + rng.normal(0, 3, illumination.shape)) * illumination
            for index in range(12)
        ]
    ).astype(np.float32)
    model = basicpy.BaSiC(get_darkfield=False, working_size=32)
    model.fit(images)
    before_values = model.transform(images[:2])
    before = np.asarray(
        before_values[0] if isinstance(before_values, tuple) else before_values,
        dtype=np.float32,
    )
    model_path = tmp_path / "basic_model"
    model.save_model(model_path, overwrite=True)
    reloaded = load_basic_model(model_path)
    after_values = reloaded.transform(images[:2])
    after = np.asarray(
        after_values[0] if isinstance(after_values, tuple) else after_values,
        dtype=np.float32,
    )
    assert np.allclose(before, after, rtol=1e-6, atol=1e-6)
    load_mode = model_load_mode(reloaded)
    assert load_mode == "basicpy_native" or load_mode.startswith(
        "basicpy_v2_nullable_float_compat:"
    )
