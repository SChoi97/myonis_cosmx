"""Load and apply approved BaSiC or Harmony-like correction models."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np

from .manifests import sha256_json, sha256_path


def illumination_utils_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "illumination_correction" / "utils"


def _import_harmony():
    import importlib.util

    path = illumination_utils_dir() / "harmony.py"
    specification = importlib.util.spec_from_file_location("four_i_harmony", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load Harmony implementation from {path}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_harmony_model(profile_path: str | Path, diagnostics_path: str | Path) -> dict:
    with np.load(profile_path) as profiles:
        model = {key: np.asarray(profiles[key]) for key in profiles.files}
    with Path(diagnostics_path).open(encoding="utf-8") as stream:
        model.update(json.load(stream))
    required = {"background_profile", "foreground_profile", "nominal_block_size", "local_window"}
    missing = required - set(model)
    if missing:
        raise ValueError(f"Harmony model is missing {sorted(missing)}.")
    return model


_BASICPY_NULLABLE_FLOAT_DEFAULTS = {
    "smoothness_flatfield",
    "smoothness_darkfield",
}


def _mark_load_mode(model, mode: str) -> None:
    """Attach provenance without relying on Pydantic's extra-field handling."""
    try:
        model.__dict__["_four_i_model_load_mode"] = mode
    except (AttributeError, TypeError):
        pass


def model_load_mode(model) -> str:
    if isinstance(model, dict):
        return str(model.get("_four_i_model_load_mode", "native"))
    return str(getattr(model, "_four_i_model_load_mode", "native"))


def load_basic_model(model_path: str | Path):
    """Load BaSiCPy profiles, working around its v2 nullable-float round-trip bug.

    BaSiCPy 2.0 declares two tuning parameters as floats with ``None`` defaults.
    Saving writes those defaults as JSON null, while loading passes explicit None
    through Pydantic validation and fails. Omitting only those null settings
    restores the constructor's original default semantics; the evaluated
    flatfield, darkfield, and baseline arrays are then loaded unchanged.
    """
    try:
        from basicpy import BaSiC
    except ImportError as error:
        raise ImportError("BaSiCPy is required to apply BaSiC models.") from error

    path = Path(model_path)
    try:
        model = BaSiC.load_model(path)
        _mark_load_mode(model, "basicpy_native")
        return model
    except Exception as native_error:
        settings_path = path / "settings.json"
        profiles_npz = path / "profiles.npz"
        if not settings_path.is_file() or not profiles_npz.is_file():
            raise native_error
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        nullable = {
            key
            for key in _BASICPY_NULLABLE_FLOAT_DEFAULTS
            if key in settings and settings[key] is None
        }
        if not nullable:
            raise native_error
        for key in nullable:
            settings.pop(key)
        try:
            model = BaSiC(**settings)
            with np.load(profiles_npz, allow_pickle=True) as profiles:
                required = {"flatfield", "darkfield", "baseline"}
                missing = required - set(profiles.files)
                if missing:
                    raise ValueError(
                        f"BaSiC profiles are missing arrays {sorted(missing)} at "
                        f"{profiles_npz}."
                    )
                model.flatfield = np.asarray(profiles["flatfield"])
                model.darkfield = np.asarray(profiles["darkfield"])
                model.baseline = np.asarray(profiles["baseline"])
        except Exception as compatibility_error:
            raise RuntimeError(
                f"BaSiC native and compatibility loading both failed at {path}. "
                f"Native error: {native_error}"
            ) from compatibility_error
        mode = "basicpy_v2_nullable_float_compat:" + ",".join(sorted(nullable))
        _mark_load_mode(model, mode)
        warnings.warn(
            f"Loaded {path} using {mode}; saved profile arrays were not modified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model


def load_model(selection_row: dict[str, str]):
    method = selection_row["chosen_method"]
    if method == "BaSiC":
        path = selection_row["basic_model_path"]
        observed = sha256_path(path)
        if observed != selection_row["basic_model_sha256"]:
            raise RuntimeError(f"Approved BaSiC model hash mismatch at {path}.")
        return method, load_basic_model(path)
    if method == "Harmony-like":
        profile_path = selection_row["harmony_profile_path"]
        diagnostics_path = selection_row["harmony_diagnostics_path"]
        if sha256_path(profile_path) != selection_row["harmony_profile_sha256"]:
            raise RuntimeError(f"Approved Harmony profile hash mismatch at {profile_path}.")
        if sha256_path(diagnostics_path) != selection_row["harmony_diagnostics_sha256"]:
            raise RuntimeError(
                f"Approved Harmony diagnostics hash mismatch at {diagnostics_path}."
            )
        model = load_harmony_model(
            profile_path,
            diagnostics_path,
        )
        model["_four_i_model_load_mode"] = "harmony_native"
        return method, model
    raise ValueError(f"Unsupported approved correction method {method!r}.")


def selected_model_hash(selection_row: dict[str, str]) -> str:
    """Return the complete approved model hash used by production."""
    if selection_row["chosen_method"] == "BaSiC":
        return selection_row["basic_model_sha256"]
    if selection_row["chosen_method"] == "Harmony-like":
        stored = selection_row.get("harmony_model_sha256", "")
        if stored:
            return stored
        return sha256_json(
            {
                "profile": selection_row["harmony_profile_sha256"],
                "diagnostics": selection_row["harmony_diagnostics_sha256"],
            }
        )
    raise ValueError(selection_row["chosen_method"])


def apply_model(images: np.ndarray, method: str, model) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(f"Expected (field, y, x); received {images.shape}.")
    if method == "BaSiC":
        values = model.transform(np.asarray(images, dtype=np.float32))
        if isinstance(values, tuple):
            values = values[0]
    elif method == "Harmony-like":
        harmony = _import_harmony()
        values, _ = harmony.apply_correction(images, model)
    else:
        raise ValueError(method)
    corrected = np.clip(np.asarray(values, dtype=np.float32), 0, None)
    if not np.all(np.isfinite(corrected)):
        raise RuntimeError(f"{method} produced NaN or infinity.")
    return corrected
