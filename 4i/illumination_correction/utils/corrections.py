"""Fit one paired-plate model per marker and apply all comparison methods."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Sequence

import numpy as np

from . import harmony
from .io import load_paths


@dataclass(frozen=True)
class CorrectionSettings:
    random_seed: int = 20260819
    basic_get_darkfield: bool = False
    basic_sort_intensity: bool = True
    harmony_block_size: int = 64
    harmony_noise_window: int = 9
    harmony_group_size: int = 128
    harmony_background_regularization: float = 0.2
    harmony_foreground_regularization: float = 20.0


@dataclass
class MarkerModels:
    basic: object
    basic_diagnostics: dict
    harmony: dict


def _array_diagnostics(values: np.ndarray) -> dict:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        "shape": list(values.shape),
        "finite_fraction": float(np.mean(np.isfinite(values))),
        "positive_fraction": float(np.mean(values > 0)),
        "minimum": float(np.min(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "maximum": float(np.max(finite)) if finite.size else float("nan"),
    }


def _validate_basic_model(model, training_images: np.ndarray) -> dict:
    """Reject a fitted model before it can silently emit all-NaN images."""
    diagnostics = {}
    for attribute in ("flatfield", "darkfield", "baseline"):
        values = getattr(model, attribute, None)
        if values is not None:
            diagnostics[attribute] = _array_diagnostics(values)
    flatfield = np.asarray(getattr(model, "flatfield", []))
    if flatfield.size == 0 or not np.all(np.isfinite(flatfield)):
        raise ValueError("BaSiC fitted a missing or non-finite flatfield profile.")
    if np.any(flatfield <= 0):
        raise ValueError("BaSiC fitted a non-positive flatfield profile.")
    # A direct transform is the definitive validation. Model arrays can appear
    # finite even when an invalid baseline causes transformation to return NaN.
    sample = np.asarray(training_images[: min(4, training_images.shape[0])])
    transformed = model.transform(sample)
    if isinstance(transformed, tuple):
        transformed = transformed[0]
    transformed = np.asarray(transformed)
    diagnostics["validation_transform"] = _array_diagnostics(transformed)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("BaSiC validation transform contained NaN or infinity.")
    if not np.any(transformed > 0):
        raise ValueError("BaSiC validation transform collapsed to an all-zero image.")
    return diagnostics


def fit_basic(
    training_images: np.ndarray,
    *,
    label: str,
    get_darkfield: bool,
    sort_intensity: bool,
) -> tuple[object, dict]:
    """Fit and validate BaSiCPy using shared FOVs from both plates."""
    try:
        from basicpy import BaSiC
    except ImportError as error:
        raise ImportError(
            "BaSiCPy is required in the cluster environment: pip install basicpy"
        ) from error
    attempts = [(get_darkfield, sort_intensity)]
    if get_darkfield:
        # Darkfield estimation is documented by BaSiCPy as potentially
        # unstable. Retry automatically with the robust default if requested.
        attempts.append((False, True))
    failures = []
    for attempt, (attempt_darkfield, attempt_sort) in enumerate(attempts, start=1):
        print(
            f"[{label}] BaSiC attempt {attempt}/{len(attempts)}: fitting "
            f"{training_images.shape[0]} shared FOVs from both plates "
            f"(darkfield={attempt_darkfield}, sort_intensity={attempt_sort})..."
        )
        model = BaSiC(
            get_darkfield=attempt_darkfield,
            sort_intensity=attempt_sort,
            smoothness_flatfield=1,
        )
        try:
            model.fit(training_images)
            diagnostics = _validate_basic_model(model, training_images)
        except TypeError as error:
            if "expected np.ndarray" in str(error):
                raise RuntimeError(
                    "The NumPy–PyTorch bridge in this environment is broken. "
                    "Use the dedicated illumination_correction_env."
                ) from error
            failures.append(f"attempt {attempt}: {error}")
            continue
        except (RuntimeError, ValueError) as error:
            failures.append(f"attempt {attempt}: {error}")
            print(f"[{label}] WARNING: {failures[-1]}")
            continue
        diagnostics.update(
            {
                "training_field_count": int(training_images.shape[0]),
                "get_darkfield": attempt_darkfield,
                "sort_intensity": attempt_sort,
                "attempt": attempt,
                "status": "validated_finite",
            }
        )
        return model, diagnostics
    raise RuntimeError(
        f"BaSiC failed finite-output validation for {label}. "
        + " | ".join(failures)
    )


def apply_basic(field_stack: np.ndarray, model) -> np.ndarray:
    images = np.asarray(field_stack, dtype=np.float32)
    transformed = model.transform(images)
    if isinstance(transformed, tuple):
        transformed = transformed[0]
    corrected = np.clip(np.asarray(transformed, dtype=np.float32), 0, None)
    if not np.all(np.isfinite(corrected)):
        finite_fraction = float(np.mean(np.isfinite(corrected)))
        raise RuntimeError(
            "BaSiC produced non-finite corrected pixels "
            f"(finite fraction={finite_fraction:.6f}). The marker is stopped "
            "instead of writing invalid metrics."
        )
    if not np.any(corrected > 0):
        raise RuntimeError(
            "BaSiC correction collapsed the well to all zeros. The marker is "
            "stopped instead of writing invalid metrics."
        )
    return corrected


def fit_marker_models(
    training_paths: Sequence[Path],
    *,
    label: str,
    settings: CorrectionSettings,
) -> MarkerModels:
    """Fit BaSiC and Harmony-like to the exact same paired-plate paths."""
    training_paths = [Path(path) for path in training_paths]
    if not training_paths:
        raise ValueError(f"No model-training fields were supplied for {label}.")
    training_images = load_paths(training_paths)
    target_shape = training_images.shape[1:]
    basic_model, basic_diagnostics = fit_basic(
        training_images,
        label=label,
        get_darkfield=settings.basic_get_darkfield,
        sort_intensity=settings.basic_sort_intensity,
    )
    del training_images
    gc.collect()
    harmony_model = harmony.fit_profiles(
        training_paths,
        target_shape=target_shape,
        label=label,
        random_seed=settings.random_seed,
        nominal_block_size=settings.harmony_block_size,
        local_window=settings.harmony_noise_window,
        group_size=settings.harmony_group_size,
        background_regularization=settings.harmony_background_regularization,
        foreground_regularization=settings.harmony_foreground_regularization,
    )
    return MarkerModels(
        basic=basic_model,
        basic_diagnostics=basic_diagnostics,
        harmony=harmony_model,
    )


def apply_all_corrections(
    raw_fields: np.ndarray,
    models: MarkerModels,
) -> OrderedDict[str, np.ndarray]:
    """Return fields in the fixed output order Original, BaSiC, Harmony-like."""
    harmony_fields, _ = harmony.apply_correction(raw_fields, models.harmony)
    return OrderedDict(
        [
            ("Original", np.asarray(raw_fields)),
            ("BaSiC", apply_basic(raw_fields, models.basic)),
            ("Harmony-like", harmony_fields),
        ]
    )
