"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistrationSettings:
    level: str = "field_level"
    reference_measurement: int = 1
    downsample: int = 1
    upsample_factor: int = 20
    max_global_shift_px: float = 1200.0
    min_common_overlap_fraction: float = 0.65
    min_global_correlation: float = 0.25
    max_field_residual_px: float = 10.0
    min_field_correlation: float = 0.20
    min_field_normalized_std: float = 0.01
    consensus_tolerance_px: float = 15.0
    consensus_min_fields: int = 6
    display_downsample: int = 8
    patch_level_max_shift_px: float = 128.0
    patch_level_min_overlap_fraction: float = 0.80


@dataclass(frozen=True)
class StorageSettings:
    chunk_y: int = 540
    chunk_x: int = 540
    compression_level: int = 5
    tiff_compression: str = "zlib"
    cleanup_corrected_after_registration: bool = True
    cleanup_registered_after_patching: bool = True


@dataclass(frozen=True)
class TestWellSettings:
    plate_name: str
    row: int
    column: int


DEFAULT_TEST_WELLS = (
    TestWellSettings("sc_rep5-6", 4, 2),
    TestWellSettings("sc_rep5-6", 4, 7),
    TestWellSettings("sc_rep7-8", 4, 2),
    TestWellSettings("sc_rep7-8", 4, 7),
)


@dataclass(frozen=True)
class PipelineConfig:
    data_root: str
    evaluation_dir: str
    patch_size: int = 1024
    random_seed: int = 20260820
    correction_batch_size: int = 4
    smoke_row: int = 4
    smoke_column: int = 2
    test_mode: bool = False
    test_wells: tuple[TestWellSettings, ...] = DEFAULT_TEST_WELLS
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


def _merge_dataclass(cls, values: dict[str, Any] | None):
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    values = values or {}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    values = json.loads(config_path.read_text(encoding="utf-8"))
    registration = _merge_dataclass(
        RegistrationSettings, values.pop("registration", None)
    )
    storage = _merge_dataclass(StorageSettings, values.pop("storage", None))
    test_wells = tuple(
        _merge_dataclass(TestWellSettings, item)
        for item in values.pop("test_wells", [asdict(item) for item in DEFAULT_TEST_WELLS])
    )
    config = PipelineConfig(
        registration=registration,
        storage=storage,
        test_wells=test_wells,
        **values,
    )
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if config.patch_size < 1:
        raise ValueError("patch_size must be positive.")
    if config.correction_batch_size < 1:
        raise ValueError("correction_batch_size must be positive.")
    for setting_name in (
        "cleanup_corrected_after_registration",
        "cleanup_registered_after_patching",
    ):
        if not isinstance(getattr(config.storage, setting_name), bool):
            raise ValueError(f"storage.{setting_name} must be true or false.")
    if not 2 <= config.smoke_row <= 7 or not 2 <= config.smoke_column <= 11:
        raise ValueError("Smoke-test well must be within occupied rows 2-7 and columns 2-11.")
    if len(config.test_wells) != 4:
        raise ValueError("test_wells must contain exactly four physical wells.")
    test_coordinates = {
        (item.plate_name, item.row, item.column) for item in config.test_wells
    }
    if len(test_coordinates) != 4:
        raise ValueError("test_wells must contain four unique physical wells.")
    if {item.plate_name for item in config.test_wells} != {"sc_rep5-6", "sc_rep7-8"}:
        raise ValueError("test_wells must include both sc_rep5-6 and sc_rep7-8.")
    counts = {
        plate_name: sum(item.plate_name == plate_name for item in config.test_wells)
        for plate_name in ("sc_rep5-6", "sc_rep7-8")
    }
    if counts != {"sc_rep5-6": 2, "sc_rep7-8": 2}:
        raise ValueError("test_wells must contain exactly two wells from each plate.")
    for item in config.test_wells:
        if not 2 <= item.row <= 7 or not 2 <= item.column <= 11:
            raise ValueError(
                f"Test well {item.plate_name} r{item.row:02d}_c{item.column:02d} "
                "is outside occupied rows 2-7 and columns 2-11."
            )
    if config.registration.level not in {"field_level", "patch_level"}:
        raise ValueError(
            "registration.level must be 'field_level' or 'patch_level'."
        )
    if config.registration.reference_measurement != 1:
        raise ValueError("This production configuration requires measurement 1 reference.")
    if config.registration.downsample < 1 or config.registration.upsample_factor < 1:
        raise ValueError("Registration downsample and upsample_factor must be positive.")
    for value, name in (
        (config.registration.min_common_overlap_fraction, "min_common_overlap_fraction"),
        (config.registration.min_global_correlation, "min_global_correlation"),
        (config.registration.min_field_correlation, "min_field_correlation"),
        (
            config.registration.patch_level_min_overlap_fraction,
            "patch_level_min_overlap_fraction",
        ),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one.")


def config_as_dict(config: PipelineConfig) -> dict:
    return asdict(config)


def with_test_mode(config: PipelineConfig, enabled: bool) -> PipelineConfig:
    """Return an immutable config with the requested execution mode."""
    updated = replace(config, test_mode=bool(enabled))
    validate_config(updated)
    return updated


def with_execution_mode(
    config: PipelineConfig, *, test_mode: bool, registration_level: str
) -> PipelineConfig:
    """Return a validated immutable config for one submitted run mode."""
    updated = replace(
        config,
        test_mode=bool(test_mode),
        registration=replace(config.registration, level=registration_level),
    )
    validate_config(updated)
    return updated
