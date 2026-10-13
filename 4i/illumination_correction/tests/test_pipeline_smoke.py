"""Small end-to-end test of shared sampling, metrics, and output layout."""

from __future__ import annotations

from collections import OrderedDict
import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import imageio.v3 as iio
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

import illumination_correction_eval as cli  # noqa: E402


def _write_channel(root: Path, plate: str, channel: str, seed: int) -> Path:
    directory = (
        root
        / f"{plate}_measurement_1"
        / "max_projection"
        / channel
    )
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:32, :32]
    illumination = 0.8 + 0.4 * (xx / 31.0)
    for row, column in ((2, 2), (2, 3)):
        for field in range(1, 37):
            image = (
                500
                + 10 * field
                + 50 * np.sin((xx + field) / 5.0)
                + 30 * np.cos((yy + column) / 6.0)
                + rng.normal(0, 3, (32, 32))
            ) * illumination
            iio.imwrite(
                directory
                / f"r{row:02d}_c{column:02d}_f{field:02d}_t01_ch01.tiff",
                np.clip(image, 0, 65535).astype(np.uint16),
            )
    return directory


def test_two_marker_paired_plate_pipeline(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    marker_paths = []
    for marker_number, channel in enumerate(
        ("channel_2_alexa_488", "channel_3_alexa_555_isabel"), start=1
    ):
        marker_paths.append(
            (
                _write_channel(
                    data_root, "sc_rep5-6", channel, seed=marker_number * 10
                ),
                _write_channel(
                    data_root, "sc_rep7-8", channel, seed=marker_number * 10 + 1
                ),
            )
        )

    selected_training_coordinates = []

    def fake_fit(paths, *, label, settings):
        selected_training_coordinates.append(
            [path.name.split("_ch")[0] for path in paths]
        )
        profile = np.ones((32, 32), dtype=np.float32)
        harmony = {
            "background_profile": profile,
            "foreground_profile": profile,
            "background_profile_small": profile[::4, ::4],
            "foreground_profile_small": profile[::4, ::4],
            "background_split_half_range_error_percent": 1.0,
            "foreground_split_half_range_error_percent": 2.0,
            "sampled_path_count": len(paths),
        }
        basic_diagnostics = {
            "get_darkfield": False,
            "sort_intensity": True,
            "status": "validated_finite",
        }
        return SimpleNamespace(
            basic=object(),
            basic_diagnostics=basic_diagnostics,
            harmony=harmony,
        )

    def fake_apply(raw, models):
        raw = np.asarray(raw, dtype=np.float32)
        x_profile = np.linspace(0.85, 1.15, raw.shape[2], dtype=np.float32)
        return OrderedDict(
            [
                ("Original", raw),
                ("BaSiC", raw / x_profile[None, None, :]),
                ("Harmony-like", raw / np.sqrt(x_profile)[None, None, :]),
            ]
        )

    monkeypatch.setattr(cli, "fit_marker_models", fake_fit)
    monkeypatch.setattr(cli, "apply_all_corrections", fake_apply)
    output = tmp_path / "output"
    marker_arguments = []
    for first, second in marker_paths:
        marker_arguments.extend(["--marker", str(first), str(second)])
    shared_arguments = [
            "--output-dir",
            str(output),
            "--training-field-count",
            "8",
            "--visualization-well-count",
            "2",
            "--display-downsample",
            "2",
            "--figure-dpi",
            "40",
            "--zoom-along-half-width",
            "12",
            "--zoom-across-half-width",
            "8",
            "--seam-score-band",
            "4",
            "--even-binning",
            "2",
        ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "illumination_correction_eval.py",
            *marker_arguments,
            *shared_arguments,
            "--prepare-only",
        ],
    )
    assert cli.main() == 0
    sampling_manifest = output / "shared_sampling_manifest.csv"
    for marker_index in range(2):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "illumination_correction_eval.py",
                *marker_arguments,
                *shared_arguments,
                "--marker-index",
                str(marker_index),
                "--sampling-manifest",
                str(sampling_manifest),
            ],
        )
        assert cli.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "illumination_correction_eval.py",
            *marker_arguments,
            "--output-dir",
            str(output),
            "--aggregate-only",
        ],
    )
    assert cli.main() == 0
    assert len(selected_training_coordinates) == 2
    assert selected_training_coordinates[0] == selected_training_coordinates[1]

    marker_names = (
        "measurement_1_channel_2_alexa_488",
        "measurement_1_channel_3_alexa_555_isabel",
    )
    for marker_name in marker_names:
        marker_dir = output / marker_name
        assert len(list((marker_dir / "visualisation" / "full_6x6").glob("*.png"))) == 2
        assert len(list((marker_dir / "visualisation" / "zoom").glob("*.png"))) == 2
        with (marker_dir / "metrics" / "per_field_metrics.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 2 * 3 * 36
        with (marker_dir / "metrics" / "per_well_metrics.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 2 * 3
        with (marker_dir / "metrics" / "average_metrics.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 3
    with (output / "all_markers_average_metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 3
