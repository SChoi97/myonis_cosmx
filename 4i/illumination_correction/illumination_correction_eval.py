#!/usr/bin/env python3
"""Structured paired-plate evaluation of microscopy illumination correction."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

from utils.corrections import (
    CorrectionSettings,
    apply_all_corrections,
    fit_marker_models,
)
from utils.io import (
    MarkerSpec,
    index_markers,
    load_paths,
    marker_spec,
    paths_for_well,
    read_marker_config,
)
from utils.layout import tile_fields
from utils.metrics import EvenSettings, evaluate_method, format_metrics
from utils.reporting import (
    read_rows,
    save_basic_diagnostics,
    save_model_diagnostics,
    summarize_metrics,
    write_rows,
)
from utils.sampling import (
    build_shared_sampling_plan,
    load_sampling_manifest,
    training_paths_for_marker,
    validate_plan_for_marker,
    write_sampling_manifest,
)
from utils.visualization import save_full_well, save_zoom, save_zoom_2x2


# These are all 12 measurement-specific DAPI acquisitions plus the 28
# non-DAPI stainings in the August 2026 experiment sheet.
# Measurement 9 Alexa 488 is deliberately absent because the 12 August
# acquisition was marked as low quality. Output folder names remain based on
# measurement and channel, while display_label is shown in figures and tables.
DEFAULT_NON_DAPI_CHANNELS = [
    (1, "channel_2_alexa_488", "eMYHC"),
    (1, "channel_3_alexa_555_isabel", "Lamin A/C"),
    (1, "channel_4_alexa_647", "53BP1"),
    (2, "channel_2_alexa_488", "RNA Pol II"),
    (2, "channel_3_alexa_555_isabel", "MyHC"),
    (2, "channel_4_alexa_647", "ATF3"),
    (3, "channel_2_alexa_488", "p53 DO-1"),
    (3, "channel_3_alexa_555_isabel", "MyoG"),
    (3, "channel_4_alexa_647", "BAF"),
    (4, "channel_2_alexa_488", "YAP1"),
    (4, "channel_3_alexa_555_isabel", "H3K27ac"),
    (4, "channel_4_alexa_647", "H3K27me3"),
    (5, "channel_2_alexa_488", "a-tubulin"),
    (5, "channel_3_alexa_555_isabel", "SC35"),
    (5, "channel_4_alexa_647", "c-CASP3"),
    (6, "channel_2_alexa_647", "cGAS"),
    (7, "channel_2_alexa_647", "p-MLKL"),
    (8, "channel_2_alexa_647", "PTEN"),
    (9, "channel_3_alexa_647", "p-4EBP1"),
    (10, "channel_2_alexa_488", "CKAP4"),
    (10, "channel_3_alexa_555_isabel", "GM130"),
    (10, "channel_4_alexa_647", "c-JUN"),
    (11, "channel_2_alexa_488", "HSP60"),
    (11, "channel_3_alexa_555_isabel", "b-Actin"),
    (11, "channel_4_alexa_647", "CTGF"),
    (12, "channel_2_alexa_488", "ATF6"),
    (12, "channel_3_alexa_555_isabel", "p-S6RP"),
    (12, "channel_4_alexa_647", "PEX14"),
]

DEFAULT_MARKER_CHANNELS = [
    item
    for measurement in range(1, 13)
    for item in (
        (
            measurement,
            "channel_1_dapi",
            f"DAPI (measurement {measurement})",
        ),
        *(channel for channel in DEFAULT_NON_DAPI_CHANNELS if channel[0] == measurement),
    )
]

DEFAULT_DATA_ROOT = (
    "/nemo/lab/tedescos/home/users/chois1/HTS/phoenix_plus/4i_experiment"
)


def built_in_marker_paths() -> list[dict]:
    """Build the two explicit plate paths for every default marker channel."""
    markers = []
    for measurement, channel, display_label in DEFAULT_MARKER_CHANNELS:
        markers.append(
            {
                "name": None,
                "display_label": display_label,
                "plate_1_path": (
                    f"{DEFAULT_DATA_ROOT}/sc_rep5-6_measurement_{measurement}/"
                    f"max_projection/{channel}"
                ),
                "plate_2_path": (
                    f"{DEFAULT_DATA_ROOT}/sc_rep7-8_measurement_{measurement}/"
                    f"max_projection/{channel}"
                ),
            }
        )
    return markers


DEFAULT_MARKERS = built_in_marker_paths()


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BaSiC and Harmony-like jointly across two plates per marker, "
            "using one shared coordinate sample across every marker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--marker-config",
        type=Path,
        help=(
            "TSV/CSV with plate_1_path and plate_2_path columns; name and "
            "display_label are optional."
        ),
    )
    input_group.add_argument(
        "--marker",
        action="append",
        nargs=2,
        metavar=("PLATE_1_PATH", "PLATE_2_PATH"),
        help=(
            "Paired channel paths for one marker. Repeat once per marker. "
            "Names are inferred from measurement and channel folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paired_plate_evaluation"),
    )
    parser.add_argument("--training-field-count", type=positive_int, default=600)
    parser.add_argument(
        "--visualization-well-count", type=positive_int, default=30
    )
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--display-downsample", type=positive_int, default=4)
    parser.add_argument("--figure-dpi", type=positive_int, default=180)
    parser.add_argument("--zoom-along-half-width", type=positive_int, default=300)
    parser.add_argument("--zoom-across-half-width", type=positive_int, default=180)
    parser.add_argument("--seam-score-band", type=positive_int, default=16)
    parser.add_argument("--even-binning", type=positive_int, default=4)
    parser.add_argument("--even-edge-threshold", type=float, default=3.0 / 8.0)
    parser.add_argument("--harmony-block-size", type=positive_int, default=64)
    parser.add_argument("--harmony-noise-window", type=positive_int, default=9)
    parser.add_argument("--harmony-group-size", type=positive_int, default=128)
    parser.add_argument("--background-regularization", type=float, default=0.2)
    parser.add_argument("--foreground-regularization", type=float, default=20.0)
    parser.add_argument(
        "--basic-darkfield",
        action="store_true",
        help=(
            "Request BaSiC darkfield estimation. Disabled by default because "
            "it can produce unstable/non-finite fits; invalid fits retry safely."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create shared manifests and exit without fitting a marker.",
    )
    mode_group.add_argument(
        "--marker-index",
        type=int,
        help="Run only this zero-based marker index using --sampling-manifest.",
    )
    mode_group.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Combine completed per-marker averages and write completion status.",
    )
    mode_group.add_argument(
        "--print-marker-count",
        action="store_true",
        help="Print the configured marker count and exit.",
    )
    parser.add_argument(
        "--sampling-manifest",
        type=Path,
        help="Prepared shared_sampling_manifest.csv used by a marker worker.",
    )
    return parser


def resolve_markers(args, parser: argparse.ArgumentParser) -> list[MarkerSpec]:
    try:
        if args.marker_config:
            markers = read_marker_config(args.marker_config)
        elif args.marker:
            markers = [marker_spec(first, second) for first, second in args.marker]
        else:
            markers = [
                marker_spec(
                    item["plate_1_path"],
                    item["plate_2_path"],
                    name=item.get("name"),
                    display_label=item.get("display_label"),
                )
                for item in DEFAULT_MARKERS
            ]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if not markers:
        parser.error("At least one paired marker must be supplied.")
    return markers


def write_marker_manifest(path: Path, markers: list[MarkerSpec]) -> None:
    rows = [
        {
            "marker": marker.name,
            "display_label": marker.display_label,
            "plate_slot": slot,
            "channel_directory": str(marker.plate_paths[slot - 1]),
        }
        for marker in markers
        for slot in (1, 2)
    ]
    write_rows(path, rows)


def write_training_paths(
    path: Path,
    training_paths: list[Path],
    plan,
    marker_indices,
    marker: MarkerSpec,
) -> None:
    rows = []
    for sequence, (coordinate, image_path) in enumerate(
        zip(plan.training_fields, training_paths), start=1
    ):
        slot, row, column, field = coordinate
        rows.append(
            {
                "marker": marker.name,
                "display_label": marker.display_label,
                "sequence": sequence,
                "plate_slot": slot,
                "plate_name": marker_indices[slot - 1].plate_name,
                "row": row,
                "column": column,
                "field": field,
                "image_path": str(image_path),
            }
        )
    write_rows(path, rows)


def process_marker(
    marker_number: int,
    marker_count: int,
    marker: MarkerSpec,
    marker_indices,
    plan,
    output_root: Path,
    correction_settings: CorrectionSettings,
    even_settings: EvenSettings,
    args,
) -> list[dict]:
    print("\n" + "=" * 96)
    print(
        f"Marker {marker_number}/{marker_count}: "
        f"{marker.name} ({marker.display_label})"
    )
    marker_dir = output_root / marker.name
    metrics_dir = marker_dir / "metrics"
    full_dir = marker_dir / "visualisation" / "full_6x6"
    zoom_dir = marker_dir / "visualisation" / "zoom"
    zoom_2x2_dir = marker_dir / "visualisation" / "zoom_2x2"
    for directory in (metrics_dir, full_dir, zoom_dir, zoom_2x2_dir):
        directory.mkdir(parents=True, exist_ok=True)

    training_paths = training_paths_for_marker(marker_indices, plan)
    write_training_paths(
        metrics_dir / "training_image_paths.csv",
        training_paths,
        plan,
        marker_indices,
        marker,
    )
    models = fit_marker_models(
        training_paths,
        label=f"{marker.name} | {marker.display_label}",
        settings=correction_settings,
    )
    # Persist the exact evaluated model so production correction reloads the
    # approved profile instead of fitting a potentially different profile.
    models.basic.save_model(metrics_dir / "basic_model", overwrite=True)
    save_basic_diagnostics(metrics_dir, models.basic_diagnostics)
    save_model_diagnostics(metrics_dir, models.harmony)
    write_rows(
        metrics_dir / "marker_metadata.csv",
        [
            {
                "marker": marker.name,
                "display_label": marker.display_label,
                "plate_1_path": str(marker.plate_paths[0]),
                "plate_2_path": str(marker.plate_paths[1]),
                "training_field_count": len(plan.training_fields),
                "visualization_well_count": len(plan.visualization_wells),
                "basic_darkfield": models.basic_diagnostics["get_darkfield"],
                "basic_sort_intensity": models.basic_diagnostics["sort_intensity"],
                "basic_validation_status": models.basic_diagnostics["status"],
            }
        ],
    )

    per_field_rows = []
    per_well_rows = []
    for panel_index, (plate_slot, row, column) in enumerate(
        plan.visualization_wells, start=1
    ):
        index = marker_indices[plate_slot - 1]
        plate_name = index.plate_name
        print(
            f"[{marker.name}] panel {panel_index}/{len(plan.visualization_wells)}: "
            f"{plate_name} r{row:02d}_c{column:02d}"
        )
        raw_fields = load_paths(paths_for_well(index, (row, column)))
        corrected = apply_all_corrections(raw_fields, models)
        mosaics = {
            method: tile_fields(fields) for method, fields in corrected.items()
        }

        current_well_rows = []
        current_field_rows = []
        metadata = {
            "marker": marker.name,
            "display_label": marker.display_label,
            "visualization_index": panel_index,
            "plate_slot": plate_slot,
            "plate_name": plate_name,
            "row": row,
            "column": column,
        }
        for method, fields in corrected.items():
            well_metrics, field_metrics = evaluate_method(
                fields,
                mosaics[method],
                method=method,
                settings=even_settings,
            )
            current_well_rows.append({**metadata, **well_metrics})
            current_field_rows.extend(
                {**metadata, **field_row} for field_row in field_metrics
            )

        title = (
            f"{marker.name} ({marker.display_label}) | {plate_name} | "
            f"r{row:02d}_c{column:02d} | panel {panel_index:03d}"
        )
        file_stem = (
            f"panel_{panel_index:03d}_{plate_name}_r{row:02d}_c{column:02d}"
        )
        save_full_well(
            mosaics,
            title=title,
            metric_rows=current_well_rows,
            output_path=full_dir / f"{file_stem}_full_6x6.png",
            display_downsample=args.display_downsample,
            dpi=args.figure_dpi,
        )
        save_zoom_2x2(
            mosaics,
            title=title,
            metric_rows=current_well_rows,
            output_path=zoom_2x2_dir / f"{file_stem}_zoom_2x2.png",
            display_downsample=args.display_downsample,
            dpi=args.figure_dpi,
        )
        seam = save_zoom(
            mosaics,
            title=title,
            metric_rows=current_well_rows,
            output_path=zoom_dir / f"{file_stem}_maximum_disparity_zoom.png",
            along_half_width=args.zoom_along_half_width,
            across_half_width=args.zoom_across_half_width,
            score_band=args.seam_score_band,
            dpi=args.figure_dpi,
        )
        for metric_row in current_well_rows:
            metric_row.update(
                {
                    "boundary_discontinuity": seam["method_discontinuity"][
                        metric_row["method"]
                    ],
                    "selected_seam_disparity_score": seam["disparity_score"],
                    "seam_orientation": seam["orientation"],
                    "seam_field_1": seam["field_pair"][0],
                    "seam_field_2": seam["field_pair"][1],
                    "seam_score_band": seam["score_band"],
                }
            )
            print(
                f"  {metric_row['method']:<13} {format_metrics(metric_row)} | "
                f"boundary={metric_row['boundary_discontinuity']:.6f}"
            )
        per_well_rows.extend(current_well_rows)
        per_field_rows.extend(current_field_rows)
        averages = summarize_metrics(
            marker.name,
            marker.display_label,
            per_field_rows,
            per_well_rows,
            models.harmony,
        )
        # Save after every completed well so long cluster runs remain recoverable.
        write_rows(metrics_dir / "per_field_metrics.csv", per_field_rows)
        write_rows(metrics_dir / "per_well_metrics.csv", per_well_rows)
        write_rows(metrics_dir / "average_metrics.csv", averages)
        del raw_fields, corrected, mosaics
        gc.collect()

    print(
        f"[{marker.name}] Harmony-like split-half errors: "
        f"background="
        f"{models.harmony['background_split_half_range_error_percent']:.3f}%, "
        f"foreground="
        f"{models.harmony['foreground_split_half_range_error_percent']:.3f}%"
    )
    return summarize_metrics(
        marker.name,
        marker.display_label,
        per_field_rows,
        per_well_rows,
        models.harmony,
    )


def aggregate_completed_markers(
    markers: list[MarkerSpec], output_root: Path
) -> tuple[list[dict], list[dict]]:
    """Combine independent worker outputs and record missing/failed markers."""
    sampling_path = output_root / "shared_sampling_manifest.csv"
    expected_wells = (
        len(load_sampling_manifest(sampling_path).visualization_wells)
        if sampling_path.is_file()
        else None
    )
    combined = []
    status_rows = []
    for marker_index, marker in enumerate(markers):
        average_path = output_root / marker.name / "metrics" / "average_metrics.csv"
        basic_diagnostics_path = (
            output_root
            / marker.name
            / "metrics"
            / "basic_model_diagnostics.json"
        )
        if average_path.is_file() and basic_diagnostics_path.is_file():
            rows = read_rows(average_path)
            with basic_diagnostics_path.open(encoding="utf-8") as stream:
                basic_diagnostics = json.load(stream)
            basic_rows = [row for row in rows if row.get("method") == "BaSiC"]
            basic_is_finite = bool(basic_rows) and all(
                math.isfinite(float(basic_rows[0][column]))
                for column in (
                    "edge_energy_mean",
                    "p_positive_mean",
                    "p_negative_mean",
                    "p_magnitude_mean",
                    "boundary_discontinuity_mean",
                )
            )
            expected_counts_match = expected_wells is not None and all(
                int(row["visualization_well_count"]) == expected_wells
                and int(row["field_observation_count"]) == expected_wells * 36
                for row in rows
            )
            if (
                len(rows) == 3
                and basic_diagnostics.get("status") == "validated_finite"
                and basic_is_finite
                and expected_counts_match
            ):
                status = "complete"
                combined.extend(rows)
            else:
                status = "invalid_or_incomplete_metrics"
        else:
            rows = []
            status = "missing_validated_output"
        status_rows.append(
            {
                "marker_index": marker_index,
                "marker": marker.name,
                "display_label": marker.display_label,
                "status": status,
                "average_metrics_path": str(average_path),
                "basic_diagnostics_path": str(basic_diagnostics_path),
                "method_row_count": len(rows),
                "expected_visualization_well_count": expected_wells,
            }
        )
    write_rows(output_root / "all_markers_average_metrics.csv", combined)
    write_rows(output_root / "marker_completion_status.csv", status_rows)
    return combined, status_rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    markers = resolve_markers(args, parser)
    if args.print_marker_count:
        print(len(markers))
        return 0
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        combined, statuses = aggregate_completed_markers(markers, output_root)
        complete = sum(row["status"] == "complete" for row in statuses)
        print(
            f"Aggregated {len(combined)} method rows from "
            f"{complete}/{len(markers)} completed markers."
        )
        print(f"Completion table: {output_root / 'marker_completion_status.csv'}")
        if complete != len(markers):
            incomplete = [
                f"{row['marker_index']}:{row['marker']} ({row['status']})"
                for row in statuses
                if row["status"] != "complete"
            ]
            print(
                "Evaluation is incomplete; production preprocessing cannot start. "
                "Incomplete array entries: " + ", ".join(incomplete),
                file=sys.stderr,
            )
            return 2
        return 0

    try:
        even_settings = EvenSettings(
            edge_threshold=args.even_edge_threshold,
            binning=args.even_binning,
        )
        if args.marker_index is not None:
            if not 0 <= args.marker_index < len(markers):
                raise ValueError(
                    f"Marker index must be in [0, {len(markers) - 1}]; "
                    f"received {args.marker_index}."
                )
            if args.sampling_manifest is None:
                raise ValueError(
                    "--marker-index requires --sampling-manifest from a "
                    "completed --prepare-only run."
                )
            plan = load_sampling_manifest(args.sampling_manifest)
            selected_markers = [markers[args.marker_index]]
            indexed = index_markers(selected_markers)
            validate_plan_for_marker(indexed[selected_markers[0].name], plan)
        else:
            write_marker_manifest(output_root / "marker_inputs.csv", markers)
            indexed = index_markers(markers)
            plan = build_shared_sampling_plan(
                indexed,
                training_field_count=args.training_field_count,
                visualization_well_count=args.visualization_well_count,
                seed=args.random_seed,
            )
            write_sampling_manifest(
                output_root / "shared_sampling_manifest.csv", plan, indexed
            )
            selected_markers = markers
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Output root: {output_root}")
    print(
        f"Shared plan: {len(plan.training_fields)} model-training fields and "
        f"{len(plan.visualization_wells)} visualization wells across two plates."
    )
    if args.prepare_only:
        print("Preparation complete; no correction models were fitted.")
        return 0

    correction_settings = CorrectionSettings(
        random_seed=args.random_seed,
        basic_get_darkfield=args.basic_darkfield,
        basic_sort_intensity=True,
        harmony_block_size=args.harmony_block_size,
        harmony_noise_window=args.harmony_noise_window,
        harmony_group_size=args.harmony_group_size,
        harmony_background_regularization=args.background_regularization,
        harmony_foreground_regularization=args.foreground_regularization,
    )
    all_average_rows = []
    for marker in selected_markers:
        marker_number = markers.index(marker) + 1
        averages = process_marker(
            marker_number,
            len(markers),
            marker,
            indexed[marker.name],
            plan,
            output_root,
            correction_settings,
            even_settings,
            args,
        )
        all_average_rows.extend(averages)
        if args.marker_index is None:
            write_rows(
                output_root / "all_markers_average_metrics.csv", all_average_rows
            )
        gc.collect()
    if args.marker_index is None:
        print("\nEvaluation complete.")
        print(f"Combined averages: {output_root / 'all_markers_average_metrics.csv'}")
    else:
        print(
            f"\nMarker worker {args.marker_index} complete: "
            f"{selected_markers[0].display_label}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
