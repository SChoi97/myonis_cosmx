#!/usr/bin/env python3
"""Create the reviewed illumination-method handoff from evaluation metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from utils.manifests import read_csv, sha256_json, sha256_path, write_csv
from utils.registry import ACQUISITIONS
from utils.review_workbook import read_review_workbook


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--evaluation-dir", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--review-workbook",
        type=Path,
        help="Workbook containing the 28 marker and 12 DAPI method choices.",
    )
    return result


def finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def recommendation(method_rows: dict[str, dict[str, str]]) -> tuple[str, str, float, float]:
    basic = method_rows.get("BaSiC", {})
    harmony = method_rows.get("Harmony-like", {})
    scores = {}
    for method, row in (("BaSiC", basic), ("Harmony-like", harmony)):
        boundary = finite(row, "boundary_discontinuity_mean")
        prominence = finite(row, "p_magnitude_mean")
        scores[method] = (
            0.6 * math.log(max(boundary, 1e-12))
            + 0.4 * math.log(max(prominence, 1e-12))
            if math.isfinite(boundary) and math.isfinite(prominence)
            else float("inf")
        )
    winner = min(scores, key=scores.get)
    if not math.isfinite(scores[winner]):
        return "", "missing_or_nonfinite_metrics", scores["BaSiC"], scores["Harmony-like"]
    gap = abs(scores["BaSiC"] - scores["Harmony-like"])
    review_note = "close_scores" if gap < math.log(1.10) else "metric_recommendation"
    boundary_winner = min(
        ("BaSiC", "Harmony-like"),
        key=lambda method: finite(method_rows.get(method, {}), "boundary_discontinuity_mean"),
    )
    prominence_winner = min(
        ("BaSiC", "Harmony-like"),
        key=lambda method: finite(method_rows.get(method, {}), "p_magnitude_mean"),
    )
    if boundary_winner != prominence_winner:
        review_note = "metrics_disagree"
    return winner, review_note, scores["BaSiC"], scores["Harmony-like"]


def main() -> int:
    args = parser().parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    metrics_path = evaluation_dir / "all_markers_average_metrics.csv"
    completion_path = evaluation_dir / "marker_completion_status.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            "The completed evaluation summary does not exist: "
            f"{metrics_path}. Run `sbatch illumination_correction_eval.sh` from "
            "4i/illumination_correction and wait for both the 40-task array and "
            "its aggregation job to finish before submitting preprocessing."
        )
    rows = read_csv(metrics_path)
    if not rows:
        raise RuntimeError(
            f"Evaluation summary is empty: {metrics_path}. Inspect "
            f"{completion_path} and the illumination worker logs; no evaluated "
            "model can be approved or applied yet."
        )
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["marker"], {})[row["method"]] = row
    output_rows = []
    missing = []
    for acquisition in ACQUISITIONS:
        method_rows = grouped.get(acquisition.evaluation_name)
        if not method_rows:
            missing.append(acquisition.evaluation_name)
            continue
        recommended, note, basic_score, harmony_score = recommendation(method_rows)
        metrics_dir = evaluation_dir / acquisition.evaluation_name / "metrics"
        basic_model = metrics_dir / "basic_model"
        harmony_profile = metrics_dir / "harmony_profiles.npz"
        harmony_diagnostics = metrics_dir / "harmony_model_diagnostics.json"
        training_paths_file = metrics_dir / "training_image_paths.csv"
        shared_sampling_file = evaluation_dir / "shared_sampling_manifest.csv"
        metadata_rows = read_csv(metrics_dir / "marker_metadata.csv")
        training_rows = read_csv(training_paths_file)
        required = (
            basic_model,
            harmony_profile,
            harmony_diagnostics,
            training_paths_file,
            shared_sampling_file,
        )
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            raise FileNotFoundError(
                f"{acquisition.evaluation_name} lacks model artifacts: {absent}"
            )
        basic_row = method_rows["BaSiC"]
        harmony_row = method_rows["Harmony-like"]
        harmony_profile_hash = sha256_path(harmony_profile)
        harmony_diagnostics_hash = sha256_path(harmony_diagnostics)
        output_rows.append(
            {
                "acquisition_id": acquisition.acquisition_id,
                "evaluation_name": acquisition.evaluation_name,
                "measurement": acquisition.measurement,
                "source_channel": acquisition.source_channel,
                "fluorophore": acquisition.fluorophore,
                "marker": acquisition.marker,
                "recommended_method": recommended,
                "recommendation_note": note,
                "chosen_method": "",
                "review_status": "pending",
                "basic_artifact_score": basic_score,
                "harmony_artifact_score": harmony_score,
                "basic_boundary_mean": basic_row.get("boundary_discontinuity_mean", ""),
                "harmony_boundary_mean": harmony_row.get("boundary_discontinuity_mean", ""),
                "basic_p_magnitude_mean": basic_row.get("p_magnitude_mean", ""),
                "harmony_p_magnitude_mean": harmony_row.get("p_magnitude_mean", ""),
                "basic_model_path": str(basic_model),
                "basic_model_sha256": sha256_path(basic_model),
                "harmony_profile_path": str(harmony_profile),
                "harmony_profile_sha256": harmony_profile_hash,
                "harmony_diagnostics_path": str(harmony_diagnostics),
                "harmony_diagnostics_sha256": harmony_diagnostics_hash,
                "harmony_model_sha256": sha256_json(
                    {
                        "profile": harmony_profile_hash,
                        "diagnostics": harmony_diagnostics_hash,
                    }
                ),
                "training_field_count": len(training_rows),
                "training_paths_path": str(training_paths_file),
                "training_paths_sha256": sha256_path(training_paths_file),
                "shared_sampling_manifest_path": str(shared_sampling_file),
                "shared_sampling_manifest_sha256": sha256_path(shared_sampling_file),
                "evaluation_plate_1_path": metadata_rows[0].get("plate_1_path", "") if metadata_rows else "",
                "evaluation_plate_2_path": metadata_rows[0].get("plate_2_path", "") if metadata_rows else "",
            }
        )
    if missing:
        completion_rows = read_csv(completion_path)
        completed = sum(row.get("status") == "complete" for row in completion_rows)
        completion_summary = (
            f" Completion table reports {completed}/{len(ACQUISITIONS)} complete "
            f"acquisitions at {completion_path}."
            if completion_rows
            else f" No completion table was found at {completion_path}."
        )
        raise RuntimeError(
            f"Evaluation contains {len(grouped)}/{len(ACQUISITIONS)} required "
            f"acquisitions and {len(rows)}/120 expected method rows."
            + completion_summary
            + " Missing: "
            + ", ".join(missing)
        )
    if args.review_workbook:
        choices = read_review_workbook(args.review_workbook)
        expected = {item.acquisition_id for item in ACQUISITIONS}
        if set(choices) != expected:
            raise ValueError(
                "Review workbook must contain exactly all 40 acquisitions; "
                f"missing={sorted(expected - set(choices))}, "
                f"unexpected={sorted(set(choices) - expected)}"
            )
        pending = []
        for row in output_rows:
            choice = choices[row["acquisition_id"]]
            row.update(choice)
            row["review_workbook_sha256"] = sha256_path(args.review_workbook)
            if choice["review_status"] != "approved":
                pending.append(row["acquisition_id"])
        if pending:
            raise ValueError(
                "Fill chosen_method for every workbook row before submitting. "
                "Pending acquisitions: " + ", ".join(pending)
            )
    write_csv(args.output, output_rows)
    if args.review_workbook:
        print(f"Wrote {len(output_rows)} approved workbook choices to {args.output}")
    else:
        print(f"Wrote {len(output_rows)} pending review rows to {args.output}")
        print("Fill chosen_method and set review_status=approved only after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
