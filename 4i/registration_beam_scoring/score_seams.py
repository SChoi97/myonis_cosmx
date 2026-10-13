#!/usr/bin/env python3
"""Score old source-field joins in cached field-registered quantitative patches.

The independent scores intentionally have no exclusion threshold. Inspect the
annotated examples before deciding how to classify patches for quantification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from progress import progress

HERE = Path(__file__).resolve().parent
FIELD_SIZE = 1080
WIDTHS = (8, 12, 16, 20, 24, 32)
ALONG_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
MIN_CONTEXT = max(WIDTHS) + max(abs(v) for v in ALONG_OFFSETS)
SCORE_VERSION = "seam_scores_20261002_v2"
SCORE_METRICS = (
    "duplicate_ncc", "duplicate_ncc_excess", "matched_window_fraction",
    "matched_bright_runs", "matched_bright_pixel_fraction",
    "duplicate_foreground_fraction", "foreground_burden_fraction", "jump_ratio",
)


def host_path(path: str | Path) -> Path:
    """Resolve the same Nemo share on the cluster or mounted macOS host."""
    path = Path(path)
    if path.exists():
        return path
    value = str(path)
    pairs = (
        ("/Volumes/lab-tedescos/", "/nemo/lab/tedescos/"),
        ("/nemo/lab/tedescos/", "/Volumes/lab-tedescos/"),
    )
    for source, target in pairs:
        if value.startswith(source):
            alternate = Path(target + value[len(source):])
            if alternate.exists():
                return alternate
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            if rows:
                names = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def two_way_center(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return values - values.mean(axis=0, keepdims=True) - values.mean(axis=1, keepdims=True) + values.mean()


def ncc(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size < 64:
        return float("nan")
    a = two_way_center(first)
    b = two_way_center(second)
    denom = float(np.sqrt(np.sum(a * a, dtype=np.float64) * np.sum(b * b, dtype=np.float64)))
    return float(np.sum(a * b, dtype=np.float64) / denom) if denom > 1e-8 else float("nan")


def paired_strips(view: np.ndarray, cut: int, width: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    length = view.shape[0] - abs(offset)
    if offset >= 0:
        return view[:length, cut - width:cut], view[offset:offset + length, cut:cut + width]
    return view[-offset:-offset + length, cut - width:cut], view[:length, cut:cut + width]


def best_repeat(view: np.ndarray, cut: int) -> dict:
    if cut < MIN_CONTEXT or cut > view.shape[1] - MIN_CONTEXT:
        return {"ncc": float("nan"), "width": 0, "offset": 0}
    candidates = []
    for width in WIDTHS:
        a, b = paired_strips(view, cut, width)
        candidates.append((ncc(a, b), width, 0))
    finite = [item for item in candidates if math.isfinite(item[0])]
    if not finite:
        return {"ncc": float("nan"), "width": 0, "offset": 0}
    _, best_width, _ = max(finite)
    for offset in ALONG_OFFSETS:
        if offset == 0:
            continue
        a, b = paired_strips(view, cut, best_width, offset)
        candidates.append((ncc(a, b), best_width, offset))
    value, width, offset = max((item for item in candidates if math.isfinite(item[0])), key=lambda item: item[0])
    return {"ncc": value, "width": width, "offset": offset}


def foreground_threshold(image: np.ndarray) -> float:
    sample = np.asarray(image, dtype=np.float32)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return max(float(np.percentile(sample, 82)), median + 3.0 * mad)


def count_runs(active: np.ndarray, min_length: int = 3) -> int:
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.sum(ends - starts >= min_length))


def score_join(image: np.ndarray, orientation: str, cut: int) -> dict:
    """Return independent repeated-content, bright-fragment and jump scores."""
    view = np.asarray(image, dtype=np.float32)
    if orientation == "horizontal":
        view = view.T
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError(orientation)
    if cut < MIN_CONTEXT or cut > view.shape[1] - MIN_CONTEXT:
        return {"score_status": "insufficient_context"}
    peak = best_repeat(view, cut)
    if not math.isfinite(peak["ncc"]):
        return {"score_status": "no_texture"}
    controls = [
        best_repeat(view, control)["ncc"]
        for control in (cut - 128, cut - 80, cut + 80, cut + 128)
        if MIN_CONTEXT <= control <= view.shape[1] - MIN_CONTEXT
    ]
    controls = [value for value in controls if math.isfinite(value)]
    control = float(np.median(controls)) if controls else float("nan")
    width, offset = peak["width"], peak["offset"]
    a, b = paired_strips(view, cut, width, offset)
    threshold = foreground_threshold(image)
    both_bright = (a > threshold) & (b > threshold)
    either_bright = (a > threshold) | (b > threshold)
    row_pair_pixels = np.sum(both_bright, axis=1)
    run_count = count_runs(row_pair_pixels >= max(2, width // 8))
    matched_windows = informative_windows = 0
    local_dynamic = max(float(np.percentile(view, 99) - np.percentile(view, 10)), 1.0)
    for start in range(0, a.shape[0] - 31, 64):
        left = a[start:start + 64]
        right = b[start:start + 64]
        if left.shape[0] < 32:
            continue
        if min(float(left.std()), float(right.std())) < 0.008 * local_dynamic:
            continue
        informative_windows += 1
        if ncc(left, right) >= 0.6:
            matched_windows += 1
    total_windows = max(1, math.ceil(a.shape[0] / 64))
    jump = float(np.percentile(np.abs(view[:, cut] - view[:, cut - 1]), 90))
    control_jumps = [
        float(np.percentile(np.abs(view[:, other] - view[:, other - 1]), 90))
        for other in (cut - 128, cut - 80, cut + 80, cut + 128)
        if 1 <= other < view.shape[1]
    ]
    jump_ratio = jump / max(float(np.median(control_jumps)), 0.01 * local_dynamic) if control_jumps else float("nan")
    total_foreground = int(np.sum(image > threshold))
    local_foreground = int(np.sum(view[:, cut - width:cut + width] > threshold))
    duplicate_foreground = int(np.sum(both_bright))
    return {
        "score_status": "ok",
        "duplicate_ncc": round(peak["ncc"], 5),
        "control_ncc": round(control, 5) if math.isfinite(control) else "",
        "duplicate_ncc_excess": round(peak["ncc"] - control, 5) if math.isfinite(control) else "",
        "repeat_width_px": width,
        "along_seam_offset_px": offset,
        "matched_window_fraction": round(matched_windows / total_windows, 5),
        "matched_window_count": matched_windows,
        "informative_window_count": informative_windows,
        "matched_bright_runs": run_count,
        "matched_bright_pixel_fraction": round(float(both_bright.sum()) / max(int(either_bright.sum()), 1), 5),
        "duplicate_foreground_fraction": round(duplicate_foreground / max(total_foreground, 1), 5),
        "foreground_burden_fraction": round(local_foreground / max(total_foreground, 1), 5),
        "jump_ratio": round(jump_ratio, 5) if math.isfinite(jump_ratio) else "",
    }


def seam_locations(row: dict, size: int = FIELD_SIZE) -> list[tuple[str, int, int]]:
    """Projected old source-field boundaries in this registered patch."""
    ref_y, ref_x = int(row["reference_y"]), int(row["reference_x"])
    patch_size = int(row["patch_size"])
    shift_y, shift_x = float(row["registration_shift_y"]), float(row["registration_shift_x"])
    locations = []
    for index in range(1, 6):
        for orientation, global_cut, origin in (
            ("vertical", index * size + shift_x, ref_x),
            ("horizontal", index * size + shift_y, ref_y),
        ):
            local_cut = int(round(global_cut - origin))
            if 0 <= local_cut < patch_size:
                locations.append((orientation, index, local_cut))
    return locations


def patch_path(run: Path, row: dict) -> Path:
    # The manifest's absolute path was written on Nemo and may not exist on macOS.
    path = run / "quantitative_patches" / row["acquisition_id"] / Path(row["quantitative_path"]).name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def chosen_acquisitions(args, available_ids: set[str]) -> set[str]:
    if args.all_acquisitions:
        return available_ids
    if args.acquisition_id:
        return set(args.acquisition_id)
    measurements = range(1, 13) if args.measurements == "all" else [int(v) for v in args.measurements.split(",")]
    if any(not 1 <= value <= 12 for value in measurements):
        raise ValueError("Measurements must be in 1..12.")
    return {f"measurement_{value:02d}_dapi" for value in measurements}


def wells(run: Path) -> list[Path]:
    return sorted(
        path for path in (run / "manifests/wells").iterdir()
        if path.is_dir() and (path / "patches.csv").is_file()
    )


def summarize(rows: list[dict], patch_ids: set[str], well_name: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["well_slug"], row["patch_id"]), []).append(row)
    summary = []
    for patch in sorted(patch_ids):
        well = well_name
        items = grouped.get((well, patch), [])
        ok = [item for item in items if item["score_status"] == "ok"]
        result = {
            "well_slug": well,
            "patch_id": patch,
            "seam_records": len(items),
            "evaluable_seams": len(ok),
            "measurements_with_evaluable_seam": len({item["measurement"] for item in ok}),
        }
        for metric in SCORE_METRICS:
            values = [float(item[metric]) for item in ok if item.get(metric) not in ("", None)]
            result[f"max_{metric}"] = round(max(values), 5) if values else ""
        summary.append(result)
    return summary


def summarize_images(scored: list[dict], inputs: list[dict], well_name: str) -> list[dict]:
    """One record for every input patch image, including images without a seam."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in scored:
        grouped.setdefault((row["patch_id"], row["acquisition_id"]), []).append(row)
    summary = []
    for image in inputs:
        items = grouped.get((image["patch_id"], image["acquisition_id"]), [])
        ok = [item for item in items if item["score_status"] == "ok"]
        result = {
            "well_slug": well_name, "patch_id": image["patch_id"],
            "acquisition_id": image["acquisition_id"],
            "measurement": image["measurement"], "marker": image["marker"],
            "image_score_status": "ok" if ok else ("no_evaluable_seam" if items else "no_source_seam"),
            "seam_records": len(items), "evaluable_seams": len(ok),
        }
        for metric in SCORE_METRICS:
            values = [float(item[metric]) for item in ok if item.get(metric) not in ("", None)]
            result[f"max_{metric}"] = round(max(values), 5) if values else ""
        summary.append(result)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, default=HERE / "results/cache_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/scores_v1")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--well", help="For example sc_rep5-6__r04_c02")
    target.add_argument("--task-index", type=int, help="Zero-based well index for a Slurm array")
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--measurements", default="all", help="Comma list, or all (DAPI only)")
    parser.add_argument("--acquisition-id", action="append", help="Override DAPI selection; repeat for multiple markers")
    parser.add_argument("--all-acquisitions", action="store_true", help="Score every available DAPI and marker acquisition")
    parser.add_argument("--max-patches", type=int, help="Pilot only: score this many patch IDs per well")
    parser.add_argument("--scoring-plan", type=Path, help="Saved balanced sample and acquisition selection")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.scoring_plan and args.max_patches is not None:
        parser.error("Use --scoring-plan or --max-patches, not both")
    cache = json.loads(args.cache_manifest.read_text())
    run = host_path(cache["patch_run"])
    all_wells = wells(run)
    if args.count_tasks:
        print(len(all_wells))
        return
    if args.well:
        well = run / "manifests/wells" / args.well
        if well not in all_wells:
            raise ValueError(f"Unknown well {args.well}")
    elif args.task_index is not None and 0 <= args.task_index < len(all_wells):
        well = all_wells[args.task_index]
    else:
        parser.error("Choose --well or --task-index for scoring.")
    all_rows = read_csv(well / "patches.csv")
    plan = json.loads(host_path(args.scoring_plan).read_text()) if args.scoring_plan else None
    selection = None
    if plan:
        if plan["cache_manifest_sha256"] != sha256(args.cache_manifest):
            raise ValueError("Scoring plan/cache provenance mismatch")
        if well.name not in plan["wells"]:
            raise ValueError(f"Well {well.name} is not selected in the scoring plan")
        selection = plan["wells"][well.name]
    ids = set(plan["acquisition_ids"]) if plan else chosen_acquisitions(args, {row["acquisition_id"] for row in all_rows})
    rows = [row for row in all_rows if row["acquisition_id"] in ids]
    unknown = ids - {row["acquisition_id"] for row in rows}
    if unknown:
        raise ValueError(f"No patches for these acquisitions: {sorted(unknown)}")
    if selection:
        if selection["image_keys"] is not None:
            keep_images = {tuple(key) for key in selection["image_keys"]}
            rows = [row for row in rows if (row["patch_id"], row["acquisition_id"]) in keep_images]
        else:
            keep = set(selection["patch_ids"])
            rows = [row for row in rows if row["patch_id"] in keep]
        if len(rows) != selection["image_count"]:
            raise ValueError(f"Expected {selection['image_count']} selected images in {well.name}, found {len(rows)}")
    if args.max_patches is not None:
        keep = set(sorted({row["patch_id"] for row in rows})[:args.max_patches])
        rows = [row for row in rows if row["patch_id"] in keep]
    output = args.output_dir / "wells" / well.name
    config = {
        "score_version": SCORE_VERSION,
        "scorer_sha256": sha256(Path(__file__)),
        "cache_manifest_sha256": sha256(args.cache_manifest),
        "acquisition_ids": sorted(ids),
        "max_patches": args.max_patches,
        "selection_sha256": selection["selection_sha256"] if selection else None,
        "well_slug": well.name,
        "input_patches_csv_sha256": sha256(well / "patches.csv"),
    }
    meta_path = output / "score_metadata.json"
    if not args.force and meta_path.is_file() and all(
        (output / name).is_file() for name in ("seam_scores.csv", "image_scores.csv", "patch_scores.csv")
    ):
        previous = json.loads(meta_path.read_text())
        if all(previous.get(key) == value for key, value in config.items()):
            print(f"{well.name}: reused matching score outputs", flush=True)
            return
    scored = []
    patch_keys = set()
    print(f"{well.name}: scoring {len(rows)} TIFF images from {len({row['patch_id'] for row in rows})} "
          f"spatial patches", flush=True)
    for row in progress(rows, desc=well.name, unit="image"):
        image_path = patch_path(run, row)
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file, dtype=np.float32)
        if image.shape != (int(row["patch_size"]), int(row["patch_size"])):
            raise ValueError(f"Unexpected image size: {image_path}")
        patch_keys.add(row["patch_id"])
        for orientation, seam_index, local_cut in seam_locations(row):
            result = score_join(image, orientation, local_cut)
            scored.append({
                "well_slug": well.name,
                "patch_id": row["patch_id"],
                "acquisition_id": row["acquisition_id"],
                "measurement": row["measurement"],
                "marker": row["marker"],
                "orientation": orientation,
                "source_seam_index": seam_index,
                "local_cut_px": local_cut,
                "reference_y": row["reference_y"],
                "reference_x": row["reference_x"],
                "registration_shift_y": row["registration_shift_y"],
                "registration_shift_x": row["registration_shift_x"],
                "quantitative_path": str(image_path),
                **result,
            })
    atomic_csv(output / "seam_scores.csv", scored)
    atomic_csv(output / "image_scores.csv", summarize_images(scored, rows, well.name))
    atomic_csv(output / "patch_scores.csv", summarize(scored, patch_keys, well.name))
    atomic_json(meta_path, {**config, "scored_patch_ids": len(patch_keys),
                           "scored_patch_images": len(rows), "seam_records": len(scored),
                           "scoring_plan": str(args.scoring_plan) if args.scoring_plan else None})
    print(f"{well.name}: scored {len(rows)} patch images ({len(patch_keys)} patch IDs), "
          f"{len(scored)} seams -> {output}", flush=True)


if __name__ == "__main__":
    main()
