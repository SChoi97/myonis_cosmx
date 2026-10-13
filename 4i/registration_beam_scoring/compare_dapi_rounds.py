#!/usr/bin/env python3
"""Compare DAPI seam scores at matching spatial patches across all 12 rounds.

Reads saved image_scores.csv tables only; no image loading or rescoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_scores import load_font
from progress import progress

ROUNDS = tuple(range(1, 13))
DAPI_IDS = {f"measurement_{r:02d}_dapi": r for r in ROUNDS}
METRIC = "max_duplicate_ncc_excess"


def finite_score(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return ranks


def correlation(x: list[float], y: list[float], min_pairs: int, *, rank: bool = False):
    if len(x) < min_pairs:
        return None
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if rank:
        a, b = average_ranks(a), average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def assessable(record: dict | None) -> bool:
    return record is not None and (record["score"] is not None or record["status"] == "no_source_seam")


def comparison_data(rows, cutoffs: list[float], min_pairs: int = 3) -> dict:
    patches = {}
    for row in rows:
        round_number = DAPI_IDS.get(row["acquisition_id"])
        if round_number is None:
            continue
        if int(row["measurement"]) != round_number:
            raise ValueError(f"DAPI measurement/acquisition mismatch: {row['acquisition_id']}")
        key = (row["well_slug"], row["patch_id"])
        entry = patches.setdefault(key, {})
        if round_number in entry:
            raise ValueError(f"Duplicate DAPI image summary: {key}, round {round_number}")
        status = row["image_score_status"]
        if status not in ("ok", "no_source_seam", "no_evaluable_seam"):
            raise ValueError(f"Unknown DAPI image status: {status}")
        score = finite_score(row.get(METRIC)) if status == "ok" else None
        entry[round_number] = {"score": score, "status": status}
    correlations, overlaps = [], []
    for source in ROUNDS:
        for target in ROUNDS:
            paired = [(entry[source], entry[target]) for entry in patches.values()
                      if source in entry and target in entry]
            finite = [(a["score"], b["score"]) for a, b in paired
                      if a["score"] is not None and b["score"] is not None]
            x, y = [a for a, _ in finite], [b for _, b in finite]
            correlations.append({
                "source_round": source, "target_round": target,
                "matched_images": len(paired), "paired_evaluable_patches": len(finite),
                "pearson_r": correlation(x, y, min_pairs),
                "spearman_rho": correlation(x, y, min_pairs, rank=True),
            })
            eligible = [(a, b) for a, b in paired if assessable(a) and assessable(b)]
            for cutoff in cutoffs:
                flags = [(a["score"] is not None and a["score"] >= cutoff,
                          b["score"] is not None and b["score"] >= cutoff) for a, b in eligible]
                source_flagged = sum(a for a, _ in flags)
                target_flagged = sum(b for _, b in flags)
                both = sum(a and b for a, b in flags)
                either = sum(a or b for a, b in flags)
                overlaps.append({
                    "cutoff": cutoff, "source_round": source, "target_round": target,
                    "paired_assessable_patches": len(eligible),
                    "matched_but_unassessable_patches": len(paired) - len(eligible),
                    "source_flagged_in_pair": source_flagged, "target_flagged_in_pair": target_flagged,
                    "both_flagged": both,
                    "probability_target_flagged_given_source": both / source_flagged if source_flagged else None,
                    "jaccard_flagged_patches": both / either if either else None,
                })
    wide, shared_masks = [], []
    for (well, patch), entry in sorted(patches.items()):
        values = [record["score"] for record in entry.values() if record["score"] is not None]
        worst = max(values) if values else None
        wide_row = {"well_slug": well, "patch_id": patch,
                    "available_dapi_rounds": len(entry), "evaluable_dapi_rounds": len(values),
                    "all_12_dapi_images_present": len(entry) == 12, "max_dapi_ncc_excess": worst}
        for r in ROUNDS:
            wide_row[f"round_{r:02d}_ncc_excess"] = entry.get(r, {}).get("score")
            wide_row[f"round_{r:02d}_status"] = entry.get(r, {}).get("status", "missing_image")
        wide.append(wide_row)
        for cutoff in cutoffs:
            triggered = [str(r) for r in ROUNDS if entry.get(r, {}).get("score") is not None
                         and entry[r]["score"] >= cutoff]
            shared_masks.append({
                "well_slug": well, "patch_id": patch, "cutoff": cutoff,
                "max_dapi_ncc_excess": worst, "reject_all_rounds": bool(triggered),
                "triggering_dapi_rounds": ",".join(triggered),
                "decision_status": "reject" if triggered else ("below_cutoff" if values else "no_evaluable_dapi_score"),
                "available_dapi_rounds": len(entry), "evaluable_dapi_rounds": len(values),
                "all_12_dapi_images_present": len(entry) == 12,
            })
    return {"patches": patches, "correlations": correlations, "overlaps": overlaps,
            "wide": wide, "shared_masks": shared_masks}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heat_color(value, *, signed: bool):
    if value is None:
        return (63, 64, 70)
    middle = (24, 32, 43)
    endpoint = (60, 131, 230) if signed and value < 0 else ((242, 151, 50) if signed else (0, 220, 195))
    strength = min(1.0, abs(value))
    return tuple(round(a + (b - a) * strength) for a, b in zip(middle, endpoint))


def draw_matrix(draw, records, field, count_field, origin, cell, title, *, signed):
    left, top = origin
    number_font, count_font = load_font(13), load_font(10)
    draw.text((left - 30, top - 65), title, font=load_font(19), fill=(240, 243, 247))
    lookup = {(row["source_round"], row["target_round"]): row for row in records}
    for r in ROUNDS:
        draw.text((left + (r - 1) * cell + cell / 2 - 10, top - 28), f"D{r}", font=number_font, fill=(220, 224, 232))
        draw.text((left - 39, top + (r - 1) * cell + cell / 2 - 8), f"D{r}", font=number_font, fill=(220, 224, 232))
        for c in ROUNDS:
            row = lookup[r, c]
            value = row[field]
            x, y = left + (c - 1) * cell, top + (r - 1) * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=heat_color(value, signed=signed))
            text = "NA" if value is None else (f"{value:.2f}" if signed else f"{value:.0%}")
            color = (12, 18, 24) if value is not None and value > .65 else (245, 247, 250)
            box = draw.textbbox((0, 0), text, font=number_font)
            draw.text((x + (cell - box[2]) / 2, y + 7), text, font=number_font, fill=color)
            count = f"n={row[count_field]}"
            box = draw.textbbox((0, 0), count, font=count_font)
            draw.text((x + (cell - box[2]) / 2, y + cell - 17), count, font=count_font, fill=color)
    end = left + 12 * cell, top + 12 * cell
    draw.line((left + 8 * cell, top, left + 8 * cell, end[1]), fill=(255, 100, 160), width=3)
    draw.line((left, top + 8 * cell, end[0], top + 8 * cell), fill=(255, 100, 160), width=3)


def plot_correlations(path: Path, rows: list[dict], min_pairs: int):
    image = Image.new("RGB", (1540, 890), (17, 21, 27))
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "DAPI seam score similarity across rounds", font=load_font(25), fill=(245, 247, 250))
    draw.text((25, 57), "Match well + patch ID; score = maximum NCC excess across that image's seams", font=load_font(17), fill=(205, 212, 222))
    draw_matrix(draw, rows, "pearson_r", "paired_evaluable_patches", (85, 180), 52, "Pearson: similarity of score values", signed=True)
    draw_matrix(draw, rows, "spearman_rho", "paired_evaluable_patches", (855, 180), 52, "Spearman: similarity of patch rankings", signed=True)
    draw.text((25, 825), f"n = patches with finite scores in both rounds. NA = fewer than {min_pairs} pairs or constant scores.", font=load_font(16), fill=(215, 221, 231))
    draw.text((25, 853), "Blue: negative correlation; dark: near zero; orange: positive. Pink lines separate rounds 1-8 and 9-12.", font=load_font(16), fill=(215, 221, 231))
    image.save(path)


def plot_overlaps(path: Path, rows: list[dict], cutoffs: list[float]):
    columns = min(2, len(cutoffs))
    height = 140 + 655 * math.ceil(len(cutoffs) / columns) + 80
    width = max(1280, 680 * columns)
    image = Image.new("RGB", (width, height), (17, 21, 27))
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "If one DAPI round is flagged, are the others flagged too?", font=load_font(23), fill=(245, 247, 250))
    draw.text((25, 57), "Rows: round already flagged. Columns: round checked. Flag if max NCC excess >= cutoff.", font=load_font(16), fill=(210, 217, 227))
    draw.text((25, 87), "Cells: fraction also flagged; n = flagged source patches with an assessable target image.", font=load_font(16), fill=(210, 217, 227))
    for index, cutoff in enumerate(cutoffs):
        subset = [row for row in rows if row["cutoff"] == cutoff]
        origin = (80 + (width - 680 * columns) // 2 + (index % columns) * 680,
                  200 + (index // columns) * 655)
        draw_matrix(draw, subset, "probability_target_flagged_given_source", "source_flagged_in_pair",
                    origin, 44, f"Illustrative NCC excess cutoff {cutoff:g}", signed=False)
    draw.text((25, height - 66), "No source seam = unflagged. Missing images or unevaluable seams are excluded from this comparison.", font=load_font(15), fill=(215, 221, 231))
    draw.text((25, height - 38), "Grey NA: no flagged source patches to compare. Pink lines separate rounds 1-8 and 9-12.", font=load_font(15), fill=(215, 221, 231))
    image.save(path)


def shared_retention_data(patches: dict, cutoffs: list[float]) -> list[dict]:
    available = sum(len(entry) for entry in patches.values())
    worst = {key: max((record["score"] for record in entry.values() if record["score"] is not None), default=None)
             for key, entry in patches.items()}
    finite = [value for value in worst.values() if value is not None]
    lower = min(-.1, min(finite, default=0) - .05)
    upper = max(1.05, max(finite, default=1) + .05)
    thresholds = sorted(set(cutoffs + [lower + (upper - lower) * i / 200 for i in range(201)]))
    result = []
    for cutoff in thresholds:
        rejected = {key for key, value in worst.items() if value is not None and value >= cutoff}
        discarded_images = sum(len(patches[key]) for key in rejected)
        independent_flags = sum(record["score"] is not None and record["score"] >= cutoff
                                for entry in patches.values() for record in entry.values())
        result.append({
            "cutoff": cutoff, "spatial_patches": len(patches), "available_dapi_images": available,
            "patches_with_all_12_dapi_images": sum(len(entry) == 12 for entry in patches.values()),
            "patches_without_evaluable_dapi_score": len(patches) - len(finite),
            "patches_rejected_in_all_rounds": len(rejected), "patches_kept_in_all_rounds": len(patches) - len(rejected),
            "fraction_spatial_patches_kept": (len(patches) - len(rejected)) / len(patches),
            "dapi_images_discarded_by_shared_filter": discarded_images,
            "dapi_images_kept_by_shared_filter": available - discarded_images,
            "fraction_dapi_images_kept_by_shared_filter": (available - discarded_images) / available,
            "dapi_images_flagged_independently": independent_flags,
            "fraction_dapi_images_kept_independently": (available - independent_flags) / available,
            "additional_dapi_images_discarded_for_consistency": discarded_images - independent_flags,
        })
    return result


def plot_shared_retention(path: Path, rows: list[dict]):
    image = Image.new("RGB", (1280, 760), (17, 21, 27))
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), "Shared DAPI filter: one flagged round discards the patch in every round", font=load_font(23), fill=(245, 247, 250))
    total, available = rows[0]["spatial_patches"], rows[0]["available_dapi_images"]
    complete = rows[0]["patches_with_all_12_dapi_images"]
    draw.text((24, 56), f"{total} patch locations; {available} DAPI images; {complete} locations with all 12 rounds", font=load_font(17), fill=(205, 215, 227))
    left, top, right, bottom = 100, 110, 1170, 575
    lower, upper = rows[0]["cutoff"], rows[-1]["cutoff"]
    for fraction in (0, .2, .4, .6, .8, 1):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=(55, 65, 76))
        draw.text((45, y - 8), f"{fraction:.0%}", font=load_font(14), fill=(210, 217, 227))
    for cutoff in (0, .2, .4, .6, .8, 1):
        x = left + (cutoff - lower) / (upper - lower) * (right - left)
        if left <= x <= right:
            draw.line((x, top, x, bottom), fill=(55, 65, 76))
            draw.text((x - 12, bottom + 12), f"{cutoff:g}", font=load_font(14), fill=(210, 217, 227))
    series = (("Each round filtered independently", "fraction_dapi_images_kept_independently", (255, 180, 75)),
              ("All rounds discarded if any DAPI round is flagged", "fraction_dapi_images_kept_by_shared_filter", (0, 225, 205)))
    for index, (label, field, color) in enumerate(series):
        points = [(left + (row["cutoff"] - lower) / (upper - lower) * (right - left),
                   bottom - row[field] * (bottom - top)) for row in rows]
        draw.line(points, fill=color, width=3)
        y = 640 + index * 30
        draw.line((100, y + 8, 132, y + 8), fill=color, width=4)
        draw.text((145, y), label, font=load_font(16), fill=(230, 235, 242))
    draw.text((485, 608), "NCC excess cutoff", font=load_font(17), fill=(230, 235, 242))
    draw.text((24, 716), "No evaluable DAPI score is reported separately and retained by this exploratory rule.", font=load_font(15), fill=(200, 210, 222))
    image.save(path)


def write_round_comparisons(rows, output: Path, cutoffs: list[float], min_pairs: int = 3) -> dict:
    if min_pairs < 3 or not cutoffs or any(not math.isfinite(c) for c in cutoffs):
        raise ValueError("Require finite cutoffs and at least three pairs for correlations")
    data = comparison_data(rows, cutoffs, min_pairs)
    if not data["patches"]:
        return {"status": "no_dapi_images", "figures": []}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dapi_patch_round_scores.csv", data["wide"])
    write_csv(output / "dapi_round_correlations.csv", data["correlations"])
    write_csv(output / "dapi_round_rejection_overlap.csv", data["overlaps"])
    write_csv(output / "dapi_shared_patch_filter_sweep.csv", data["shared_masks"])
    retention = shared_retention_data(data["patches"], cutoffs)
    write_csv(output / "dapi_shared_filter_retention.csv", retention)
    figures = ["dapi_round_score_correlations.png", "dapi_round_rejection_overlap.png", "dapi_shared_filter_retention.png"]
    plot_correlations(output / figures[0], data["correlations"], min_pairs)
    plot_overlaps(output / figures[1], data["overlaps"], cutoffs)
    plot_shared_retention(output / figures[2], retention)
    metadata = {
        "status": "complete", "metric": METRIC, "rounds": list(ROUNDS),
        "spatial_patch_count": len(data["patches"]),
        "patches_with_all_12_dapi_images": sum(len(entry) == 12 for entry in data["patches"].values()),
        "correlation_min_pairs": min_pairs, "illustrative_cutoffs": cutoffs, "figures": figures,
        "correlation_policy": "pairwise finite scores only; constant or insufficient pairs = NA",
        "overlap_policy": "pairwise assessable images; no source seam = unflagged; missing/unevaluable = omitted",
        "shared_filter_rule": "reject matching spatial patch in all markers/rounds if any DAPI round reaches cutoff",
    }
    (output / "dapi_round_comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--min-pairs", type=int, default=3)
    args = parser.parse_args()
    paths = sorted((args.score_dir / "wells").glob("*/image_scores.csv"))
    if not paths:
        parser.error("No saved image_scores.csv tables found")
    def rows():
        for path in progress(paths, desc="Loading DAPI image summaries", unit="well"):
            with path.open(newline="", encoding="utf-8") as handle:
                yield from csv.DictReader(handle)
    metadata = write_round_comparisons(rows(), args.output_dir, [float(c) for c in args.cutoffs.split(",")], args.min_pairs)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
