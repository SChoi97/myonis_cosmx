#!/usr/bin/env python3
"""Render ranked seam examples with every score printed beside the image."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from score_seams import HERE, host_path, read_csv

METRICS = (
    "duplicate_ncc_excess",
    "matched_window_fraction",
    "matched_bright_runs",
    "jump_ratio",
    "duplicate_foreground_fraction",
)
METRIC_TITLES = {
    "duplicate_ncc_excess": "NCC excess",
    "matched_window_fraction": "Matched-window fraction",
    "matched_bright_runs": "Bright fragment runs",
    "jump_ratio": "Seam brightness jump",
    "duplicate_foreground_fraction": "Dup. bright burden",
}


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def clip_limits(run: Path) -> dict[str, tuple[float, float]]:
    """Use the recorded PNG display limits without requiring Excel libraries."""
    manifests = sorted((run / "manifests/clipping").glob("*.json"))
    if manifests:
        limits = {}
        for path in manifests:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("status") != "complete":
                raise ValueError(f"Incomplete clipping manifest: {path}")
            key = item["acquisition"]["acquisition_id"]
            values = item["clip_limits"]
            minimum, maximum = float(values["min_16bit"]), float(values["max_16bit"])
            if not (math.isfinite(minimum) and math.isfinite(maximum) and maximum > minimum):
                raise ValueError(f"Invalid display limits in {path}")
            if key in limits:
                raise ValueError(f"Duplicate clipping acquisition: {key}")
            limits[key] = (minimum, maximum)
        return limits
    # Compatibility with older runs that only retained the workbook.
    from openpyxl import load_workbook

    book = load_workbook(run / "provenance/clip_limits.xlsx", read_only=True, data_only=True)
    try:
        iterator = book["clip_limits"].values
        header = next(iterator)
        rows = [dict(zip(header, values)) for values in iterator if values[0]]
        return {row["acquisition_id"]: (float(row["min_16bit"]), float(row["max_16bit"])) for row in rows}
    finally:
        book.close()


def format_number(value: str | float | int | None, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "NA"
    return f"{number:.{decimals}f}" if math.isfinite(number) else "NA"


def render_example(row: dict, limits: dict[str, tuple[float, float]], label: str, *, zoom: bool = False) -> Image.Image:
    path = host_path(row["quantitative_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        raw = np.asarray(source, dtype=np.float32)
    minimum, maximum = limits[row["acquisition_id"]]
    visual = np.uint8(np.rint(255 * np.clip((raw - minimum) / (maximum - minimum), 0, 1)))
    local_cut = int(row["local_cut_px"])
    if zoom:
        view = visual.T if row["orientation"] == "horizontal" else visual
        cross_start = max(0, min(view.shape[1] - 192, local_cut - 96))
        cross_stop = cross_start + 192
        band = view[:, max(0, local_cut - 20):min(view.shape[1], local_cut + 20)]
        signal = np.mean(band, axis=1)
        window = np.convolve(signal, np.ones(96, dtype=np.float32), mode="valid")
        peak = int(np.argmax(window)) + 48
        along_start = max(0, min(view.shape[0] - 192, peak - 96))
        crop = view[along_start:along_start + 192, cross_start:cross_stop]
        if row["orientation"] == "horizontal":
            crop = crop.T
        visual = crop
        cut = (local_cut - cross_start) * 2
    else:
        cut = local_cut * 384 / raw.shape[0]
    image = Image.fromarray(visual, mode="L").convert("RGB").resize((384, 384))
    draw = ImageDraw.Draw(image)
    if row["orientation"] == "vertical":
        draw.line((cut, 0, cut, 383), fill=(0, 255, 255), width=2)
    else:
        draw.line((0, cut, 383, cut), fill=(0, 255, 255), width=2)
    canvas = Image.new("RGB", (710, 410), (19, 22, 27))
    canvas.paste(image, (8, 17))
    annotation = ImageDraw.Draw(canvas)
    title_font = load_font(16)
    body_font = load_font(14)
    annotation.text((405, 15), label, font=title_font, fill=(255, 220, 100))
    lines = (
        f"{row['well_slug']}  {row['patch_id']}",
        f"{row['acquisition_id']}",
        f"{row['orientation']} seam {row['source_seam_index']}",
        f"position {row['local_cut_px']} px",
        "",
        f"Repeat NCC     {format_number(row.get('duplicate_ncc'))}",
        f"Control NCC    {format_number(row.get('control_ncc'))}",
        f"NCC excess     {format_number(row.get('duplicate_ncc_excess'))}",
        f"Repeat width   {row.get('repeat_width_px', 'NA')} px",
        f"Along offset   {row.get('along_seam_offset_px', 'NA')} px",
        f"Matched windows {format_number(row.get('matched_window_fraction'))}",
        f"Bright runs    {row.get('matched_bright_runs', 'NA')}",
        f"Bright match   {format_number(row.get('matched_bright_pixel_fraction'))}",
        f"Dup. bright burden {format_number(row.get('duplicate_foreground_fraction'))}",
        f"Foreground burden {format_number(row.get('foreground_burden_fraction'))}",
        f"Jump ratio     {format_number(row.get('jump_ratio'))}",
    )
    for index, text in enumerate(lines):
        annotation.text((405, 46 + index * 22), text, font=body_font, fill=(230, 230, 230))
    return canvas


def score_rows(score_dir: Path, well: str | None) -> list[dict]:
    if well:
        files = [score_dir / "wells" / well / "seam_scores.csv"]
    else:
        files = sorted((score_dir / "wells").glob("*/seam_scores.csv"))
    rows = []
    for path in files:
        if path.is_file():
            rows.extend(read_csv(path))
    if not rows:
        raise ValueError(f"No seam score rows in {score_dir} for well={well!r}.")
    return [row for row in rows if row.get("score_status") == "ok"]


def patch_score_rows(score_dir: Path, well: str | None) -> list[dict]:
    if well:
        files = [score_dir / "wells" / well / "patch_scores.csv"]
    else:
        files = sorted((score_dir / "wells").glob("*/patch_scores.csv"))
    return [row for path in files if path.is_file() for row in read_csv(path)]


def ranked_examples(rows: list[dict], metric: str) -> list[tuple[str, dict]]:
    usable = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            usable.append((value, row))
    usable.sort(key=lambda item: item[0])
    if not usable:
        return []
    return [(name, usable[index][1]) for name, index in (
        ("low", 0), ("median", len(usable) // 2), ("high", len(usable) - 1)
    )]


def make_gallery(rows: list[dict], limits: dict, output: Path, *, zoom: bool = False) -> None:
    title_height = 52
    gallery = Image.new("RGB", (710 * 3, title_height + 410 * len(METRICS)), (12, 15, 19))
    draw = ImageDraw.Draw(gallery)
    title = "Old-field seam closeups" if zoom else "Old-field seam scoring"
    draw.text((14, 11), f"{title}: low, median, high examples for each metric", font=load_font(20), fill=(245, 245, 245))
    output.parent.mkdir(parents=True, exist_ok=True)
    individual = output.parent / "examples"
    individual.mkdir(exist_ok=True)
    for metric_index, metric in enumerate(METRICS):
        selected = ranked_examples(rows, metric)
        for column, (rank, row) in enumerate(selected):
            label = f"{METRIC_TITLES[metric]}: {rank}"
            panel = render_example(row, limits, label, zoom=zoom)
            gallery.paste(panel, (column * 710, title_height + metric_index * 410))
            panel.save(individual / f"{metric}__{rank}{'__zoom' if zoom else ''}.png")
    gallery.save(output)


def make_round_comparison(rows: list[dict], limits: dict, output: Path) -> None:
    """Show one informative patch across available DAPI rounds, full and zoomed."""
    by_patch: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["marker"] == "DAPI":
            by_patch.setdefault((row["well_slug"], row["patch_id"]), []).append(row)
    if not by_patch:
        return
    chosen_key = max(
        by_patch,
        key=lambda key: (
            len({row["measurement"] for row in by_patch[key]}),
            max(float(row.get("duplicate_foreground_fraction") or 0) for row in by_patch[key]),
        ),
    )
    per_round: dict[int, dict] = {}
    for row in by_patch[chosen_key]:
        measurement = int(row["measurement"])
        if measurement not in per_round or float(row.get("duplicate_foreground_fraction") or 0) > float(per_round[measurement].get("duplicate_foreground_fraction") or 0):
            per_round[measurement] = row
    preferred = [measurement for measurement in (1, 6, 9, 12) if measurement in per_round]
    if not preferred:
        preferred = sorted(per_round)[:4]
    canvas = Image.new("RGB", (1420, 48 + 410 * len(preferred)), (12, 15, 19))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f"Same patch across rounds: {chosen_key[0]} {chosen_key[1]}", font=load_font(20), fill=(245, 245, 245))
    for index, measurement in enumerate(preferred):
        row = per_round[measurement]
        canvas.paste(render_example(row, limits, f"Measurement {measurement}: full patch"), (0, 48 + 410 * index))
        canvas.paste(render_example(row, limits, f"Measurement {measurement}: seam closeup", zoom=True), (710, 48 + 410 * index))
    canvas.save(output)


def write_cutoff_sweep(rows: list[dict], patches: list[dict], output: Path) -> list[dict]:
    by_patch: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_patch.setdefault((row["well_slug"], row["patch_id"]), []).append(row)
    thresholds = {
        "duplicate_ncc_excess": (0.2, 0.4, 0.6, 0.8, 1.0),
        "matched_window_fraction": (0.125, 0.375, 0.5, 0.75, 0.875, 1.0),
        "matched_bright_runs": (1, 3, 5, 8, 10),
        "jump_ratio": (2.0, 4.0, 6.0, 10.0, 20.0, 40.0),
        "duplicate_foreground_fraction": (0.01, 0.02, 0.03, 0.05, 0.08),
    }
    data = []
    for metric, cutoffs in thresholds.items():
        per_patch = []
        for patch in patches:
            values = []
            for row in by_patch.get((patch["well_slug"], patch["patch_id"]), []):
                try:
                    value = float(row[metric])
                except (KeyError, ValueError, TypeError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            per_patch.append(max(values) if values else float("nan"))
        for cutoff in cutoffs:
            affected = sum(math.isfinite(value) and value >= cutoff for value in per_patch)
            data.append({
                "metric": metric,
                "illustrative_cutoff": cutoff,
                "total_patch_count": len(per_patch),
                "patches_with_score": sum(math.isfinite(value) for value in per_patch),
                "patches_at_or_above": affected,
                "fraction_at_or_above": round(affected / len(per_patch), 5) if per_patch else "",
            })
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return data


def make_cutoff_plot(data: list[dict], output: Path) -> None:
    """Show how many patch IDs exceed each exploratory cutoff."""
    width, height = 1040, 320 * 3 + 52
    canvas = Image.new("RGB", (width, height), (18, 21, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Exploratory cutoff sweep: fraction of patch IDs at or above each score", font=load_font(20), fill=(245, 245, 245))
    for index, metric in enumerate(("duplicate_ncc_excess", "matched_window_fraction", "matched_bright_runs", "jump_ratio", "duplicate_foreground_fraction")):
        group = [row for row in data if row["metric"] == metric]
        if not group:
            continue
        panel_x, panel_y = 25 + (index % 2) * 510, 60 + (index // 2) * 320
        draw.text((panel_x, panel_y), METRIC_TITLES[metric], font=load_font(17), fill=(255, 220, 100))
        left, top, right, bottom = panel_x + 55, panel_y + 38, panel_x + 465, panel_y + 265
        draw.line((left, top, left, bottom, right, bottom), fill=(180, 185, 195), width=2)
        for fraction in (0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - fraction * (bottom - top)
            draw.line((left, y, right, y), fill=(60, 66, 76), width=1)
            draw.text((panel_x + 2, y - 7), f"{fraction:.0%}", font=load_font(12), fill=(190, 195, 205))
        min_cut = float(group[0]["illustrative_cutoff"])
        max_cut = float(group[-1]["illustrative_cutoff"])
        span = max(max_cut - min_cut, 1e-9)
        points = []
        for row in group:
            x = left + (float(row["illustrative_cutoff"]) - min_cut) / span * (right - left)
            fraction = float(row["fraction_at_or_above"] or 0)
            y = bottom - fraction * (bottom - top)
            points.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 235, 220))
            draw.text((x - 12, bottom + 8), str(row["illustrative_cutoff"]), font=load_font(11), fill=(205, 210, 215))
        if len(points) > 1:
            draw.line(points, fill=(0, 235, 220), width=3)
        draw.text((left, bottom + 31), f"n={group[0]['total_patch_count']} patches; cutoffs are not calibrated", font=load_font(12), fill=(180, 185, 195))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, default=HERE / "results/cache_manifest.json")
    parser.add_argument("--score-dir", type=Path, default=HERE / "results/scores_v1")
    parser.add_argument("--well", help="Optional: visualize one scored well")
    parser.add_argument("--acquisition-id", action="append", help="Limit plots and cutoff sweep to one or more acquisitions")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/visualizations_v1")
    args = parser.parse_args()
    import json
    cache = json.loads(args.cache_manifest.read_text())
    run = host_path(cache["patch_run"])
    limits = clip_limits(run)
    rows = score_rows(args.score_dir, args.well)
    if args.acquisition_id:
        selected = set(args.acquisition_id)
        rows = [row for row in rows if row["acquisition_id"] in selected]
        if not rows:
            raise ValueError(f"No evaluable seams for {sorted(selected)}")
    patches = patch_score_rows(args.score_dir, args.well)
    suffix = (args.well or "all_wells") + ("__" + "_".join(args.acquisition_id) if args.acquisition_id else "")
    output = args.output_dir / suffix
    output.mkdir(parents=True, exist_ok=True)
    make_gallery(rows, limits, output / "metric_gallery.png")
    make_gallery(rows, limits, output / "metric_gallery_zoom.png", zoom=True)
    make_round_comparison(rows, limits, output / "same_patch_rounds.png")
    cutoff_data = write_cutoff_sweep(rows, patches, output / "illustrative_cutoff_sweep.csv")
    make_cutoff_plot(cutoff_data, output / "illustrative_cutoff_sweep.png")
    print(f"Rendered {len(rows)} evaluable seams to {output}")


if __name__ == "__main__":
    main()
