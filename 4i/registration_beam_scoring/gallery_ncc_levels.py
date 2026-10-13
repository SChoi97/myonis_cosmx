#!/usr/bin/env python3
"""Make reproducible, paginated galleries across fixed NCC-excess ranges.

Samples are individual evaluable seams, with at most one example from each
patch ID within a bin. Patch cutoff counts use the maximum seam score across
the selected acquisitions, matching the existing illustrative cutoff sweep.
"""

from __future__ import annotations

import argparse
import bisect
import colorsys
import csv
import html
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from score_seams import HERE, host_path, sha256
from visualize_scores import clip_limits, load_font, render_example
from progress import progress
from compare_dapi_rounds import DAPI_IDS, write_round_comparisons


@dataclass
class BinSample:
    key: tuple[str, str]
    row: dict[str, str]
    row_count: int = 1


@dataclass
class BinState:
    label: str
    seam_count: int = 0
    seen_patches: set[tuple[str, str]] = field(default_factory=set)
    samples: list[BinSample] = field(default_factory=list)
    sample_indices: dict[tuple[str, str], int] = field(default_factory=dict)

    def offer(self, row: dict[str, str], count: int, rng: random.Random) -> None:
        self.seam_count += 1
        key = (row["well_slug"], row["patch_id"])
        index = self.sample_indices.get(key)
        if index is not None:
            current = self.samples[index]
            current.row_count += 1
            if rng.randrange(current.row_count) == 0:
                current.row = row
            return
        if key in self.seen_patches:
            return
        self.seen_patches.add(key)
        if len(self.samples) < count:
            self.sample_indices[key] = len(self.samples)
            self.samples.append(BinSample(key, row))
            return
        chosen = rng.randrange(len(self.seen_patches))
        if chosen < count:
            old = self.samples[chosen]
            del self.sample_indices[old.key]
            self.samples[chosen] = BinSample(key, row)
            self.sample_indices[key] = chosen


def csv_files(score_dir: Path, well: str | None, filename: str) -> list[Path]:
    if well:
        files = [score_dir / "wells" / well / filename]
    else:
        files = sorted((score_dir / "wells").glob(f"*/{filename}"))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"No {filename} files found in {score_dir} for well={well!r}")
    return files


def rows_in(paths: list[Path]):
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            yield from csv.DictReader(stream)


def parse_edges(value: str) -> list[float]:
    edges = [float(part.strip()) for part in value.split(",")]
    if not edges or any(not math.isfinite(edge) for edge in edges):
        raise argparse.ArgumentTypeError("Bin edges must be finite numbers")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError("Bin edges must be strictly increasing")
    return edges


def bin_labels(edges: list[float]) -> list[str]:
    return [f"< {edges[0]:g}"] + [
        f"{left:g} to < {right:g}" for left, right in zip(edges, edges[1:])
    ] + [f">= {edges[-1]:g}"]


def write_csv(path: Path, data: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def make_contact_pages(state: BinState, bin_index: int, output: Path,
                       limits: dict, per_page: int) -> list[str]:
    names = []
    ordered = sorted(state.samples, key=lambda item: (
        float(item.row["duplicate_ncc_excess"]), item.key))
    for page_index, start in enumerate(range(0, len(ordered), per_page), 1):
        batch = ordered[start:start + per_page]
        canvas = Image.new("RGB", (1420, 62 + len(batch) * 410), (12, 15, 19))
        draw = ImageDraw.Draw(canvas)
        title = (f"NCC excess {state.label} | {state.seam_count} seams, "
                 f"{len(state.seen_patches)} patch IDs | page {page_index}")
        draw.text((14, 13), title, font=load_font(20), fill=(245, 245, 245))
        for index, sample in enumerate(batch):
            row = sample.row
            label = f"NCC excess {float(row['duplicate_ncc_excess']):.3f}"
            canvas.paste(render_example(row, limits, label), (0, 62 + index * 410))
            canvas.paste(render_example(row, limits, label + " | closeup", zoom=True),
                         (710, 62 + index * 410))
        name = f"bin_{bin_index:02d}_page_{page_index:02d}.png"
        canvas.save(output / name)
        names.append(name)
    return names


def write_index(output: Path, summary: list[dict], cutoffs: list[dict],
                pages: list[list[str]], scope: str, acquisition_plots: list[str], round_comparison: dict) -> None:
    comparisons = []
    if round_comparison.get("status") == "complete":
        comparisons = [
            "<h2>DAPI comparisons across rounds</h2>",
            f"<p>{round_comparison['patches_with_all_12_dapi_images']} of "
            f"{round_comparison['spatial_patch_count']} locations have all 12 DAPI images. "
            "Comparisons match the same well and patch ID; each image's score is its maximum seam NCC excess. "
            "Pink lines mark the boundary between rounds 1–8 and 9–12.</p>",
            "<h3>Score similarity</h3><p>Pearson compares score values; Spearman compares patch rankings. "
            "Each cell reports its paired sample count. Missing or unevaluable scores are excluded; "
            "NA means insufficient pairs or constant scores. "
            "<a href='dapi_round_correlations.csv'>Correlation data</a>; "
            "<a href='dapi_patch_round_scores.csv'>Scores by patch and round</a>.</p>",
            "<a href='dapi_round_score_correlations.png'><img src='dapi_round_score_correlations.png' "
            "alt='DAPI score correlations across rounds'></a>",
            "<h3>If one round is flagged, are other rounds flagged too?</h3>"
            "<p>Rows are the source round already flagged; columns are the round checked. "
            "Percentages show how often the target is also flagged at each illustrative cutoff. "
            "n is the number of flagged source patches with an assessable target image. "
            "An image without a source seam is unflagged; missing images or unevaluable seams are excluded. "
            "<a href='dapi_round_rejection_overlap.csv'>Overlap data</a>.</p>",
            "<a href='dapi_round_rejection_overlap.png'><img src='dapi_round_rejection_overlap.png' "
            "alt='Conditional DAPI rejection overlap across rounds'></a>",
            "<h3>One flagged DAPI round discards the patch in all rounds</h3>"
            "<p>The shared rule uses the maximum DAPI score across all 12 rounds at a location. "
            "When it reaches the cutoff, discard that well + patch ID in every round and marker channel. "
            "The curve compares this shared rule with filtering each DAPI image independently. "
            "For complete round coverage, every discarded patch removes 12 DAPI images. "
            "<a href='dapi_shared_filter_retention.csv'>Patch and image counts at each cutoff</a>; "
            "<a href='dapi_shared_patch_filter_sweep.csv'>Shared patch flags at illustrative cutoffs</a>.</p>",
            "<a href='dapi_shared_filter_retention.png'><img src='dapi_shared_filter_retention.png' "
            "alt='Retention when one flagged DAPI round rejects all rounds of a patch'></a>",
        ]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>NCC excess seam gallery</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;"
        "padding:0 1rem;background:#11151a;color:#eee}a{color:#71e3df}"
        "table{border-collapse:collapse;margin:1rem 0}th,td{padding:.4rem .8rem;"
        "border:1px solid #56606b;text-align:right}th:first-child,td:first-child{text-align:left}"
        "img{max-width:100%;height:auto;display:block;margin:1rem 0 2rem}</style></head><body>",
        "<h1>NCC excess seam examples</h1>",
        f"<p>{html.escape(scope)}. Examples are sampled from individual seam rows; "
        "each bin uses distinct patch IDs. Each row shows the full patch and a "
        "closeup of its former source-field seam. Cyan marks that seam.</p>",
        "<p>Cutoff counts below are <strong>patch IDs</strong>: one patch is above "
        "a cutoff when any selected acquisition or seam in that patch reaches it. "
        "Patches without an evaluable seam are shown separately.</p>",
        "<h2>Retention curves</h2><p>With a cutoff, patches below it and patches "
        "without an evaluable seam are retained. The latter have no old seam "
        "score to reject on. <a href='retention_curve.csv'>Download curve data</a>. "
        "<a href='retention_by_acquisition.csv'>Download per-acquisition data</a>.</p>",
        "<p>The individual-image curve counts each scored TIFF separately. The spatial-patch curve "
        "uses the worst score across the acquisitions scored at that location. "
        "<a href='image_retention_curve.csv'>Download individual-image curve data</a>.</p>",
        "<a href='image_retention_curve.png'><img src='image_retention_curve.png' "
        "alt='Fraction of individual TIFF patch images retained at each cutoff'></a>",
        "<a href='patch_retention_curve.png'><img src='patch_retention_curve.png' "
        "alt='Fraction of patch images retained at each NCC excess cutoff'></a>",
        *[f"<a href='{name}'><img src='{name}' loading='lazy' "
          "alt='Fraction of patch images retained by acquisition'></a>" for name in acquisition_plots],
        *comparisons,
        "<h2>Score bins</h2><table><tr><th>NCC excess</th><th>Seams</th>"
        "<th>Distinct patch IDs</th><th>Examples shown</th></tr>",
    ]
    for row in summary:
        parts.append(f"<tr><td>{html.escape(row['bin_label'])}</td>"
                     f"<td>{row['seam_count']}</td><td>{row['distinct_patch_ids']}</td>"
                     f"<td>{row['examples_rendered']}</td></tr>")
    parts.append("</table><h2>Patch counts at candidate cutoffs</h2><table><tr>"
                 "<th>Reject if max NCC excess &ge;</th><th>Below cutoff</th>"
                 "<th>At or above</th><th>No evaluable seam</th>"
                 "<th>All scored patch IDs</th></tr>")
    for row in cutoffs:
        parts.append(f"<tr><td>{row['cutoff']}</td><td>{row['below_cutoff']}</td>"
                     f"<td>{row['at_or_above']}</td><td>{row['no_evaluable_seam']}</td>"
                     f"<td>{row['all_scored_patch_ids']}</td></tr>")
    parts.append("</table>")
    for row, image_names in zip(summary, pages):
        parts.append(f"<h2>NCC excess {html.escape(row['bin_label'])}</h2>")
        if not image_names:
            parts.append("<p>No evaluable examples in this bin.</p>")
        for name in image_names:
            parts.append(f"<a href='{name}'><img src='{name}' loading='lazy' "
                         f"alt='Seam examples for NCC excess {html.escape(row['bin_label'])}'></a>")
    parts.append("</body></html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def curve_cutoffs(values: list[float], edges: list[float]) -> list[float]:
    lower = min(-0.1, min(values, default=0.0) - 0.05)
    upper = max(1.05, max(values, default=1.0) + 0.05)
    return sorted(set(edges + [lower + (upper - lower) * i / 200 for i in range(201)]))


def retained_count(sorted_values: list[float], total: int, cutoff: float) -> tuple[int, int]:
    below = bisect.bisect_left(sorted_values, cutoff)
    return below + total - len(sorted_values), below


def make_retention_data(cutoffs: list[float], all_patches: set[tuple[str, str]],
                        patch_max: dict[tuple[str, str], float],
                        by_acquisition: dict[str, dict[tuple[str, str], float]],
                        populations: dict[str, set[tuple[str, str]]] | None = None) -> tuple[list[dict], list[dict]]:
    total = len(all_patches)
    sorted_values = sorted(patch_max.values())
    overall = []
    for cutoff in cutoffs:
        kept, below = retained_count(sorted_values, total, cutoff)
        overall.append({
            "cutoff": round(cutoff, 5), "all_scored_patch_ids": total,
            "evaluable_patch_ids": len(sorted_values),
            "no_evaluable_seam": total - len(sorted_values),
            "evaluable_below_cutoff": below,
            "kept_including_no_evaluable_seam": kept,
            "rejected_at_or_above": total - kept,
            "fraction_kept_all_patches": round(kept / total, 6),
            "fraction_kept_evaluable_patches": round(below / len(sorted_values), 6) if sorted_values else "",
        })
    by_acq = []
    for acquisition, scores in sorted(by_acquisition.items()):
        acq_total = len(populations.get(acquisition, set())) if populations is not None else total
        if not acq_total:
            continue
        values = sorted(scores.values())
        for cutoff in cutoffs:
            kept, below = retained_count(values, acq_total, cutoff)
            by_acq.append({
                "acquisition_id": acquisition, "cutoff": round(cutoff, 5),
                "all_scored_patch_ids": acq_total, "evaluable_patch_ids": len(values),
                "no_evaluable_seam": acq_total - len(values),
                "kept_including_no_evaluable_seam": kept,
                "fraction_kept_all_patches": round(kept / acq_total, 6),
                "fraction_kept_evaluable_patches": round(below / len(values), 6) if values else "",
            })
    return overall, by_acq


def image_retention_data(cutoffs: list[float], total: int, values: list[float]) -> list[dict]:
    values = sorted(values)
    result = []
    for cutoff in cutoffs:
        kept, below = retained_count(values, total, cutoff)
        result.append({"cutoff": round(cutoff, 5), "all_scored_images": total,
                       "evaluable_images": len(values), "no_evaluable_seam": total - len(values),
                       "kept_including_no_evaluable_seam": kept,
                       "rejected_at_or_above": total - kept,
                       "fraction_kept_all_images": round(kept / total, 6),
                       "fraction_kept_evaluable_images": round(below / len(values), 6) if values else ""})
    return result


def plot_retention(output: Path, series: list[tuple[str, list[dict], tuple[int, int, int]]],
                   edges: list[float], title: str) -> None:
    width, height = 1280, 760
    image = Image.new("RGB", (width, height), (17, 21, 27))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 100, 100, 1170, 580
    draw.text((24, 20), title, font=load_font(23), fill=(245, 245, 245))
    draw.text((24, 52), "Patch retained when max NCC excess < cutoff; no evaluable seam retained",
              font=load_font(16), fill=(190, 200, 210))
    minimum = min(float(row["cutoff"]) for _, rows, _ in series for row in rows)
    maximum = max(float(row["cutoff"]) for _, rows, _ in series for row in rows)
    span = max(maximum - minimum, 1e-9)
    for fraction in (0, .2, .4, .6, .8, 1.0):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=(55, 65, 76), width=1)
        draw.text((48, y - 9), f"{fraction:.0%}", font=load_font(14), fill=(200, 205, 215))
    for edge in edges:
        x = left + (edge - minimum) / span * (right - left)
        draw.line((x, top, x, bottom), fill=(75, 80, 90), width=1)
        draw.text((x - 16, bottom + 10), f"{edge:g}", font=load_font(13), fill=(205, 210, 220))
    draw.line((left, top, left, bottom, right, bottom), fill=(200, 205, 215), width=2)
    for label, rows, color in series:
        points = [
            (left + (float(row["cutoff"]) - minimum) / span * (right - left),
             bottom - float(row.get("fraction_kept_all_patches", row.get("fraction_kept_all_images"))) * (bottom - top))
            for row in rows
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    draw.text((left + 390, bottom + 40), "NCC excess cutoff", font=load_font(16), fill=(225, 225, 230))
    for index, (label, _, color) in enumerate(series):
        column, row = index % 3, index // 3
        x, y = 100 + column * 375, 645 + row * 25
        draw.line((x, y + 7, x + 27, y + 7), fill=color, width=4)
        draw.text((x + 35, y), label, font=load_font(14), fill=(230, 230, 235))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, default=HERE / "results/cache_manifest.json")
    parser.add_argument("--score-dir", type=Path, default=HERE / "results/scores_v1")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/ncc_level_gallery")
    parser.add_argument("--well", help="Optional: sample only one scored well")
    parser.add_argument("--acquisition-id", action="append", help="Limit to acquisition(s); repeat as needed")
    parser.add_argument("--include-markers", action="store_true", help="Opt into legacy marker galleries; default is DAPI only")
    parser.add_argument("--examples-per-bin", type=int, default=5)
    parser.add_argument("--bin-edges", type=parse_edges, default=parse_edges("0,0.2,0.4,0.6,0.8,1.0"),
                        help="Comma-separated thresholds; outer bins are open-ended")
    parser.add_argument("--per-page", type=int, default=10, help="Maximum examples per contact sheet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--expect-all-images", action="store_true",
                        help="Require complete metrics for every image recorded in the patch cache")
    parser.add_argument("--scoring-plan", type=Path, help="Validate the scored sample against its saved plan")
    parser.add_argument("--round-comparison-cutoffs", type=parse_edges, default=parse_edges("0.2,0.4,0.6,0.8"))
    args = parser.parse_args()
    if args.examples_per_bin < 1 or args.per_page < 1:
        parser.error("--examples-per-bin and --per-page must be positive")
    if args.well and ("/" in args.well or args.well in (".", "..")):
        parser.error("--well must be a well slug, not a path")
    if args.expect_all_images and args.well:
        parser.error("--expect-all-images requires the complete dataset, without --well")
    print(f"Loading display settings and score tables from {args.score_dir}", flush=True)
    cache = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    plan = json.loads(host_path(args.scoring_plan).read_text()) if args.scoring_plan else None
    if plan and plan["cache_manifest_sha256"] != sha256(args.cache_manifest):
        raise ValueError("Scoring plan/cache provenance mismatch")
    limits = clip_limits(host_path(cache["patch_run"]))
    selected_ids = set(args.acquisition_id) if args.acquisition_id else None
    if not args.include_markers:
        selected_ids = selected_ids or set(DAPI_IDS)
        if selected_ids - set(DAPI_IDS):
            parser.error("This gallery defaults to DAPI only; marker galleries require --include-markers")
    seam_files = csv_files(args.score_dir, args.well, "seam_scores.csv")
    patch_files = csv_files(args.score_dir, args.well, "patch_scores.csv")
    if plan:
        expected_wells = {args.well} if args.well else set(plan["wells"])
        if {path.parent.name for path in seam_files} != expected_wells:
            raise ValueError("Scored wells do not match the saved scoring plan")
    elif not args.well and len(seam_files) != int(cache["completed_wells"]):
        raise ValueError(f"Expected scores for {cache['completed_wells']} wells, found {len(seam_files)}. "
                         "Run the full scoring array first; use --well for a one-well gallery.")
    if len(patch_files) != len(seam_files):
        raise ValueError("The score directory has an incomplete set of patch_scores.csv files")
    metadata_acquisitions = set()
    scored_image_count = 0
    if not args.well or plan:
        expected_acquisitions = None
        cache_hash = sha256(args.cache_manifest)
        for path in progress(seam_files, desc="Verifying scored wells", unit="well"):
            metadata_path = path.with_name("score_metadata.json")
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("max_patches") is not None:
                raise ValueError(f"Pilot-only patch subset detected: {metadata_path}")
            if metadata.get("cache_manifest_sha256") != cache_hash:
                raise ValueError(f"Score/cache provenance mismatch: {metadata_path}")
            acquisitions = tuple(metadata.get("acquisition_ids", []))
            scored_image_count += int(metadata.get("scored_patch_images", 0))
            metadata_acquisitions.update(acquisitions)
            if expected_acquisitions is None:
                expected_acquisitions = acquisitions
            elif acquisitions != expected_acquisitions:
                raise ValueError(f"Mixed acquisition scopes in score directory: {metadata_path}")
            if plan:
                selection = plan["wells"][path.parent.name]
                if metadata.get("selection_sha256") != selection["selection_sha256"]:
                    raise ValueError(f"Score/sample selection mismatch: {metadata_path}")
                if metadata.get("scored_patch_images") != selection["image_count"]:
                    raise ValueError(f"Incomplete image scoring: {metadata_path}")
                if metadata.get("scored_patch_ids") != selection["patch_count"]:
                    raise ValueError(f"Incomplete spatial patch scoring: {metadata_path}")
            if args.expect_all_images:
                if len(acquisitions) != int(cache["acquisition_count"]):
                    raise ValueError(f"Not all acquisitions were scored: {metadata_path}")
                if not path.with_name("image_scores.csv").is_file():
                    raise FileNotFoundError(path.with_name("image_scores.csv"))
        if args.expect_all_images and scored_image_count != int(cache["quantitative_patch_file_count_from_manifests"]):
            raise ValueError(f"Expected {cache['quantitative_patch_file_count_from_manifests']} scored images, "
                             f"found {scored_image_count}")
        if plan:
            expected_images = sum(plan["wells"][well]["image_count"] for well in expected_wells)
            if scored_image_count != expected_images:
                raise ValueError(f"Expected {expected_images} planned images, found {scored_image_count}")
    else:
        metadata_path = seam_files[0].with_name("score_metadata.json")
        if metadata_path.is_file():
            metadata_acquisitions.update(json.loads(metadata_path.read_text(encoding="utf-8")).get("acquisition_ids", []))
    bins = [BinState(label) for label in bin_labels(args.bin_edges)]
    rng = random.Random(args.seed)
    patch_max: dict[tuple[str, str], float] = {}
    by_acquisition: dict[str, dict[tuple[str, str], float]] = {
        acquisition: {} for acquisition in (selected_ids or metadata_acquisitions)
    }
    found_ids: set[str] = set(metadata_acquisitions)
    for row in rows_in(seam_files):
        if selected_ids is not None and row["acquisition_id"] not in selected_ids:
            continue
        found_ids.add(row["acquisition_id"])
        by_acquisition.setdefault(row["acquisition_id"], {})
        if row["score_status"] != "ok":
            continue
        try:
            value = float(row["duplicate_ncc_excess"])
        except (ValueError, KeyError):
            continue
        if not math.isfinite(value):
            continue
        key = (row["well_slug"], row["patch_id"])
        patch_max[key] = max(patch_max.get(key, -math.inf), value)
        acq_scores = by_acquisition.setdefault(row["acquisition_id"], {})
        acq_scores[key] = max(acq_scores.get(key, -math.inf), value)
        bins[bisect.bisect_right(args.bin_edges, value)].offer(row, args.examples_per_bin, rng)
    if selected_ids and selected_ids - found_ids:
        parser.error(f"Acquisition(s) absent from seam scores: {sorted(selected_ids - found_ids)}")
    if not patch_max:
        raise ValueError("No evaluable NCC excess scores in the selected seam rows")
    all_patches = {(row["well_slug"], row["patch_id"]) for row in rows_in(patch_files)}
    population_by_acq = None
    image_files = [path.with_name("image_scores.csv") for path in seam_files]
    dapi_image_rows = []
    if all(path.is_file() for path in image_files):
        population_by_acq = {}
        selected_patch_ids = set()
        for row in progress(rows_in(image_files), total=scored_image_count or None,
                            desc="Loading image summaries", unit="image"):
            if selected_ids and row["acquisition_id"] not in selected_ids:
                continue
            key = (row["well_slug"], row["patch_id"])
            population_by_acq.setdefault(row["acquisition_id"], set()).add(key)
            selected_patch_ids.add(key)
            if row["acquisition_id"] in DAPI_IDS:
                dapi_image_rows.append(row)
        all_patches = selected_patch_ids
    elif plan:
        raise ValueError("The planned run requires image_scores.csv for every selected well")
    if not all_patches:
        raise ValueError("No scored patch IDs found")
    if patch_max.keys() - all_patches:
        raise ValueError("A seam score references a patch absent from patch_scores.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Comparing DAPI scores and shared filtering across rounds", flush=True)
    round_comparison = write_round_comparisons(dapi_image_rows, args.output_dir, args.round_comparison_cutoffs)
    summary = []
    samples = []
    pages = []
    for index, state in enumerate(progress(bins, desc="Rendering score bins", unit="bin")):
        selected = sorted(state.samples, key=lambda item: (
            float(item.row["duplicate_ncc_excess"]), item.key))
        summary.append({
            "bin_index": index,
            "bin_label": state.label,
            "seam_count": state.seam_count,
            "distinct_patch_ids": len(state.seen_patches),
            "examples_rendered": len(selected),
        })
        for rank, sample in enumerate(selected, 1):
            row = sample.row
            samples.append({"bin_index": index, "bin_label": state.label,
                            "example_rank": rank, **row})
        pages.append(make_contact_pages(state, index, args.output_dir, limits, args.per_page))
    cutoffs = []
    for cutoff in args.bin_edges:
        above = sum(value >= cutoff for value in patch_max.values())
        below = len(patch_max) - above
        cutoffs.append({"cutoff": cutoff, "all_scored_patch_ids": len(all_patches),
                        "evaluable_patch_ids": len(patch_max), "below_cutoff": below,
                        "at_or_above": above,
                        "no_evaluable_seam": len(all_patches) - len(patch_max)})
    write_csv(args.output_dir / "bin_summary.csv", summary, list(summary[0]))
    write_csv(args.output_dir / "cutoff_patch_counts.csv", cutoffs, list(cutoffs[0]))
    plot_cutoffs = curve_cutoffs(list(patch_max.values()), args.bin_edges)
    retention, retention_by_acq = make_retention_data(plot_cutoffs, all_patches, patch_max, by_acquisition, population_by_acq)
    image_population = sum(len(population) for population in population_by_acq.values()) if population_by_acq else len(all_patches) * len(by_acquisition)
    image_retention = image_retention_data(plot_cutoffs, image_population,
                                         [value for scores in by_acquisition.values() for value in scores.values()])
    write_csv(args.output_dir / "image_retention_curve.csv", image_retention, list(image_retention[0]))
    plot_retention(args.output_dir / "image_retention_curve.png",
                   [("Individual TIFF images", image_retention, (255, 180, 75))], args.bin_edges,
                   f"Individual TIFF patch images retained | n={image_population}")
    write_csv(args.output_dir / "retention_curve.csv", retention, list(retention[0]))
    write_csv(args.output_dir / "retention_by_acquisition.csv", retention_by_acq,
              list(retention_by_acq[0]) if retention_by_acq else ["acquisition_id", "cutoff"])
    plot_retention(args.output_dir / "patch_retention_curve.png",
                   [("All selected acquisitions", retention, (0, 230, 220))],
                   args.bin_edges, f"Spatial patches retained across scored acquisitions | n={len(all_patches)}")
    acquisition_plots = []
    acquisitions = sorted({row["acquisition_id"] for row in retention_by_acq})
    for start in range(0, len(acquisitions), 12):
        group = acquisitions[start:start + 12]
        palette = []
        for index, acquisition in enumerate(group):
            color = tuple(round(255 * channel) for channel in
                          colorsys.hsv_to_rgb(index / len(group), .7, .95))
            palette.append((acquisition, [row for row in retention_by_acq
                                         if row["acquisition_id"] == acquisition], color))
        page = start // 12 + 1
        name = "retention_by_acquisition.png" if page == 1 else f"retention_by_acquisition_{page:02d}.png"
        plot_retention(args.output_dir / name, palette, args.bin_edges,
                       f"Patch retention by acquisition | page {page}")
        acquisition_plots.append(name)
    if samples:
        write_csv(args.output_dir / "sampled_seams.csv", samples, list(samples[0]))
    scope = (f"Well {args.well}" if args.well else f"All {len(seam_files)} scored wells")
    if plan and plan["sampling"]["enabled"]:
        scope = (f"Balanced random sample: {plan['sampling']['requested_count']} {plan['sampling']['unit']}s, "
                 f"seed {plan['sampling']['seed']}; {len(seam_files)} wells; {image_population} TIFF images. "
                 "Retention estimates describe the scored sample")
    scope += "; acquisitions: " + (", ".join(sorted(selected_ids)) if selected_ids else "all in score directory")
    write_index(args.output_dir, summary, cutoffs, pages, scope, acquisition_plots, round_comparison)
    (args.output_dir / "gallery_metadata.json").write_text(json.dumps({
        "score_dir": str(args.score_dir), "well": args.well,
        "acquisition_ids": sorted(selected_ids) if selected_ids else sorted(found_ids),
        "examples_per_bin": args.examples_per_bin, "bin_edges": args.bin_edges,
        "per_page": args.per_page, "seed": args.seed,
        "sampling_applies_to": "scoring_and_examples" if plan and plan["sampling"]["enabled"] else "rendered_examples_only",
        "full_image_coverage_verified": args.expect_all_images,
        "scored_patch_images": image_population,
        "scoring_plan": str(args.scoring_plan) if args.scoring_plan else None,
        "planned_coverage_verified": bool(plan),
        "scored_wells": len(seam_files), "all_scored_patch_ids": len(all_patches),
        "evaluable_patch_ids": len(patch_max),
        "dapi_only": not args.include_markers,
        "dapi_round_comparison": round_comparison,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Gallery: {args.output_dir / 'index.html'}")
    if args.expect_all_images:
        print(f"Verified complete metrics for {scored_image_count} cached patch images")
    print(f"{len(all_patches)} patch IDs; {len(patch_max)} with evaluable NCC excess")
    for row in summary:
        print(f"{row['bin_label']}: {row['seam_count']} seams, "
              f"{row['distinct_patch_ids']} patch IDs, {row['examples_rendered']} examples")


if __name__ == "__main__":
    main()
