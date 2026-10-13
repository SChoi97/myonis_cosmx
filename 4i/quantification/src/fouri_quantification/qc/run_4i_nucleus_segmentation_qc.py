#!/usr/bin/env python3
"""Multi-round nucleus persistence QC; run in the existing 4i Python environment.

From the directory containing this script:
    python run_4i_nucleus_segmentation_qc.py
    python run_4i_nucleus_segmentation_qc.py --registration_metric ssim --registration-ssim-threshold 0.35 --workers 12
    python run_4i_nucleus_segmentation_qc.py --workers 12
    python run_4i_nucleus_segmentation_qc.py --registration_metric ssim --fast_mode --workers 12
    python run_4i_nucleus_segmentation_qc.py --datapath /path/to/preprocessing/run --dry-run
    python run_4i_nucleus_segmentation_qc.py --use_binary_mask_reference --n_images 100 --seed 42 --workers 12
    python run_4i_nucleus_segmentation_qc.py --use_binary_mask_reference --min-other-rounds 7 --iou-threshold 0.8
    python run_4i_nucleus_segmentation_qc.py --reference-round 1 --iou-threshold 0.5
    python run_4i_nucleus_segmentation_qc.py --patch-mode one-per-field
    python run_4i_nucleus_segmentation_qc.py --max-plots 5 --hide-rejected
    python run_4i_nucleus_segmentation_qc.py --dry-run
    python run_4i_nucleus_segmentation_qc.py --summary-only --output-dir /path/to/existing/results

Paths are resolved relative to the surrounding computational_tools directory,
so the same script works on Nemo (/nemo/lab/tedescos/...) and the Mac mount.
Use --tools-dir or --datapath to override them; --run-dir remains an alias
for --datapath. Supply the run root containing png_8bit/ and segmentation/,
not png_8bit/measurement_01_dapi. The png_8bit folder itself is also accepted.
Rounds default to 1-8; --rounds 1 2 3 4 5 6 7 8 9 10 11 12 selects all twelve.
Registration mode accepts --filter-rounds to score a subset (including the reference).
All selected --rounds still register/display; excluded rounds have blank scores and
are labelled EXCLUDED FROM DAPI FILTERING in their panels.
All selected rounds are discovered below the run root. Requires Python >=3.10 and the MYONIS-VisTools dependencies in the active
environment (see its pyproject.toml).

Edges are ALWAYS excluded: is_edge is computed from the original unsmoothed YOLO
contour/mask touching or crossing any image boundary (no added margin). The YOLO
inputs do not contain an is_edge column. Flags preserve original contour order.
Registration/binary modes use reference edges; Hungarian mode excludes an ID if
any assigned contour touches an edge. Grey means N/A or edge, orange means high
change/coverage failure in reference-based filtering. Kept reference objects are blue.
Registration assignments and nucleus tables include is_edge and exclusion_reason;
edge_count is a subset of not_evaluable_count (not added to it).

--registration_metric (alias --registration-metric) selects the MAIN filter:
  pixel_max / pixel_mean: original relative pixel difference, with fixed max/mean pooling;
  relative: original relative difference, choose aggregation separately;
  relative_calibrated: same formula with broader field-wide gain bounds;
  ssim / zncc_signal / gradient: the structural comparison methods described below.
An explicit metric enables registration mode and defaults to max across rounds,
except pixel_mean; --registration-aggregation mean changes the other methods.
With no metric option, the legacy relative + mean default is preserved.
The selected method controls plots/, assignments/, statistics and all diagnostics.
SSIM uses --registration-ssim-threshold; ZNCC/gradient use their named cutoffs;
pixel/relative methods use --registration-change-threshold. All default to 0.35.

Default: all complete patches, all eight rounds, Round 2 as reference, strict
IoU > 0.35. A retained reference ID must have a passing one-to-one Hungarian
match in EVERY round. IDs are 1-based reference contour indices, local to each
patch; they are not lineage IDs or IDs shared between different patches.

Persistent nuclei keep their ID colors across panels. Orange means the ID
passes here (or belongs to the reference) but fails in another round. Magenta
means no passing assignment here: low IoU, no counterpart, or one-to-one
competition. Completely missing contours cannot be drawn. Error colors are
reserved from the identity palette. --hide-rejected shows only retained nuclei.

Optional --use_binary_mask_reference uses fixed reference contours and ignores
instance boundaries in all other rounds. Its score is REFERENCE COVERAGE:
area(reference nucleus intersect round foreground) / area(reference nucleus).
Foreground is the union of all contours in that round, with no DAPI intensity
threshold. --iou-threshold supplies the strict coverage cutoff in this mode;
it is not an IoU against the full foreground image. Splits/merges that preserve
the foreground union do not affect the score. A reference nucleus is kept if
it passes in at least --min-other-rounds non-reference rounds (default: 2 of 7).
The reference is excluded from this count. Thus a nucleus can be retained
despite disappearing later; set --min-other-rounds 7 to require all rounds.
All panels draw reference contours: blue means kept overall, orange filtered
overall. Panel titles report local coverage passes. Assignment JSON records
per-reference scores, passing-round counts, and retained reference indices;
it does not infer Hungarian matches or current-round instance identities.
Binary-mode round tables use the reference count as denominator, report the
same overall retention in every round, and separately report local coverage
passes and detected instance counts. Default output paths separate the modes.

Optional --registration_based_filtering re-registers each DAPI patch to the
chosen reference using the ACTUAL preprocessing utils.patch_level.register_patch_field
and utils.registration.apply_translation functions. The method and QC settings
match register_patch_level.py: percentile-normalized Fourier cross-correlation
(normalization=None), subpixel shifts, linear interpolation, and translation QC.
It runs on each exported image patch (usually 512x512), not the original 1080x1080
source field. Settings come from RUN_DIR/provenance/pipeline_config.json when
available, otherwise preprocessing/config/pipeline_config.json; override with
--registration-config. Existing manual shifts/illumination correction are not
reapplied. All selected images but ONLY reference contours are required in this mode.

Scoring uses background-corrected DAPI, one robust field-wide brightness gain,
and --registration-smoothing-sigma (default 1 pixel). Per-pixel change is
clip(abs(reference - moving) / max(reference, 0.1*reference_P99.8), 0, 1).
Both MEAN and MAX maps are computed across the selected non-reference comparisons,
on the same common valid area. --registration-aggregation {mean,max} selects
which map is used for filtering; defaults follow --registration_metric above.
A nucleus score is the AREA MEAN of the selected map inside its reference contour.
Thus max mode means mean_over_pixels(max_over_rounds(change)), NOT the maximum
single pixel, and NOT max_over_rounds(mean_over_pixels(change)). Kept means score
<= --registration-change-threshold (default 0.35). Max is more sensitive to
one-round events, including acquisition artifacts; its cutoff needs review.
These are intensity-change scores, not displacement distances or probabilities.
IoU/min-other-rounds are not used.
Edge/empty/zero-signal references or objects below --registration-min-valid-fraction
(default 0.90) are excluded as not_evaluable, separately recorded from high change.
Failed registration QC fails the patch rather than fabricating loss/pass scores.

Registration outputs: plots/ contains blue/orange/grey decisions for every selected round on
re-registered grayscale DAPI. Scalar score maps default to magma (override
--registration-score-cmap). DAPI uses --intensity-percentiles for display only;
score maps use fixed 0-1 limits. registration_overlays/ contains all per-round
change maps plus mean and max maps using the same palette; grey pixels are invalid.
registration_scores/ adds five side-by-side panels: reference nuclei colored by
mean score, mean pixel map, nuclei colored by max-map score, max pixel map,
and a binary MAX decision (blue kept, orange filtered, grey not evaluable).
The binary diagnostic always uses max, even if the active filter uses mean.
All use the SAME fixed 0-1 scale (--registration-score-cmap magma or viridis;
default magma; Reds also supported). Grey means N/A / edge / invalid. All reference nuclei are
shown in this diagnostic, even with --hide-rejected.
registration_maps/*.npz saves mean_change, max_change (float32, invalid pixels
NaN), max_change_round (round attaining each pixel maximum; invalid=0), common_valid,
comparison_rounds, reference_round, score_aggregation, and change_threshold.
assignments/ and nucleus_summary.csv save BOTH scores plus change_score (the
selected score), per-round scores, and worst_round_mean_change_score. The last
metric is max_over_rounds(mean_over_pixels(change)), separately named to avoid
confusing it with the area mean of the max map. Old --summary-only results keep
their original mean scores; unavailable maximum-map scores remain blank.
--registration-alignment-panels adds registration_alignment/ with reference/moving
overlays before and after each saved translation (magenta/green; white overlap).
--skip-dense-maps keeps all figures and scores but omits the .npz archives.
--myotube-contour-dir adds myotube_overlays/ with cyan source contours on every
registered marker acquisition, including rounds excluded from nuclear filtering.
--myotube-acquisition identifies their native image frame (default: round-2 MyHC).
That round's DAPI transform shifts the contours into the reference frame; each
marker image uses its own round's saved transform and its original intensities.
Figure grids automatically expand to include every selected round.

Optional --registration-compare-metrics retains the selected main method and creates
registration_metrics/{relative_calibrated,ssim,zncc_signal,gradient}/, each with
plots/, registration_scores/, registration_overlays/ (per-round score maps),
registration_maps/, assignments/, field_summary.csv and nucleus_summary.csv.
Re-registration is performed ONCE; all methods use exactly the same aligned images.
relative_calibrated uses the existing relative-difference formula with broader
field-gain limits (--registration-comparison-gain-limits, default 0.01 100).
The existing baseline retains its original 0.25 4 gain limits. Raw and clipped
gains are saved for diagnosis; global dimming/loss remains intrinsically ambiguous.
SSIM is 1-clip(SSIM,0,1), with an odd uniform window (default 11), population
covariance and explicit data_range=1 after division by the joint valid maximum.
The shared normalization preserves differences without clipping local structure.
ZNCC+loss uses max(local correlation change, fractional local signal loss).
Gradient+loss similarly combines local gradient-vector cosine change with loss.
Both avoid treating flat reference windows as trustworthy structural matches.
The local texture floor is configurable; component scores are saved separately.
All methods require valid data across every comparison. Local methods exclude
windows touching invalid image borders (including a gradient halo), so more edge
objects may be grey/not evaluable; these are not evidence of detachment.
The independent SSIM/ZNCC/gradient thresholds default to 0.35 ONLY as exploratory
starting cutoffs, not calibrated equivalents of the intensity-change threshold.
Each method saves mean, max-map and worst-round-mean object scores, plus per-round
scores. Pooling/order and reference ID indexing are identical to the baseline.
registration_metrics/field_comparison.csv and metrics_metadata.json document
counts, settings and definitions. --summary-only also rebuilds comparison tables.

--n_images N samples N complete registered image patches without replacement
using --seed (default: 42), keeping all selected rounds together. Omit it to run
all patches. With --patch-mode one-per-field, sampling occurs after selecting
one representative patch per field. The requested N must not exceed the
eligible pool; --n_images and --max-plots cannot be combined. The selected keys
are saved in summary.json, and selection does not depend on worker count.

Outputs: plots/*.png, assignments/*.json, summary.json, field_summary.csv,
round_summary.csv and statistics_metadata.json. The CSVs are written when a
run finishes (including completed patches if interrupted). field_summary.csv
has one row per image patch, with reference_count, kept_count, filtered_count,
kept_proportion and kept_percentage. In default mode, kept means persistent across all rounds;
the denominator is the reference-round count. round_summary.csv uses each
round's own original count and separates nonpersistent IDs from no-pass matches
in default mode. Binary-mode definitions are described above and in metadata.
Zero denominators give blank proportions/percentages, not a fabricated 0%.
Failed/incomplete patches have status rows with blank statistics.

--fast_mode (alias --fast-mode) skips ALL figure rendering and dense .npz pixel
map exports, including optional metric-comparison figures/maps. It keeps the same
registration, scoring and filtering, assignments/*.json, summary.json, QC CSVs,
statistics_metadata.json and myonis_4i/. Assignments retain individual scores,
per-round evidence, edge/exclusion flags, registration QC/shifts and source hashes.
If --registration-compare-metrics is also supplied, comparison assignments and
summary tables are retained. --summary-only works on fast-mode results as usual.
No plot/map directories are created for a fresh fast-mode output directory;
files from earlier runs in an existing output directory are left untouched.
Fast-mode summary entries have plot=null and omit unsaved map/figure paths.

Every new run also writes myonis_4i/{nucleus_qc.csv,round_qc.csv,patch_qc.csv,
manifest.json}. This is metadata for ALL original reference contours, not a
filtered/renumbered segmentation. It includes is_edge, nullable is_persistent,
qc_pass and reasons. is_persistent means passes the SELECTED method/settings;
binary min-other-rounds=2 does not mean presence in all rounds, and registration
scores are not literal presence labels. Edges/unevaluable/failed patches have
unknown persistence, not a fabricated negative. Reference filenames, original
YOLO line numbers and checksums make joins safe across Nemo/Mac mount paths.
Use the SAME reference round and original contour directory in quantification;
--reference-contour-dir can override the reference source without changing the
other rounds. Quantification's example and this standalone CLI default
to Round 2. Compare all selected non-reference rounds, including earlier rounds.
Only the active/main method is exported, not the exploratory metric comparisons.
--summary-only regenerates this exchange without reading source images/contours
for new results. Older assignments lack identity checksums and are listed as
legacy_identity_missing; rerun QC before importing those flags.

--summary-only builds these tables from an existing summary.json and saved
assignments, without loading images or generating plots. It uses the recorded
reference/threshold, not new command-line matching settings. If a run is still
active, the tables are a snapshot of completed patches; rerun --summary-only
after it finishes for complete statistics. This mode needs only Python's
standard library and does not modify summary.json or any existing assignments.

Default-mode assignment files retain
the per-round IDs/statuses and original contour indices, with kept contours
indexed in the SAME ID order across all rounds. Input images/contours are never
changed. Existing outputs with the same settings are regenerated on rerun.
--workers controls how many patches run at once (default: 1). Each worker
uses one native CPU thread, and renders figures without a GUI unless --fast_mode. Use, for
example, --workers 12 within a Nemo allocation providing at least 12 CPUs.
In Hungarian/binary modes, only patches with image AND contour files in all selected rounds are processed;
incomplete patches are reported in the console and summary.json. An existing
empty contour file is valid and causes the expected loss of persistent IDs.
Individual patch failures are logged while other patches continue. Any failed
patch makes the final exit code nonzero; completed outputs remain available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
import csv
from datetime import datetime, timezone
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

if __package__:
    from .myonis_4i_qc_export import export_myonis_4i, source_snapshot, sha256
    from .myotube_overlays import discover_channels, save_myotube_overlay
else:
    from myonis_4i_qc_export import export_myonis_4i, source_snapshot, sha256
    from myotube_overlays import discover_channels, save_myotube_overlay

ROUNDS = tuple(range(1, 9))
RUN_NAME = "4i_patch_level_test_20260825_102029_f6ecb24"
NONPERSISTENT_COLOR = "#FF9F00"
NO_PASS_COLOR = "#FF00FF"
BINARY_KEPT_COLOR = "#1f77b4"
BINARY_FILTERED_COLOR = "#ff7f0e"
NOT_EVALUABLE_COLOR = "#777777"


def default_tools_dir():
    for parent in Path(__file__).resolve().parents:
        if parent.name == "myonis_cosmx":
            return parent.parent
    return Path("/nemo/lab/tedescos/home/users/chois1/computational_tools")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools-dir", type=Path, default=default_tools_dir(),
                        help="Directory containing MYONIS-VisTools and myonis_cosmx (auto-detected).")
    parser.add_argument("--datapath", "--run-dir", dest="run_dir", type=Path,
                        help="Dataset root containing png_8bit/ and segmentation/, or its png_8bit folder; defaults to the notebook's test run.")
    parser.add_argument("--output-dir", type=Path, help="Default: a settings-specific directory under RUN_DIR/qc.")
    parser.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS), metavar="N",
                        help="DAPI rounds to compare (default: 1 2 3 4 5 6 7 8).")
    parser.add_argument("--filter-rounds", type=int, nargs="+", metavar="N",
                        help="Registration mode: rounds used for nuclear filtering, including the reference; all --rounds still register and display.")
    parser.add_argument("--reference-round", type=int, default=2)
    parser.add_argument("--reference-contour-dir", type=Path,
                        help="Optional exact unfiltered reference YOLO directory used by quantification; overrides only the reference round.")
    parser.add_argument("--myotube-contour-dir", type=Path,
                        help="Add myotube_overlays/ showing these YOLO contours on all registered marker images.")
    parser.add_argument("--myotube-acquisition", default="measurement_02_555_myhc",
                        help="Native MyHC image acquisition for the myotube contours (default: measurement_02_555_myhc).")
    parser.add_argument("--iou-threshold", type=float, default=0.35,
                        help="Strict score > threshold in [0, 1): IoU normally, reference coverage in binary-mask mode.")
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--use_binary_mask_reference", "--use-binary-mask-reference", action="store_true",
                        help="Score fixed reference nuclei against each round's combined foreground, without Hungarian matching.")
    filters.add_argument("--registration_based_filtering", "--registration-based-filtering", action="store_true",
                         help="Re-register DAPI with the preprocessing helper, then filter reference nuclei by image change.")
    parser.add_argument("--registration_metric", "--registration-metric",
                        choices=("pixel_max", "pixel_mean", "relative", "relative_calibrated", "ssim", "zncc_signal", "gradient"),
                        help="Select the method used by the main plots/assignments. Enables registration filtering. "
                             "pixel_max/pixel_mean use the original relative pixel change; other explicit methods default to max aggregation.")
    parser.add_argument("--registration-compare-metrics", action="store_true",
                        help="Also compare wider-gain relative difference, SSIM, ZNCC+loss and gradient+loss; save separate visualization folders.")
    parser.add_argument("--registration-alignment-panels", action="store_true",
                        help="Also save reference/moving DAPI overlays before and after re-registration.")
    parser.add_argument("--skip-dense-maps", action="store_true",
                        help="Render the QC figures without exporting dense .npz score maps.")
    parser.add_argument("--registration-comparison-gain-limits", type=float, nargs=2, default=(0.01, 100.0), metavar=("MIN", "MAX"),
                        help="Field-wide gain limits for comparison metrics only (default: 0.01 100); baseline remains 0.25 4.")
    parser.add_argument("--registration-local-window", type=int, default=11,
                        help="Odd local window width for SSIM/correlation/gradient comparisons (default: 11 pixels).")
    parser.add_argument("--registration-texture-floor", type=float, default=0.01,
                        help="Reference texture/gradient RMS floor in normalized units for correlation metrics (default: 0.01).")
    parser.add_argument("--registration-ssim-threshold", type=float, default=0.35,
                        help="Exploratory SSIM-change cutoff; independent of the intensity cutoff (default: 0.35).")
    parser.add_argument("--registration-zncc-threshold", type=float, default=0.35,
                        help="Exploratory ZNCC+loss cutoff (default: 0.35).")
    parser.add_argument("--registration-gradient-threshold", type=float, default=0.35,
                        help="Exploratory gradient+loss cutoff (default: 0.35).")
    parser.add_argument("--registration-aggregation", choices=("mean", "max"),
                        help="Temporal map aggregation before averaging nucleus pixels; default max with an explicit metric, otherwise legacy mean.")
    parser.add_argument("--registration-score-cmap", choices=("magma", "viridis", "Reds"), default="magma",
                        help="Colormap for scalar score maps (default: magma); plots/ DAPI backgrounds remain grayscale.")
    parser.add_argument("--registration-change-threshold", type=float, default=0.35,
                        help="Registration mode: keep if selected mean/max-map nucleus score <= this value (default: 0.35; lower is stricter).")
    parser.add_argument("--registration-smoothing-sigma", type=float, default=1.0,
                        help="Gaussian smoothing before image scoring, in source pixels (default: 1; 0 disables).")
    parser.add_argument("--registration-min-valid-fraction", type=float, default=0.90,
                        help="Minimum reference contour area inside all re-registered image bounds (default: 0.90).")
    parser.add_argument("--registration-config", type=Path,
                        help="Preprocessing pipeline JSON; defaults to run provenance, then preprocessing/config/pipeline_config.json.")
    parser.add_argument("--min-other-rounds", type=int, default=2,
                        help="Binary-mask mode: require coverage above the cutoff in at least N non-reference rounds (default: 2).")
    parser.add_argument("--patch-mode", choices=("all", "one-per-field"), default="all",
                        help="All complete patches (default), or one representative patch per original field.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling and one-per-field selection (default: 42).")
    parser.add_argument("--n_images", "--n-images", type=int,
                        help="Randomly sample N complete image patches without replacement using --seed; omit to process all.")
    parser.add_argument("--max-plots", type=int, help="Limit to the first N selected plots for a small test run.")
    parser.add_argument("--workers", type=int, default=1, help="Number of patches processed concurrently, e.g. 12 (default: 1).")
    parser.add_argument("--fast_mode", "--fast-mode", action="store_true",
                        help="Skip all figures and dense pixel-map files; keep myonis_4i exports, assignments, QC tables and run diagnostics.")
    parser.add_argument("--hide-rejected", action="store_true", help="Show only kept contours.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--render-scale", type=int, default=6, help="Contour supersampling; 1 is faster, 6 matches the notebook.")
    parser.add_argument("--fill-alpha", type=float, default=0.15, help="Fill opacity for kept contours.")
    parser.add_argument("--line-width", type=float, default=1.5, help="Kept outline width in source pixels.")
    parser.add_argument("--intensity-percentiles", type=float, nargs=2, default=(1.0, 99.8), metavar=("LOW", "HIGH"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Validate/discover paths and print counts without loading or plotting.")
    modes.add_argument("--summary-only", action="store_true", help="Write CSV statistics from existing results in --output-dir; no plots or analysis dependencies needed.")
    args = parser.parse_args(argv)
    if len(set(args.rounds)) != len(args.rounds) or len(args.rounds) < 2 or any(not 1 <= r <= 255 for r in args.rounds):
        parser.error("--rounds needs at least two distinct round numbers in [1,255]")
    args.rounds = sorted(args.rounds)
    if args.reference_round not in args.rounds:
        parser.error("--reference-round must be one of --rounds")
    explicit_filter_rounds = args.filter_rounds is not None
    args.filter_rounds = sorted(args.filter_rounds) if explicit_filter_rounds else list(args.rounds)
    if (len(set(args.filter_rounds)) != len(args.filter_rounds) or len(args.filter_rounds) < 2 or
            not set(args.filter_rounds).issubset(args.rounds) or args.reference_round not in args.filter_rounds):
        parser.error("--filter-rounds needs distinct selected rounds, the reference and at least one comparison")
    if args.use_binary_mask_reference and not 1 <= args.min_other_rounds < len(args.rounds):
        parser.error("--min-other-rounds must be between 1 and the number of non-reference rounds")
    explicit_metric = args.registration_metric is not None
    if explicit_metric:
        if args.use_binary_mask_reference:
            parser.error("--registration_metric cannot be combined with --use_binary_mask_reference")
        args.registration_based_filtering = True
    args.registration_metric = args.registration_metric or 'relative'
    if explicit_filter_rounds and not args.registration_based_filtering:
        parser.error("--filter-rounds requires registration filtering")
    if args.myotube_contour_dir is not None and not args.registration_based_filtering:
        parser.error("--myotube-contour-dir requires registration filtering")
    if args.registration_metric in ('pixel_mean', 'pixel_max'):
        required = args.registration_metric.removeprefix('pixel_')
        if args.registration_aggregation not in (None, required):
            parser.error(f"--registration_metric {args.registration_metric} requires --registration-aggregation {required}")
        args.registration_aggregation = required
    else:
        args.registration_aggregation = args.registration_aggregation or ('max' if explicit_metric else 'mean')
    args.registration_active_threshold = {
        'ssim': args.registration_ssim_threshold, 'zncc_signal': args.registration_zncc_threshold,
        'gradient': args.registration_gradient_threshold,
    }.get(args.registration_metric, args.registration_change_threshold)
    if not math.isfinite(args.iou_threshold) or not 0 <= args.iou_threshold < 1:
        parser.error("--iou-threshold must be finite and in [0, 1)")
    if not math.isfinite(args.registration_change_threshold) or not 0 <= args.registration_change_threshold <= 1:
        parser.error("--registration-change-threshold must be finite and in [0, 1]")
    if not math.isfinite(args.registration_smoothing_sigma) or args.registration_smoothing_sigma < 0:
        parser.error("--registration-smoothing-sigma must be finite and nonnegative")
    if not math.isfinite(args.registration_min_valid_fraction) or not 0 < args.registration_min_valid_fraction <= 1:
        parser.error("--registration-min-valid-fraction must be finite and in (0, 1]")
    if args.max_plots is not None and args.max_plots < 1:
        parser.error("--max-plots must be positive")
    if args.n_images is not None and args.n_images < 1:
        parser.error("--n_images must be positive")
    if args.n_images is not None and args.max_plots is not None:
        parser.error("Use either --n_images (random sample) or --max-plots (first N), not both")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.dpi < 1 or args.render_scale < 1:
        parser.error("--dpi and --render-scale must be positive")
    if not math.isfinite(args.fill_alpha) or not 0 <= args.fill_alpha <= 1:
        parser.error("--fill-alpha must be in [0, 1]")
    if not math.isfinite(args.line_width) or args.line_width < 0:
        parser.error("--line-width must be finite and nonnegative")
    low, high = args.intensity_percentiles
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        parser.error("--intensity-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if args.registration_compare_metrics and not args.registration_based_filtering:
        parser.error("--registration-compare-metrics requires --registration_based_filtering")
    if args.registration_local_window < 3 or args.registration_local_window % 2 != 1:
        parser.error("--registration-local-window must be odd and at least 3")
    if not math.isfinite(args.registration_texture_floor) or args.registration_texture_floor <= 0:
        parser.error("--registration-texture-floor must be finite and positive")
    low_gain, high_gain = args.registration_comparison_gain_limits
    if not (math.isfinite(low_gain) and math.isfinite(high_gain) and 0 < low_gain <= 1 <= high_gain):
        parser.error("--registration-comparison-gain-limits must be finite with 0 < MIN <= 1 <= MAX")
    for option in ('ssim', 'zncc', 'gradient'):
        value = getattr(args, f'registration_{option}_threshold')
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f'--registration-{option}-threshold must be in [0,1]')
    args.tools_dir = args.tools_dir.expanduser().resolve()
    args.run_dir = (args.run_dir or args.tools_dir / "myonis_cosmx/4i/preprocessing/results/test_runs" / RUN_NAME).expanduser().resolve()
    if args.run_dir.name == "png_8bit":
        args.run_dir = args.run_dir.parent
    if args.reference_contour_dir is not None:
        args.reference_contour_dir = args.reference_contour_dir.expanduser().resolve()
    if args.myotube_contour_dir is not None:
        args.myotube_contour_dir = args.myotube_contour_dir.expanduser().resolve()
    mode = "persistent_only" if args.hide_rejected else "rejected_shown"
    tag = f"ref_{args.reference_round}_iou_{args.iou_threshold}"
    category = "nucleus_persistence"
    if args.use_binary_mask_reference:
        category = "nucleus_reference_coverage"
        tag = f"ref_{args.reference_round}_coverage_{args.iou_threshold}_min_other_{args.min_other_rounds}"
        mode = "kept_only" if args.hide_rejected else "rejected_shown"
    if args.registration_based_filtering:
        category = "nucleus_registration_change"
        tag = (f"ref_{args.reference_round}_{args.registration_metric}_change_{args.registration_active_threshold}"
               f"_sigma_{args.registration_smoothing_sigma}_valid_{args.registration_min_valid_fraction}")
        if args.registration_aggregation != "mean":
            tag += f"_aggregation_{args.registration_aggregation}"
        tag += "_edges_excluded"
        if args.registration_compare_metrics:
            tag += "_compare_metrics"
        excluded = sorted(set(args.rounds)-set(args.filter_rounds))
        if excluded:
            tag += "_exclude_" + "-".join(map(str, excluded))
        mode = "kept_only" if args.hide_rejected else "rejected_shown"
    args.output_dir = (args.output_dir or args.run_dir / "qc" / category / tag / mode).expanduser().resolve()
    return args


def load_dependencies(tools_dir, verbose=True):
    # --help, --dry-run and --summary-only work without the analysis environment.
    global np, plt, Patch, hsv_to_rgb, rgb_to_hsv, to_rgb, to_hex, mvt
    package_dir = tools_dir / "MYONIS-VisTools"
    if not (package_dir / "myonis_vis/__init__.py").is_file():
        raise FileNotFoundError(f"MYONIS-VisTools not found at {package_dir}; set --tools-dir")
    sys.path.insert(0, str(package_dir))
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")  # No display/server GUI needed on Nemo.
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb, to_hex
        import myonis_vis as mvt
    except ImportError as exc:
        raise RuntimeError("Activate the 4i environment with MYONIS-VisTools dependencies installed. "
                           f"Missing dependency: {exc}") from exc
    if "markers" not in inspect.signature(mvt.render_contours).parameters:
        raise RuntimeError(f"MYONIS-VisTools at {mvt.__file__} is outdated: render_contours needs markers support.")
    if verbose:
        print(f"Using MYONIS-VisTools: {mvt.__file__}", flush=True)


def initialize_worker(tools_dir, verbose=False):
    # Set before importing NumPy/SciPy: 12 workers should not each start 12 BLAS threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    load_dependencies(tools_dir, verbose=verbose)


def registered_patch_key(path):
    ignored = ("measurement-", "source-", "fluor-", "marker-")
    return tuple(token for token in Path(path).stem.split("__") if not token.startswith(ignored))


def source_field_key(key):
    return tuple(token for token in key if token.startswith(("plate-", "rep-", "well-", "field-", "cohort-")))


def field_label(key):
    return " | ".join(token for token in key if token.startswith(("plate-", "rep-", "well-", "field-", "patch_")))


def discover_patches(run_dir, reference_only=None, reference_round=None, reference_contour_dir=None, rounds=None):
    round_numbers = tuple(rounds) if rounds is not None else ROUNDS
    indices, all_keys, round_counts = {}, set(), {}
    for r in round_numbers:
        image_dir = run_dir / f"png_8bit/measurement_{r:02d}_dapi"
        contour_dir = run_dir / (f"segmentation/multiround_nuclei/dapi_r{r}/"
                                f"4i_phenix_myonis_segmentation_nuclei_r{r}/02_nuclei_deduplication/contours")
        if reference_contour_dir is not None and r == reference_round:
            contour_dir = Path(reference_contour_dir)
        require_contours = reference_only is None or r == reference_only
        if require_contours and not contour_dir.is_dir() and not (reference_contour_dir is not None and r == reference_round):
            padded = run_dir / (f"segmentation/multiround_nuclei/dapi_r{r:02d}/"
                                 f"4i_phenix_myonis_segmentation_nuclei_r{r:02d}/02_nuclei_deduplication/contours")
            if padded.is_dir():
                contour_dir = padded
        for directory in ((image_dir, contour_dir) if require_contours else (image_dir,)):
            if not directory.is_dir():
                raise FileNotFoundError(f"Round {r}: missing directory {directory}")
        image_paths = sorted(image_dir.glob("*.png"))
        contour_names = {p.stem for p in contour_dir.glob("*.txt")} if require_contours else set()
        index, seen = {}, set()
        for image_path in image_paths:
            key = registered_patch_key(image_path)
            if not key or not source_field_key(key):
                raise ValueError(f"Cannot identify field from filename: {image_path.name}")
            if key in seen:
                raise ValueError(f"Round {r}: ambiguous registered patch {image_path.name}")
            seen.add(key)
            all_keys.add(key)
            if not require_contours or image_path.stem in contour_names:
                index[key] = (image_path, contour_dir / f"{image_path.stem}.txt" if require_contours else None)
        indices[r] = index
        round_counts[r] = dict(images=len(image_paths), paired=len(index),
                               images_without_contours=len(image_paths) - len(index))
        kind = "image/contour pairs" if require_contours else "images (contours not needed)"
        print(f"Round {r}: {len(index)} {kind}; "
              f"{len(image_paths) - len(index)} missing required contours", flush=True)
    common = sorted(set.intersection(*(set(index) for index in indices.values())))
    incomplete = [dict(patch_key="__".join(key), missing_rounds=[r for r in round_numbers if key not in indices[r]])
                  for key in sorted(all_keys - set(common))]
    if not common:
        raise ValueError(f"No complete patches with all {len(round_numbers)} images and required contour files")
    if incomplete:
        print(f"Skipping {len(incomplete)} incomplete patches; details will be recorded in summary.json.", flush=True)
    return indices, common, dict(round_counts=round_counts, incomplete_patches=incomplete)


def select_patches(common, patch_mode, seed, max_plots, n_images=None):
    grouped = defaultdict(list)
    for key in common:
        grouped[source_field_key(key)].append(key)
    if patch_mode == "all":
        selected = sorted(common)
    else:
        rng = random.Random(seed)
        selected = [rng.choice(sorted(grouped[key])) for key in sorted(grouped)]
    if n_images is not None:
        if not 1 <= n_images <= len(selected):
            raise ValueError(f"--n_images must be between 1 and {len(selected)} eligible patches ({patch_mode})")
        # Select whole registered patches, keeping all chosen rounds together.
        selected = sorted(random.Random(seed).sample(selected, n_images))
    return selected[:max_plots], len(grouped)


def load_field(key, paths_by_round):
    images, contours, paths, contour_paths = {}, {}, {}, {}
    shape = None
    for r in sorted(paths_by_round):
        image_path, contour_path = paths_by_round[r]
        image = mvt.read_image(image_path)
        if image.ndim != 2:
            raise ValueError(f"Expected a grayscale image: {image_path}")
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f"Round {r}: image grid differs for {field_label(key)}: {image.shape} != {shape}")
        images[r] = image
        contours[r] = mvt.read_yolo_contours(contour_path, shape) if contour_path is not None else []
        paths[r], contour_paths[r] = str(image_path), str(contour_path) if contour_path is not None else None
    return dict(key=key, label=field_label(key), images=images, contours=contours,
                paths=paths, contour_paths=contour_paths, shape=shape,
                is_edge={r: contour_edge_flags(contours[r], shape) for r in images})


def contour_edge_flags(contours, shape):
    """True when original contour geometry/mask touches or crosses the patch border.

    No morphology/roundness criterion, smoothing, erosion or extra border margin.
    YOLO files do not carry an is_edge column; retain flags in original contour order.
    """
    height, width = shape
    flags = []
    for contour in contours:
        points = np.concatenate([np.asarray(ring).reshape(-1, 2)
                                 for ring in getattr(contour, 'rings', (contour,))], axis=0)
        if not len(points):
            flags.append(False)
            continue
        outside = np.any((points[:, 0] <= 0) | (points[:, 0] >= width - 1) |
                         (points[:, 1] <= 0) | (points[:, 1] >= height - 1))
        mask = mvt.contours_to_masks([contour], shape)[0]
        flags.append(bool(outside or mask[0].any() or mask[-1].any() or
                          mask[:, 0].any() or mask[:, -1].any()))
    return np.asarray(flags, dtype=bool)


def reference_edge_flags(field, reference_round):
    if reference_round in field.get('is_edge', {}):
        return np.asarray(field['is_edge'][reference_round], dtype=bool)
    return contour_edge_flags(field['contours'][reference_round], field['shape'])


def display_dapi(image, args):
    """Prepare grayscale DAPI for plots/, applying percentile contrast once.

    Preserve these uint8 RGB channels during contour rendering. Score-map
    colormaps are independent of this grayscale image background.
    """
    from myonis_vis.utils.images import prepare_image
    return prepare_image(image, intensity_percentiles=tuple(args.intensity_percentiles))


def persistence_palette(n):
    """Distinct ID colors, reserving the error colors and nearby hues."""
    if n == 0:
        return {}
    error_rgb = np.array([to_rgb(NONPERSISTENT_COLOR), to_rgb(NO_PASS_COLOR)])
    error_hues = rgb_to_hsv(error_rgb)[:, 0]

    def safe(rgb):
        distance = np.linalg.norm(rgb[:, None, :] - error_rgb[None, :, :], axis=2)
        hue_distance = np.abs(rgb_to_hsv(rgb)[:, None, 0] - error_hues[None, :])
        hue_distance = np.minimum(hue_distance, 1 - hue_distance)
        return (distance.min(axis=1) >= 0.50) & (hue_distance.min(axis=1) >= 0.075)

    hues = np.linspace(0, 1, max(256, 4 * n), endpoint=False)
    candidates = hsv_to_rgb(np.array([
        (h, s, v) for h in hues for s in (0.65, 0.85, 1.0) for v in (0.65, 0.82, 1.0)
    ]))
    candidates = np.unique(np.rint(candidates * 255).astype(np.uint8), axis=0) / 255.0
    candidates = candidates[safe(candidates)]
    seeds = np.array([to_rgb(c) for c in ("#e41a1c", "#377eb8", "#4daf4a", "#00cccc")])
    chosen = list(seeds[safe(seeds)])
    if not chosen:
        chosen = [candidates[0]]
    distance = np.min(np.sum((candidates[:, None, :] - np.array(chosen)[None, :, :]) ** 2,
                             axis=2), axis=1)
    while len(chosen) < n:
        index = int(np.argmax(distance))
        if distance[index] < 1e-12:
            raise ValueError("Too many IDs for the reserved-color palette")
        chosen.append(candidates[index])
        distance = np.minimum(distance, np.sum((candidates - chosen[-1]) ** 2, axis=1))
    return {i + 1: to_hex(c) for i, c in enumerate(chosen[:n])}


def filter_persistent_nuclei(field, reference_round, iou_threshold):
    reference = field["contours"][reference_round]
    round_numbers = sorted(field["images"])
    ids_by_round, ious_by_round = {}, {}
    for r in round_numbers:
        if r == reference_round:
            ids_by_round[r] = np.arange(1, len(reference) + 1, dtype=int)
            ious_by_round[r] = np.full(len(reference), np.nan)
        else:
            matched = mvt.match_contours(reference, field["contours"][r], field["shape"],
                                         iou_threshold=np.nextafter(float(iou_threshold), 1.0))
            ids_by_round[r] = matched.prediction_to_reference.copy() + 1
            ious_by_round[r] = matched.prediction_values.copy()
    present = {r: set(ids[ids > 0].tolist()) for r, ids in ids_by_round.items()}
    persistent = set(range(1, len(reference) + 1))
    for r in round_numbers:
        persistent.intersection_update(present[r])
    edge_by_round = {r: reference_edge_flags(field, r) for r in round_numbers}
    edge_ids = set()
    for r in round_numbers:
        edge_ids.update(ids_by_round[r][edge_by_round[r] & (ids_by_round[r] > 0)].tolist())
    persistent.difference_update(edge_ids)
    persistent_ids = np.array(sorted(persistent), dtype=int)
    statuses, kept_indices = {}, {}
    for r, ids in ids_by_round.items():
        keep = np.isin(ids, persistent_ids)
        status = np.full(len(ids), "no_passing_match", dtype=object)
        status[ids > 0] = "not_persistent"
        status[keep] = "persistent"
        status[edge_by_round[r] | np.isin(ids, list(edge_ids))] = "not_evaluable"
        statuses[r] = status
        indices = np.flatnonzero(keep)
        kept_indices[r] = indices[np.argsort(ids[indices])]
        if not np.array_equal(ids[kept_indices[r]], persistent_ids):
            raise RuntimeError(f"Retained IDs do not align in Round {r}")
    return dict(field=field, reference_round=reference_round, iou_threshold=iou_threshold,
                persistent_ids=persistent_ids, colors=persistence_palette(len(reference)), is_edge_by_round=edge_by_round,
                ids_by_round=ids_by_round, ious_by_round=ious_by_round,
                status_by_round=statuses, kept_indices_by_round=kept_indices,
                missing_rounds_by_id={i: [r for r in round_numbers if i not in present[r]]
                                     for i in range(1, len(reference) + 1) if i not in persistent})


def reference_coverage_by_round(field, reference_round):
    """Fraction of each fixed reference region covered by each round's union.

    Use the original, unsmoothed geometry and pixel-centre rasterization used
    by MYONIS-VisTools IoU. Rasterize one contour at a time to avoid allocating
    a full (n_nuclei, height, width) mask stack in each parallel worker.
    """
    reference_pixels = [np.flatnonzero(mvt.contours_to_masks([contour], field["shape"])[0])
                        for contour in field["contours"][reference_round]]
    areas = np.array([len(pixels) for pixels in reference_pixels], dtype=int)
    scores = {}
    for r in sorted(field["images"]):
        if r == reference_round:
            scores[r] = (areas > 0).astype(float)
            continue
        foreground = np.zeros(field["shape"], dtype=bool)
        for contour in field["contours"][r]:
            foreground |= mvt.contours_to_masks([contour], field["shape"])[0]
        flat = foreground.ravel()
        scores[r] = np.array([np.count_nonzero(flat[pixels]) / len(pixels) if len(pixels) else 0.0
                              for pixels in reference_pixels], dtype=float)
    return scores, areas


def filter_binary_mask_reference(field, reference_round, threshold, min_other_rounds):
    """Keep reference objects supported in at least N non-reference rounds."""
    scores, areas = reference_coverage_by_round(field, reference_round)
    passing = {r: values > threshold for r, values in scores.items()}
    passing_other_counts = np.sum([passing[r] for r in sorted(field["images"]) if r != reference_round], axis=0)
    is_edge = reference_edge_flags(field, reference_round)
    keep = (areas > 0) & ~is_edge & (passing_other_counts >= min_other_rounds)
    return dict(field=field, reference_round=reference_round, iou_threshold=threshold,
                min_other_rounds=min_other_rounds, coverage_by_round=scores,
                passing_by_round=passing, reference_areas=areas,
                passing_other_counts=passing_other_counts, keep=keep, is_edge=is_edge,
                kept_reference_indices=np.flatnonzero(keep))


def panel_grid(count, columns=4, width=4.0, height=4.25):
    """Allocate enough axes to show every selected round; preserve the eight-round layout."""
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*width, rows*height),
                             squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    return fig, axes


def save_binary_plot(result, path, args):
    """Draw the fixed reference geometry and global keep/filter status on all rounds."""
    field, reference_round = result["field"], result["reference_round"]
    contours = field["contours"][reference_round]
    kept = result["kept_reference_indices"]
    rejected = np.flatnonzero(~result["keep"])
    layers = {}
    if not args.hide_rejected:
        for label, indices, color in (
                ('filtered', np.flatnonzero(~result['keep'] & ~result['is_edge']), BINARY_FILTERED_COLOR),
                ('edge', np.flatnonzero(result['is_edge']), NOT_EVALUABLE_COLOR)):
            if len(indices):
                layers[label] = dict(contours=[contours[i] for i in indices], color=color,
                                     alpha=1, fill_alpha=0.45, line_width=2.5,
                                     line_width_mode='absolute', render_scale=args.render_scale)
    fig, axes = panel_grid(len(field["images"]))
    temporary = path.with_suffix(".tmp.png")
    try:
        for ax, r in zip(axes.flat, sorted(field["images"])):
            rendered = mvt.render_contours(
                display_dapi(field["images"][r], args), [contours[i] for i in kept], color=BINARY_KEPT_COLOR, markers=layers,
                fill_alpha=args.fill_alpha, line_width=args.line_width, line_width_mode="absolute",
                render_scale=args.render_scale, intensity_range=(0, 255))
            ax.imshow(rendered)
            title = f"Round {r}"
            if r == reference_round:
                title += f" (reference)\n{len(kept)} kept · {len(rejected)} filtered"
            else:
                title += f"\n{np.count_nonzero(result['passing_by_round'][r])}/{len(contours)} pass coverage here"
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        fig.suptitle(f"{field['label']}\nReference Round {reference_round} contours · "
                     f"coverage > {result['iou_threshold']:g} in at least {result['min_other_rounds']}/{len(field['images'])-1} other rounds · "
                     f"{len(kept)}/{len(contours)} kept", fontsize=12)
        handles = [Patch(color=BINARY_KEPT_COLOR, label="Kept reference nucleus (overall)")]
        if not args.hide_rejected:
            handles.extend([Patch(color=BINARY_FILTERED_COLOR, label="Filtered reference nucleus (overall)"),
                            Patch(color=NOT_EVALUABLE_COLOR, label="Edge (excluded)")])
        fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.94))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
        fig.savefig(temporary, dpi=args.dpi, facecolor="white")
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def binary_assignment_record(result):
    """All per-object arrays index reference contours, never current-round instances."""
    field, reference_round = result["field"], result["reference_round"]
    kept = result["kept_reference_indices"]
    reference_count = len(result["keep"])
    rounds = {}
    for r in sorted(field["images"]):
        n_pass = int(np.count_nonzero(result["passing_by_round"][r]))
        rounds[r] = dict(image_path=field["paths"][r], contour_path=field["contour_paths"][r],
                         reference_coverage=result["coverage_by_round"][r].tolist(),
                         passes_coverage=result["passing_by_round"][r].tolist(),
                         counts=dict(reference_count=reference_count, detected_count=len(field["contours"][r]),
                                     kept=len(kept), filtered=reference_count - len(kept),
                                     coverage_pass=n_pass, coverage_fail=reference_count - n_pass,
                                      edge_count=int(np.count_nonzero(result["is_edge"]))))
    return dict(patch_key="__".join(field["key"]), filter_mode="binary_mask_reference",
                score_metric="reference_coverage", reference_round=reference_round,
                iou_threshold=result["iou_threshold"], threshold_rule=">",
                min_other_rounds=result["min_other_rounds"], shape=list(field["shape"]),
                reference_count=reference_count, reference_contour_path=field["contour_paths"][reference_round],
                reference_ids=list(range(1, reference_count + 1)),
                reference_area_pixels=result["reference_areas"].tolist(),
                passing_other_round_counts=result["passing_other_counts"].tolist(),
                is_edge=result["is_edge"].tolist(),
                statuses=np.where(result["is_edge"], "not_evaluable", np.where(result["keep"], "kept", "filtered")).tolist(),
                kept_reference_ids=(kept + 1).tolist(), kept_reference_contour_indices=kept.tolist(),
                status_colors=dict(kept=BINARY_KEPT_COLOR, filtered=BINARY_FILTERED_COLOR, not_evaluable=NOT_EVALUABLE_COLOR), rounds=rounds,
                score_definition="area(reference nucleus intersect round foreground) / area(reference nucleus); "
                                 "foreground is the union of unsmoothed contours; zero-area references score 0",
                index_note="All object arrays use the reference contour order. IDs are one-based, indices zero-based. "
                           "No Hungarian assignments or per-round instance IDs are inferred. The reference round "
                           "is excluded from passing_other_round_counts. Colors show overall keep/filter status.")


def prepare_registration_settings(args):
    """Read the run's frozen parameters, with the submitted config as fallback."""
    import hashlib
    preprocessing = args.tools_dir / 'myonis_cosmx/4i/preprocessing'
    config_path = args.registration_config
    if config_path is None:
        frozen = args.run_dir / 'provenance/pipeline_config.json'
        config_path = frozen if frozen.is_file() else preprocessing / 'config/pipeline_config.json'
    args.registration_config = config_path.expanduser().resolve()
    payload = json.loads(args.registration_config.read_text(encoding='utf-8'))
    if 'registration' not in payload:
        raise ValueError(f'Missing registration section: {args.registration_config}')
    args.registration_settings = payload['registration']
    args.registration_source_hashes = {}
    for name in ('patch_level.py', 'registration.py', 'config.py', 'qc_visualization.py'):
        path = preprocessing / 'utils' / name
        args.registration_source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.registration_config_sha256 = hashlib.sha256(args.registration_config.read_bytes()).hexdigest()


def registration_dependencies(tools_dir):
    """Import the actual preprocessing functions under an isolated package name.

    Do not import a generic `utils`: other active projects can use that name.
    No copied registration implementation and no change to preprocessing files.
    """
    import hashlib
    import importlib
    import importlib.util
    root = tools_dir / 'myonis_cosmx/4i/preprocessing/utils'
    package = '_qc_preprocessing_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, root / '__init__.py',
                                                       submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return tuple(importlib.import_module(f'{package}.{name}')
                 for name in ('patch_level', 'registration', 'config', 'qc_visualization'))


def corrected_dapi_pair(reference, moving, valid, sigma, gain_limits=(0.25, 4.0), return_images=False):
    """Background correction, one robust field-wide gain, and noise smoothing.

    Unlike registration's independent percentile scaling, scoring preserves local
    loss. Estimate a single gain from mutually bright pixels (most must be stable).
    Empty/near-empty moving signal is never independently scaled up to full range.
    """
    from scipy.ndimage import gaussian_filter
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if not np.any(valid):
        raise ValueError('No valid image overlap after re-registration')
    ref_background = float(np.percentile(reference[valid], 1))
    mov_background = float(np.percentile(moving[valid], 1))
    ref = np.maximum(reference - ref_background, 0)
    mov = np.maximum(moving - mov_background, 0)
    scale = float(np.percentile(ref[valid], 99.8))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('No usable reference DAPI dynamic range')
    floor = 0.10 * scale
    # No object-specific normalization: disappearing objects must stay dim.
    calibration = valid & (ref > floor) & (mov > floor)
    ratios = ref[calibration] / np.maximum(mov[calibration], np.finfo(np.float32).eps)
    raw_gain = float(np.median(ratios)) if ratios.size >= 64 else 1.0
    gain = float(np.clip(raw_gain, *gain_limits))
    mov *= gain
    if sigma > 0:
        # Normalize convolution at invalid translation borders instead of treating
        # zero-filled pixels as lost DAPI or spreading their zeros into valid data.
        weights = gaussian_filter(valid.astype(np.float32), sigma, mode='constant')
        ref = gaussian_filter(np.where(valid, ref, 0), sigma, mode='constant') / np.maximum(weights, 1e-6)
        mov = gaussian_filter(np.where(valid, mov, 0), sigma, mode='constant') / np.maximum(weights, 1e-6)
    change = np.minimum(np.abs(ref - mov) / np.maximum(ref, floor), 1.0).astype(np.float32)
    change[~valid] = np.nan
    correction = dict(reference_background=ref_background, moving_background=mov_background,
                      moving_gain=gain, raw_moving_gain=raw_gain, gain_limits=list(gain_limits),
                      gain_was_clipped=bool(gain != raw_gain), gain_calibration_pixels=int(ratios.size),
                      reference_scale=scale, denominator_floor=floor, smoothing_sigma_px=sigma)
    return (change, correction, ref, mov) if return_images else (change, correction)


def aggregate_change_maps(maps, comparisons, common_valid):
    """Same valid support and fixed range for both temporal reducers; no reference self-pair."""
    values = np.asarray([maps[r][common_valid] for r in comparisons], dtype=np.float32)
    mean_change = np.full(common_valid.shape, np.nan, dtype=np.float32)
    max_change = np.full(common_valid.shape, np.nan, dtype=np.float32)
    max_round = np.zeros(common_valid.shape, dtype=np.uint8)
    mean_change[common_valid] = np.mean(values, axis=0)
    max_change[common_valid] = np.max(values, axis=0)
    max_round[common_valid] = np.asarray(comparisons, dtype=np.uint8)[np.argmax(values, axis=0)]
    return mean_change, max_change, max_round


def comparison_rounds(result):
    """Scored comparisons; registration and display may include additional rounds."""
    return result.get('comparison_rounds', [r for r in sorted(result['field']['images'])
                                            if r != result['reference_round']])


def registration_filter(field, args):
    """Mean/max non-reference DAPI change maps, pooled within reference masks."""
    patch_registration, registration, config_module, _ = registration_dependencies(args.tools_dir)
    settings = config_module.RegistrationSettings(**args.registration_settings)
    reference_round = args.reference_round
    reference = field['images'][reference_round]
    registered_images = {reference_round: reference}
    registrations = {}
    maps, valid_masks = {}, {}
    all_comparisons = [r for r in sorted(field['images']) if r != reference_round]
    filter_rounds = getattr(args, 'filter_rounds', None) or sorted(field['images'])
    comparisons = [r for r in all_comparisons if r in filter_rounds]
    for r in all_comparisons:
        moving = field['images'][r]
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(moving)):
            raise ValueError(f'Round {r}: nonfinite DAPI pixels')
        # Identical helper/parameters to register_patch_level.py; the current QC
        # unit is an exported patch, so explicitly supply its shape (usually 512²).
        checked = patch_registration.register_patch_field(reference, moving, settings,
                                                           expected_shape=field['shape'])
        if checked['global_qc_status'] != 'pass':
            raise ValueError(f"Round {r}: preprocessing re-registration QC failed: "
                             f"{checked['global_qc_reasons']}; no object decisions were made")
        registered = registration.apply_translation(moving, checked['shift_y'], checked['shift_x'])
        registered_images[r] = registered
        valid = np.zeros(field['shape'], dtype=bool)
        y0, y1, x0, x1 = checked['valid_bounds']
        valid[y0:y1, x0:x1] = True
        if r in comparisons:
            maps[r], intensity = corrected_dapi_pair(reference, registered, valid, args.registration_smoothing_sigma)
        valid_masks[r] = valid
        # Low-information failure records contain NaNs, but successful registration
        # should only save finite scalars; represent any optional nonfinite value as null.
        registrations[r] = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                             for key, value in checked.items()}
        if r in comparisons:
            registrations[r]['intensity_correction'] = intensity
    common_valid = np.logical_and.reduce([valid_masks[r] for r in comparisons])
    if not np.any(common_valid):
        raise ValueError('No common valid pixels across the re-registered comparisons')
    # Every evaluated pixel has all chosen comparisons. Never average signed
    # differences or include the reference's trivial zero difference.
    mean_change, max_change, max_round = aggregate_change_maps(maps, comparisons, common_valid)
    reference_pixels = [np.flatnonzero(mvt.contours_to_masks([c], field['shape'])[0])
                        for c in field['contours'][reference_round]]
    areas = np.asarray([len(p) for p in reference_pixels], dtype=int)
    valid_pixels = [p[common_valid.ravel()[p]] for p in reference_pixels]
    valid_fractions = np.asarray([len(v) / len(p) if len(p) else 0.0
                                  for p, v in zip(reference_pixels, valid_pixels)])
    evaluable = (areas > 0) & (valid_fractions >= args.registration_min_valid_fraction)
    # Low-signal reference regions cannot establish whether a nucleus was present.
    ref_values = np.maximum(reference.astype(float) - np.percentile(reference[common_valid], 1), 0)
    ref_signal = np.asarray([float(np.mean(ref_values.ravel()[p])) if len(p) else 0.0
                            for p in valid_pixels])
    is_edge = reference_edge_flags(field, reference_round)
    evaluable &= (ref_signal > 0) & ~is_edge
    def pool_map(change_map):
        return np.asarray([float(np.mean(change_map.ravel()[p])) if ok else np.nan
                           for p, ok in zip(valid_pixels, evaluable)])
    mean_scores = pool_map(mean_change)
    max_scores = pool_map(max_change)
    scores = mean_scores if args.registration_aggregation == 'mean' else max_scores
    by_round = {r: np.asarray([float(np.mean(maps[r].ravel()[p])) if ok else np.nan
                               for p, ok in zip(valid_pixels, evaluable)]) for r in comparisons}
    by_round[reference_round] = np.where(evaluable, 0.0, np.nan)
    for r in set(field['images']) - set(comparisons) - {reference_round}:
        by_round[r] = np.full(len(areas), np.nan)
    worst_round_mean_scores = np.max(np.asarray([by_round[r] for r in comparisons]), axis=0)
    keep = evaluable & (scores <= args.registration_change_threshold)
    statuses = np.where(~evaluable, 'not_evaluable', np.where(keep, 'kept', 'filtered_change'))
    return dict(field=field, reference_round=reference_round,
                comparison_rounds=comparisons,
                registered_images=registered_images, registrations=registrations,
                mean_change=mean_change, max_change=max_change, max_change_round=max_round,
                common_valid=common_valid, aggregation=args.registration_aggregation, maps_by_round=maps,
                is_edge=is_edge, metric_title="Relative DAPI change",
                mean_scores=mean_scores, max_scores=max_scores,
                worst_round_mean_scores=worst_round_mean_scores,
                scores=scores, scores_by_round=by_round, reference_areas=areas,
                valid_fractions=valid_fractions, evaluable=evaluable, keep=keep,
                statuses=statuses, kept_reference_indices=np.flatnonzero(keep), _reference_pixels=reference_pixels)


def registration_metric_specs(args):
    """Exploratory cutoffs are independent; scores are not probabilities."""
    return {
        'relative_calibrated': ('Relative difference, wider gain range', args.registration_change_threshold),
        'ssim': ('Local SSIM change', args.registration_ssim_threshold),
        'zncc_signal': ('Local ZNCC + signal loss', args.registration_zncc_threshold),
        'gradient': ('Normalized gradient + signal loss', args.registration_gradient_threshold),
    }


def comparison_pair_maps(reference, moving, valid, args):
    """Whole-field gain only. No per-nucleus normalization or further warping.

    SSIM: 1 - clip(SSIM, 0, 1), on jointly scaled [0,1] inputs.
    ZNCC: 1 - clip(local Pearson correlation, 0, 1).
    Gradient: 1 - clip(local gradient-vector cosine, 0, 1).
    ZNCC/gradient use max(structural change, local fractional signal loss).
    Flat reference windows have no structural evidence; use signal loss there.
    All local maps exclude windows intersecting invalid registration borders.
    """
    from scipy.ndimage import uniform_filter, minimum_filter
    from skimage.metrics import structural_similarity
    relative, correction, ref, mov = corrected_dapi_pair(
        reference, moving, valid, args.registration_smoothing_sigma,
        gain_limits=tuple(args.registration_comparison_gain_limits), return_images=True)
    window = args.registration_local_window
    if min(ref.shape) < window + 2:
        raise ValueError('Image too small for --registration-local-window')
    scale = correction['reference_scale']
    # Preserve dynamic range: clipping would destroy correlation under local brightening.
    x = (ref / scale).astype(np.float64)
    y = (mov / scale).astype(np.float64)
    avg = lambda a: uniform_filter(a, size=window, mode='constant')
    mx, my = avg(x), avg(y)
    vx = np.maximum(avg(x*x) - mx*mx, 0)
    vy = np.maximum(avg(y*y) - my*my, 0)
    cov = avg(x*y) - mx*my
    noise_variance = args.registration_texture_floor ** 2
    textured = vx > noise_variance
    correlation = np.divide(cov, np.sqrt(vx*vy), out=np.zeros_like(cov),
                            where=(vx*vy > 1e-16))
    zncc = np.where(textured, 1 - np.clip(correlation, 0, 1), 0)
    # Loss uses UNCLIPPED corrected images, rather than independently rescaled crops.
    local_ref, local_mov = avg(ref), avg(mov)
    loss = np.clip((local_ref-local_mov) / np.maximum(local_ref, correction['denominator_floor']), 0, 1)
    ssim_scale = float(max(np.max(ref[valid]), np.max(mov[valid]), 1e-12))
    correction["ssim_data_scale"] = ssim_scale
    _, ssim = structural_similarity(ref.astype(np.float64)/ssim_scale, mov.astype(np.float64)/ssim_scale,
                                    win_size=window, data_range=1.0,
                                    gaussian_weights=False, use_sample_covariance=False, full=True)
    gx = np.gradient(x)
    gy = np.gradient(y)
    ex = np.maximum(avg(gx[0]**2 + gx[1]**2), 0)
    ey = np.maximum(avg(gy[0]**2 + gy[1]**2), 0)
    dot = avg(gx[0]*gy[0] + gx[1]*gy[1])
    cosine = np.divide(dot, np.sqrt(ex*ey), out=np.zeros_like(dot), where=ex*ey > 1e-16)
    gradient = np.where(ex > noise_variance, 1 - np.clip(cosine, 0, 1), 0)
    # One extra pixel on every side accommodates np.gradient's finite differences.
    local_valid = minimum_filter(valid.astype(np.uint8), size=window+2, mode='constant', cval=0).astype(bool)
    maps = dict(relative_calibrated=relative, ssim=1-np.clip(ssim, 0, 1),
                zncc_signal=np.maximum(zncc, loss), gradient=np.maximum(gradient, loss))
    for name in maps:
        maps[name] = np.asarray(maps[name], dtype=np.float32)
        maps[name][~(valid if name == 'relative_calibrated' else local_valid)] = np.nan
    components = dict(zncc_structure=zncc, gradient_structure=gradient, signal_loss=loss)
    for name in components:
        components[name] = np.asarray(components[name], dtype=np.float32)
        components[name][~local_valid] = np.nan
    return maps, components, correction


def score_comparison_maps(base, maps, threshold, name, title, args):
    """Pool on each method's common finite support; retain not-evaluable separately."""
    field, ref = base['field'], base['reference_round']
    comparisons = comparison_rounds(base)
    valid = np.logical_and.reduce([np.isfinite(maps[r]) for r in comparisons])
    mean, maximum, max_round = aggregate_change_maps(maps, comparisons, valid)
    contour_pixels = base.get('_reference_pixels')
    if contour_pixels is None:
        contour_pixels = [np.flatnonzero(mvt.contours_to_masks([c], field['shape'])[0])
                          for c in field['contours'][ref]]
    inside = [p[valid.ravel()[p]] for p in contour_pixels]
    fractions = np.asarray([len(v)/len(p) if len(p) else 0.0 for p, v in zip(contour_pixels, inside)])
    evaluable = base['evaluable'] & (fractions >= args.registration_min_valid_fraction)
    def pool(a):
        return np.asarray([float(np.mean(a.ravel()[p])) if ok and len(p) else np.nan
                           for p, ok in zip(inside, evaluable)])
    by_round = {r: pool(maps[r]) for r in comparisons}
    by_round[ref] = np.where(evaluable, 0.0, np.nan)
    for r in set(field['images']) - set(comparisons) - {ref}:
        by_round[r] = np.full(len(fractions), np.nan)
    mean_scores, max_scores = pool(mean), pool(maximum)
    scores = mean_scores if args.registration_aggregation == 'mean' else max_scores
    keep = evaluable & (scores <= threshold)
    return dict(base, metric_name=name, metric_title=title, threshold=threshold,
                common_valid=valid, mean_change=mean, max_change=maximum,
                max_change_round=max_round, mean_scores=mean_scores, max_scores=max_scores,
                scores=scores, scores_by_round=by_round, maps_by_round=maps,
                valid_fractions=fractions, evaluable=evaluable, keep=keep,
                worst_round_mean_scores=np.max([by_round[r] for r in comparisons], axis=0),
                statuses=np.where(~evaluable, 'not_evaluable', np.where(keep, 'kept', 'filtered_change')),
                kept_reference_indices=np.flatnonzero(keep))


def compare_registration_metrics(base, args, names=None):
    comparisons = comparison_rounds(base)
    specs = registration_metric_specs(args)
    if names is not None:
        specs = {name: specs[name] for name in names}
    all_maps = {name: {} for name in specs}
    corrections, components = {}, {}
    reference = base['field']['images'][base['reference_round']]
    for r in comparisons:
        valid = np.zeros(base['field']['shape'], dtype=bool)
        y0, y1, x0, x1 = base['registrations'][r]['valid_bounds']
        valid[y0:y1, x0:x1] = True
        pair, component, correction = comparison_pair_maps(reference, base['registered_images'][r], valid, args)
        for name in specs:
            all_maps[name][r] = pair[name]
        components[r], corrections[r] = component, correction
    results = {}
    for name, (title, threshold) in specs.items():
        result = score_comparison_maps(base, all_maps[name], threshold, name, title, args)
        result['comparison_corrections'] = corrections
        result['comparison_components'] = components
        results[name] = result
    return results


def save_metric_map_plot(result, path, args):
    """All local change maps, mean, and max; unlike the appearance overlay."""
    comparisons = comparison_rounds(result)
    fig, axes = panel_grid(len(comparisons)+2, columns=3 if len(comparisons)+2 <= 9 else 4,
                           width=13/3, height=13/3)
    cmap = plt.get_cmap(args.registration_score_cmap).copy()
    cmap.set_bad(NOT_EVALUABLE_COLOR)
    temporary = path.with_suffix('.tmp.png')
    try:
        maps = [result['maps_by_round'][r] for r in comparisons] + [result['mean_change'], result['max_change']]
        titles = [f'Round {r} vs reference {result["reference_round"]}' for r in comparisons] + ['Mean change map', 'Max change map']
        for ax, values, title in zip(axes.flat, maps, titles):
            # Use exactly the same support as the object scores.
            values = np.where(result['common_valid'], values, np.nan)
            heat = ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        fig.colorbar(heat, ax=list(axes.flat), shrink=0.65, label='Change score (0–1): low → high')
        fig.suptitle(f"{result['field']['label']}\n{result['metric_title']} · grey = invalid window", fontsize=12)
        fig.savefig(temporary, dpi=args.dpi, facecolor='white')
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def save_metric_comparisons(base, stem, args, results=None):
    """Save comparisons alongside the selected main method; reuse computed maps."""
    if results is None:
        results = compare_registration_metrics(base, args)
    outputs = {}
    for name, result in results.items():
        root = args.output_dir / 'registration_metrics' / name
        folders = ['assignments']
        if not args.fast_mode:
            folders += ['plots', 'registration_scores', 'registration_overlays']
            if not getattr(args, 'skip_dense_maps', False):
                folders.append('registration_maps')
        for folder in folders:
            (root / folder).mkdir(parents=True, exist_ok=True)
        metric_args = argparse.Namespace(**vars(args))
        metric_args.registration_change_threshold = result['threshold']
        if not args.fast_mode:
            save_registration_plots(result, root/'plots'/f'{stem}.png',
                                    root/'registration_overlays'/f'{stem}.png',
                                    None if getattr(args, 'skip_dense_maps', False) else root/'registration_maps'/f'{stem}.npz', metric_args)
            save_registration_score_plot(result, root/'registration_scores'/f'{stem}.png', metric_args)
        record = registration_assignment_record(result, metric_args)
        record.update(score_metric=name, score_definition=metric_definitions(args)[name],
                      comparison_intensity_correction=result['comparison_corrections'],
                      exploratory_threshold=True,
                      score_settings=dict(local_window=args.registration_local_window,
                                          texture_floor=args.registration_texture_floor,
                                          comparison_gain_limits=args.registration_comparison_gain_limits))
        # Save component object scores so loss and structural disagreement remain distinguishable.
        component_scores = {}
        for component in ('zncc_structure', 'gradient_structure', 'signal_loss'):
            values = {r: result['comparison_components'][r][component] for r in result['comparison_components']}
            pooled = score_comparison_maps(base, values, result['threshold'], component, component, args)
            component_scores[component] = dict(mean_score=finite_list(pooled['mean_scores']),
                                               max_score=finite_list(pooled['max_scores']),
                                               scores_by_round={r: finite_list(v) for r, v in pooled['scores_by_round'].items()})
        record['component_scores'] = component_scores if name in ('zncc_signal', 'gradient') else {}
        assignment = root/'assignments'/f'{stem}.json'
        write_json(assignment, record)
        outputs[name] = dict(assignments=str(assignment.relative_to(args.output_dir)),
                             reference_count=len(result['keep']), kept_count=int(np.count_nonzero(result['keep'])),
                             max_kept_count=int(np.count_nonzero(result['max_scores'] <= result['threshold'])),
                             not_evaluable_count=int(np.count_nonzero(~result['evaluable'])))
    return outputs


def metric_definitions(args):
    prefix = ('Background correction and a single robust field gain in '
              f'{list(args.registration_comparison_gain_limits)}; no per-nucleus rescaling. ')
    return {
        'relative_calibrated': prefix + 'clip(abs(R-J)/max(R,0.1*P99.8(R)),0,1). Same baseline formula with wider gain limits.',
        'ssim': prefix + '1-clip(local SSIM,0,1); uniform window, population covariance; '
                'both corrected images divided by their joint valid maximum (one shared scale, no clipping), data_range=1.',
        'zncc_signal': prefix + 'max(1-clip(local zero-mean normalized correlation,0,1), local fractional DAPI loss). '
                       'Untextured reference windows use loss only; correlation is not squared.',
        'gradient': prefix + 'max(1-clip(local cosine of gradient-vector patches,0,1), local fractional DAPI loss). '
                    'Untextured reference windows use loss only. This is a gradient-similarity variant, not displacement.',
    }


def write_metric_comparison_statistics(output_dir, summary):
    """Rebuild comparison summaries without image dependencies, including partial runs."""
    settings = summary['settings']
    round_numbers = settings.get('rounds', ROUNDS)
    records, objects = [], []
    for entry in summary.get('completed', []):
        for name, output in entry.get('metric_comparisons', {}).items():
            data = json.loads((output_dir/output['assignments']).read_text())
            n = data['reference_count']
            threshold = data['change_threshold']
            mean_keep = sum(v is not None and v <= threshold for v in data['mean_change_score'])
            max_keep = sum(v is not None and v <= threshold for v in data['max_change_score'])
            common = dict(patch_key=entry['patch_key'], metric=name, threshold=threshold,
                          aggregation=data['score_aggregation'], reference_round=data['reference_round'])
            records.append(dict(common, reference_count=n, **retention_statistics(n, len(data['kept_reference_ids'])),
                                evaluated_count=n-data['statuses'].count('not_evaluable'),
                                high_change_count=data['statuses'].count('filtered_change'),
                                mean_kept_count=mean_keep, max_kept_count=max_keep,
                                max_kept_percentage=100*max_keep/n if n else None,
                                not_evaluable_count=data['statuses'].count('not_evaluable'),
                                edge_count=sum(data.get('is_edge', []))))
            for i in range(n):
                maximum = data['max_change_score'][i]
                objects.append(dict(common, reference_id=i+1, decision=data['statuses'][i],
                                    is_edge=data.get('is_edge', [False]*n)[i],
                                    exclusion_reason=data.get('exclusion_reason', [None]*n)[i],
                                    change_score=data['change_score'][i], mean_score=data['mean_change_score'][i],
                                    max_score=maximum, worst_round_mean_score=data['worst_round_mean_change_score'][i],
                                    max_decision='not_evaluable' if maximum is None else ('kept' if maximum <= threshold else 'filtered_change'),
                                    valid_area_fraction=data['valid_area_fraction'][i],
                                    **{f'round_{r}_score': data['rounds'][str(r)]['mean_change_score'][i] for r in round_numbers}))
    root = output_dir/'registration_metrics'
    root.mkdir(parents=True, exist_ok=True)
    field_columns = ['patch_key','metric','threshold','aggregation','reference_round','reference_count',
                     'kept_count','filtered_count','kept_proportion','kept_percentage','filtered_proportion','filtered_percentage',
                     'evaluated_count','high_change_count','mean_kept_count','max_kept_count','max_kept_percentage','not_evaluable_count','edge_count']
    object_columns = ['patch_key','metric','threshold','aggregation','reference_round','reference_id','decision','is_edge','exclusion_reason',
                      'change_score','mean_score','max_score','worst_round_mean_score','max_decision','valid_area_fraction'] + [f'round_{r}_score' for r in round_numbers]
    write_csv(root/'field_comparison.csv', field_columns, records)
    for name in sorted({v['metric'] for v in records}):
        (root/name).mkdir(parents=True, exist_ok=True)
        write_csv(root/name/'field_summary.csv', field_columns, [v for v in records if v['metric']==name])
        write_csv(root/name/'nucleus_summary.csv', object_columns, [v for v in objects if v['metric']==name])
    write_json(root/'metrics_metadata.json', dict(definitions=metric_definitions(argparse.Namespace(**settings)),
               local_window=settings['registration_local_window'], texture_floor=settings['registration_texture_floor'],
               aggregation_order='Pixelwise mean/max across non-reference rounds, then area mean within reference contours.',
               validity='Local methods require the whole window plus gradient halo to be inside all registered images. Grey means not evaluable.',
               cutoff_note='Independent exploratory cutoffs; not calibrated to equal false-positive rates. Review examples before choosing a production filter.',
               baseline='Main outputs use the selected registration metric. All methods share re-registered images and exclude reference edges.',
               run_status=summary.get('status'), failed_patches=summary.get('failed', [])))


def finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def registration_assignment_record(result, args):
    field = result['field']
    reference_round = result['reference_round']
    n = len(result['keep'])
    kept = result['kept_reference_indices']
    comparisons = comparison_rounds(result)
    rounds = {}
    for r in sorted(field['images']):
        values = result['scores_by_round'][r]
        passing = np.isfinite(values) & (values <= args.registration_change_threshold)
        n_evaluable = int(np.count_nonzero(np.isfinite(values)))
        rounds[r] = dict(image_path=field['paths'][r],
                         registration=result['registrations'].get(r),
                         used_for_filtering=r in comparisons,
                         mean_change_score=finite_list(values),
                         counts=dict(reference_count=n, kept=len(kept), filtered=n - len(kept),
                                     change_pass=int(np.count_nonzero(passing)),
                                     change_fail=n_evaluable - int(np.count_nonzero(passing)),
                                     not_evaluable=n - n_evaluable if r == reference_round or r in comparisons else 0,
                                     not_scored=n if r != reference_round and r not in comparisons else 0))
    return dict(patch_key='__'.join(field['key']), filter_mode='registration_based',
                score_metric=result.get('metric_name', f'{result["aggregation"]}_relative_absolute_dapi_change'),
                score_aggregation=result['aggregation'],
                reference_round=reference_round, shape=list(field['shape']), reference_count=n,
                change_threshold=args.registration_change_threshold, threshold_rule='<=',
                comparison_rounds=comparisons,
                excluded_filter_rounds=sorted(set(field['images'])-set(comparisons)-{reference_round}),
                reference_contour_path=field['contour_paths'][reference_round],
                reference_ids=list(range(1, n + 1)), reference_area_pixels=result['reference_areas'].tolist(),
                valid_area_fraction=result['valid_fractions'].tolist(),
                is_edge=result.get('is_edge', np.zeros(n, dtype=bool)).tolist(),
                exclusion_reason=[('is_edge' if edge else 'not_evaluable') if status == 'not_evaluable' else
                                  'high_change' if status == 'filtered_change' else None
                                  for edge, status in zip(result.get('is_edge', np.zeros(n, dtype=bool)), result['statuses'])],
                change_score=finite_list(result['scores']),
                mean_change_score=finite_list(result['mean_scores']),
                max_change_score=finite_list(result['max_scores']),
                worst_round_mean_change_score=finite_list(result['worst_round_mean_scores']),
                statuses=result['statuses'].tolist(),
                kept_reference_ids=(kept + 1).tolist(), kept_reference_contour_indices=kept.tolist(),
                registration_settings=args.registration_settings,
                registration_config=str(args.registration_config),
                registration_config_sha256=args.registration_config_sha256,
                registration_source_hashes=args.registration_source_hashes,
                status_colors=dict(kept=BINARY_KEPT_COLOR, filtered=BINARY_FILTERED_COLOR,
                                   not_evaluable=NOT_EVALUABLE_COLOR, is_edge=NOT_EVALUABLE_COLOR), rounds=rounds,
                comparison_intensity_correction=result.get('comparison_corrections'),
                score_settings=dict(local_window=args.registration_local_window,
                                    texture_floor=args.registration_texture_floor,
                                    comparison_gain_limits=args.registration_comparison_gain_limits),
                score_definition=(metric_definitions(args)[result['metric_name']] +
                                  f' Pixelwise {result["aggregation"]} across rounds, then AREA MEAN within reference contour.'
                                  if result.get('metric_name') else f'{result["aggregation"].capitalize()} over {len(comparisons)} non-reference rounds at each pixel, '
                                 'then AREA MEAN inside each reference contour. Pixel change: '
                                 'clip(abs(R - gain*J) / max(R, 0.1*P99.8(R)), 0, 1). '
                                 'R/J are background-corrected, optionally Gaussian-smoothed DAPI. '
                                 'One field-wide moving gain is estimated from mutually bright pixels, capped '
                                 'at [0.25,4]. Zero is unchanged; one is maximal relative change. '
                                 'This is neither IoU nor a probability of detachment.'),
                not_evaluable_definition=f'Edge object (is_edge), empty contour, no reference signal, or less than '
                                        f'{args.registration_min_valid_fraction:g} of contour area valid '
                                        'in all non-reference comparisons; excluded, not interpreted as detached.',
                index_note='All object arrays index original reference contours; no non-reference segmentation '
                           'is used. Colors show the overall selected-score decision in every panel. '
                           'min-other-rounds and iou-threshold do not apply in this mode.')


def save_registration_plots(result, plot_path, overlay_path, map_path, args):
    field = result['field']
    ref = result['reference_round']
    contours = field['contours'][ref]
    kept = result['kept_reference_indices']
    rejected = np.flatnonzero(~result['keep'])
    layers = {}
    if not args.hide_rejected:
        for label, indices, color in (
                ('filtered', np.flatnonzero(result['evaluable'] & ~result['keep']), BINARY_FILTERED_COLOR),
                ('not_evaluable', np.flatnonzero(~result['evaluable']), NOT_EVALUABLE_COLOR)):
            if len(indices):
                layers[label] = dict(contours=[contours[i] for i in indices], color=color,
                                     alpha=1, fill_alpha=0.45, line_width=2.5,
                                     line_width_mode='absolute', render_scale=args.render_scale)
    fig, axes = panel_grid(len(field['images']))
    temporary = plot_path.with_suffix('.tmp.png')
    try:
        for ax, r in zip(axes.flat, sorted(field['images'])):
            shown = display_dapi(result['registered_images'][r], args)
            rendered = mvt.render_contours(shown, [contours[i] for i in kept],
                                          color=BINARY_KEPT_COLOR, markers=layers,
                                          fill_alpha=args.fill_alpha, line_width=args.line_width,
                                          line_width_mode='absolute', render_scale=args.render_scale,
                                          intensity_range=(0, 255))
            ax.imshow(rendered)
            if r == ref:
                subtitle = f'(reference)\n{len(kept)} kept · {len(rejected)} excluded'
            elif r not in comparison_rounds(result):
                subtitle = 're-registered\nEXCLUDED FROM DAPI FILTERING'
            else:
                values = result['scores_by_round'][r]
                n_pass = np.count_nonzero(values <= args.registration_change_threshold)
                subtitle = f're-registered\n{n_pass}/{len(contours)} below change cutoff here'
            ax.set_title(f'Round {r} {subtitle}', fontsize=10)
            ax.set_axis_off()
        for axis in axes.flat:
            axis.set_axis_off()
        unknown = int(np.count_nonzero(~result['evaluable']))
        fig.suptitle(f"{field['label']} · {result.get('metric_title', 'Relative DAPI change')}\n{result['aggregation'].capitalize()}-map nucleus score ≤ {args.registration_change_threshold:g} "
                     f"across {len(comparison_rounds(result))} comparisons · {len(kept)}/{len(contours)} kept · {unknown} not evaluable", fontsize=12)
        handles = [Patch(color=BINARY_KEPT_COLOR, label='Kept reference nucleus (overall)')]
        if not args.hide_rejected:
            handles.extend([Patch(color=BINARY_FILTERED_COLOR, label='Filtered: high change'),
                            Patch(color=NOT_EVALUABLE_COLOR, label='N/A or edge (excluded)')])
        fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.94))
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9)
        fig.savefig(temporary, dpi=args.dpi, facecolor='white')
        temporary.replace(plot_path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    comparisons = comparison_rounds(result)
    save_metric_map_plot(result, overlay_path, args)
    if map_path is None:
        return
    temporary = map_path.with_suffix('.tmp.npz')
    try:
        np.savez_compressed(temporary, mean_change=result['mean_change'], max_change=result['max_change'],
                            max_change_round=result['max_change_round'],
                            score_aggregation=np.asarray(result['aggregation']),
                            metric=np.asarray(result.get('metric_name', 'relative')),
                            common_valid=result['common_valid'],
                            comparison_rounds=np.asarray(comparisons),
                            reference_round=np.asarray(ref),
                            change_threshold=np.asarray(args.registration_change_threshold))
        temporary.replace(map_path)
    finally:
        temporary.unlink(missing_ok=True)


def save_registration_appearance_plot(result, overlay_path, args):
    """Compatibility wrapper: scalar change maps now use the shared display palette."""
    save_metric_map_plot(result, overlay_path, args)


def save_registration_alignment_plot(result, path, args):
    """Visualize the saved DAPI translations, before and after, for every comparison."""
    field, ref = result['field'], result['reference_round']
    comparisons = [r for r in sorted(field['images']) if r != ref]
    reference = display_dapi(field['images'][ref], args)[..., 0].astype(np.float32)/255
    fig, axes = panel_grid(2*len(comparisons))
    temporary = path.with_suffix('.tmp.png')
    try:
        for i, r in enumerate(comparisons):
            registration = result['registrations'][r]
            for j, image in enumerate((field['images'][r], result['registered_images'][r])):
                moving = display_dapi(image, args)[..., 0].astype(np.float32)/255
                overlay = np.stack((reference, moving, reference), axis=-1)
                axis = axes.flat[2*i+j]
                axis.imshow(overlay, interpolation='nearest')
                title = f'Round {r}: ' + ('before' if j == 0 else 'after')
                if r not in comparison_rounds(result):
                    title += '\nEXCLUDED FROM DAPI FILTERING'
                if j:
                    title += (f"\ndy={registration['shift_y']:.2f}, dx={registration['shift_x']:.2f} px"
                              f" · corr={registration['post_registration_correlation']:.3f}")
                axis.set_title(title, fontsize=10)
        fig.suptitle(f"{field['label']} · DAPI alignment to reference Round {ref}\n"
                     "Reference = magenta · moving = green · overlap = white", fontsize=12)
        fig.supxlabel('Display contrast is adjusted per image; SSIM filtering uses the existing quantitative correction.', fontsize=9)
        fig.savefig(temporary, dpi=args.dpi, facecolor='white')
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def save_registration_score_plot(result, path, args):
    """Compare area-pooled nucleus scores and pixel maps on one fixed color scale."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib import patheffects
    field = result['field']
    ref = result['reference_round']
    contours = field['contours'][ref]
    cmap = plt.get_cmap(args.registration_score_cmap).copy()
    invalid_color = NOT_EVALUABLE_COLOR
    cmap.set_bad(invalid_color)
    norm = Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 5, figsize=(26, 6.5))
    for i, ax in enumerate(axes):
        ax.set_position((0.006 + 0.191*i, 0.14, 0.18, 0.72))
    temporary = path.with_suffix('.tmp.png')
    try:
        for j, aggregation in enumerate(('mean', 'max')):
            scores = result[f'{aggregation}_scores']
            colors = [cmap(norm(value))[:3] if np.isfinite(value) else to_rgb(invalid_color)
                      for value in scores]
            rendered = mvt.render_contours(np.zeros(field['shape'], dtype=np.float32), contours,
                                           colors=colors, alpha=1, fill_alpha=1, line_width=0,
                                           render_scale=args.render_scale)
            axes[2*j].imshow(rendered, interpolation='nearest')
            passes = int(np.count_nonzero(scores <= args.registration_change_threshold))
            axes[2*j].set_title(f'Reference nuclei: {aggregation}-map score\n'
                                f'{passes}/{len(scores)} at or below {args.registration_change_threshold:g}', fontsize=11)
            # IDs link colored objects to nucleus_summary.csv; no intensity blending
            # in the fills, so a nucleus's color corresponds directly to the colorbar.
            for i, contour in enumerate(contours):
                pixels = np.argwhere(mvt.contours_to_masks([contour], field['shape'])[0])
                if not len(pixels):
                    continue
                center = np.mean(pixels, axis=0)
                label = axes[2*j].text(center[1], center[0], str(i+1), fontsize=6,
                                       ha='center', va='center', color='white')
                label.set_path_effects([patheffects.withStroke(linewidth=1.4, foreground='black')])
            axes[2*j+1].imshow(np.ma.masked_invalid(result[f'{aggregation}_change']),
                               cmap=cmap, norm=norm, interpolation='nearest')
            axes[2*j+1].set_title(f'{aggregation.capitalize()} change map\n'
                                  f'Pixelwise {aggregation} across {len(comparison_rounds(result))} comparisons', fontsize=11)
        max_pass = np.isfinite(result['max_scores']) & (result['max_scores'] <= args.registration_change_threshold)
        binary_colors = [to_rgb(invalid_color if not np.isfinite(v) else
                               BINARY_KEPT_COLOR if kept else BINARY_FILTERED_COLOR)
                         for v, kept in zip(result['max_scores'], max_pass)]
        binary = mvt.render_contours(np.zeros(field['shape'], dtype=np.float32), contours,
                                     colors=binary_colors, alpha=1, fill_alpha=1, line_width=0,
                                     render_scale=args.render_scale)
        axes[4].imshow(binary, interpolation='nearest')
        axes[4].set_title(f'Binary decision using MAX\n{np.count_nonzero(max_pass)}/{len(contours)} kept · cutoff {args.registration_change_threshold:g}', fontsize=11)
        axes[4].legend(handles=[Patch(color=BINARY_KEPT_COLOR, label='Kept'),
                               Patch(color=BINARY_FILTERED_COLOR, label='Filtered'),
                               Patch(color=invalid_color, label='N/A or edge (excluded)')],
                       loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=8, frameon=False)
        for ax in axes:
            ax.set_axis_off()
        color_axis = fig.add_axes((0.968, 0.24, 0.006, 0.5))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
        bar.set_label('Change score: 0 = similar, 1 = high change')
        bar.ax.axhline(args.registration_change_threshold, color='cyan', linewidth=1.5)
        fig.suptitle(f"{field['label']} · reference Round {ref} · {result.get('metric_title', 'Relative DAPI change')}\n"
                     f"Active filter: {result['aggregation']}-map nucleus score ≤ {args.registration_change_threshold:g} · "
                     'binary panel always uses MAX; exploratory cutoff', fontsize=12)
        fig.supxlabel('Nucleus score = average of map pixels inside its reference contour. '
                      'Numbers = reference IDs; grey = N/A / edge / invalid.\n'
                      'Max is across rounds at each pixel, not a displacement measurement.', fontsize=10)
        fig.savefig(temporary, dpi=args.dpi, facecolor='white')
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def write_registration_statistics(output_dir, summary):
    """Reference-based counts and per-nucleus mean/per-round scores; stdlib only."""
    settings = summary['settings']
    round_numbers = settings.get('rounds', ROUNDS)
    ref = int(settings['reference_round'])
    filter_rounds = settings.get('filter_rounds') or round_numbers
    comparisons = [r for r in filter_rounds if r != ref]
    threshold = float(settings.get('registration_active_threshold', settings['registration_change_threshold']))
    aggregation = settings.get('registration_aggregation', 'mean')
    run_status = summary.get('status', 'unknown')
    fields, rounds, objects = [], [], []
    n_total = n_kept = n_unknown = n_edge = 0
    identity = ['plate', 'replicate', 'well', 'field', 'cohort', 'patch', 'reference_round',
                'filter_mode', 'score_metric', 'score_aggregation', 'change_threshold']
    metrics = ['kept_count', 'filtered_count', 'kept_proportion', 'kept_percentage',
               'filtered_proportion', 'filtered_percentage']

    def base(entry, status):
        return dict(**patch_columns(entry['patch_key']), patch_key=entry['patch_key'],
                    reference_round=ref, filter_mode='registration_based',
                    score_metric=(settings.get('registration_metric') if settings.get('registration_metric') in
                                  ('ssim', 'zncc_signal', 'gradient', 'relative_calibrated') else
                                  f'{aggregation}_relative_absolute_dapi_change'),
                    score_aggregation=aggregation, change_threshold=threshold,
                    status=status, run_status=run_status)

    for entry in sorted(summary.get('completed', []), key=lambda x: x['patch_key']):
        data = json.loads((output_dir / entry['assignments']).read_text(encoding='utf-8'))
        n, k = data['reference_count'], len(data['kept_reference_ids'])
        if (data['patch_key'] != entry['patch_key'] or data['reference_round'] != ref or
                data['change_threshold'] != threshold or data['filter_mode'] != 'registration_based' or
                data.get('score_aggregation', 'mean') != aggregation or
                n != entry['reference_count'] or k != entry['kept_count']):
            raise ValueError(f"Registration assignment disagrees with summary: {entry['patch_key']}")
        unknown = data['statuses'].count('not_evaluable')
        row = base(entry, 'complete')
        fields.append(dict(row, reference_count=n, **retention_statistics(n, k), not_evaluable_count=unknown,
                           edge_count=sum(data.get('is_edge', [])), high_change_count=data['statuses'].count('filtered_change')))
        n_total += n
        n_kept += k
        n_unknown += unknown
        n_edge += sum(data.get('is_edge', []))
        if set(map(int, data['rounds'])) != set(round_numbers):
            raise ValueError(f'Expected registration records for rounds {list(round_numbers)}')
        for r in round_numbers:
            rd = data['rounds'][str(r)]
            counts, reg = rd['counts'], rd['registration']
            if (counts['change_pass'] + counts['change_fail'] + counts['not_evaluable'] + counts.get('not_scored', 0) != n or
                    counts['kept'] != k or counts['filtered'] != n-k):
                raise ValueError(f'Inconsistent registration counts in Round {r}')
            rounds.append(dict(row, round=r, is_reference=int(r == ref), used_for_filtering=int(r in comparisons), total_count=n,
                               **retention_statistics(n, k), change_pass_count=counts['change_pass'],
                               change_fail_count=counts['change_fail'], not_evaluable_count=counts['not_evaluable'],
                               not_scored_count=counts.get('not_scored', 0),
                               shift_y=reg['shift_y'] if reg else 0, shift_x=reg['shift_x'] if reg else 0,
                               registration_correlation=reg['post_registration_correlation'] if reg else None))
        for i, status in enumerate(data['statuses']):
            round_means = [data['rounds'][str(r)]['mean_change_score'][i] for r in comparisons]
            worst_round_mean = max(round_means) if all(v is not None for v in round_means) else None
            objects.append(dict(row, reference_id=i+1, decision=status,
                                is_edge=data.get('is_edge', [False]*n)[i],
                                exclusion_reason=data.get('exclusion_reason', [None]*n)[i],
                                reference_area_pixels=data['reference_area_pixels'][i],
                                change_score=data.get('change_score', data['mean_change_score'])[i],
                                mean_change_score=data['mean_change_score'][i],
                                max_change_score=data.get('max_change_score', [None]*n)[i],
                                worst_round_mean_change_score=worst_round_mean,
                                valid_area_fraction=data['valid_area_fraction'][i],
                                **{f'round_{r}_change': data['rounds'][str(r)]['mean_change_score'][i]
                                   for r in round_numbers}))
    for entry in summary.get('failed', []):
        fields.append(dict(base(entry, 'failed'), error=entry['error']))
    incomplete = summary.get('discovery', {}).get('incomplete_patches', [])
    for entry in incomplete:
        fields.append(dict(base(entry, 'skipped_incomplete'), missing_rounds=';'.join(map(str, entry['missing_rounds']))))
    tail = ['status', 'run_status', 'missing_rounds', 'error', 'patch_key']
    write_csv(output_dir / 'field_summary.csv', identity + ['reference_count'] + metrics + ['not_evaluable_count', 'edge_count', 'high_change_count'] + tail,
              sorted(fields, key=lambda x: x['patch_key']))
    write_csv(output_dir / 'round_summary.csv', identity + ['round', 'is_reference', 'used_for_filtering', 'total_count'] + metrics +
              ['change_pass_count', 'change_fail_count', 'not_evaluable_count', 'not_scored_count', 'shift_y', 'shift_x',
               'registration_correlation'] + tail, rounds)
    write_csv(output_dir / 'nucleus_summary.csv', identity + ['reference_id', 'decision', 'is_edge', 'exclusion_reason', 'reference_area_pixels', 'change_score',
              'mean_change_score', 'max_change_score', 'worst_round_mean_change_score', 'valid_area_fraction'] + [f'round_{r}_change' for r in round_numbers] + tail, objects)
    metadata = dict(generated_at=datetime.now(timezone.utc).isoformat(), source_run_status=run_status,
                    filter_mode='registration_based', reference_round=ref, change_threshold=threshold,
                    score_aggregation=aggregation,
                    threshold_rule='<=', completed_patches=len(summary.get('completed', [])),
                    failed_patches=len(summary.get('failed', [])),
                    skipped_incomplete_patches=len(incomplete),
                    field_denominator='Original reference nucleus count, per patch',
                    kept_definition=f'Evaluable reference nuclei with area mean of the {aggregation} change map <= cutoff',
                    not_evaluable_definition='Includes is_edge; excluded separately from high-change objects, not evidence of detachment',
                    score_metric=settings.get('registration_metric', 'relative'),
                    filter_rounds=list(filter_rounds), comparison_rounds=comparisons,
                    excluded_filter_rounds=sorted(set(round_numbers)-set(filter_rounds)),
                    aggregation=f'Pixelwise {aggregation} of all {len(comparisons)} selected non-reference maps on their common valid image area, '
                                'then area mean within each reference contour',
                    threshold_note='0.35 is a starting QC cutoff, not a calibrated probability or biological guarantee. '
                                   'Mean dilutes transient changes; max can amplify one-round acquisition artifacts. '
                                   'Scores quantify the selected image-change metric, not displacement.',
                    preprocessing_registration='Actual register_patch_field + apply_translation helpers; no manual '
                                               'preprocessing overrides are reapplied to already registered patches.',
                    registration_settings=settings.get('registration_settings'),
                    overall_reference_statistics=dict(reference_count=n_total, **retention_statistics(n_total, n_kept),
                                                       not_evaluable_count=n_unknown, edge_count=n_edge))
    write_json(output_dir / 'statistics_metadata.json', metadata)
    if settings.get('registration_compare_metrics', False):
        write_metric_comparison_statistics(output_dir, summary)
    if not settings.get('quiet', False):
        print(f'Statistics: {len(fields)} patch rows, {len(objects)} nucleus rows (registration mode).', flush=True)
    return metadata


def save_plot(result, path, args):
    field = result["field"]
    count = len(result["persistent_ids"])
    n_reference = len(field["contours"][result["reference_round"]])
    fig, axes = panel_grid(len(field["images"]))
    temporary = path.with_suffix(".tmp.png")
    try:
        for ax, r in zip(axes.flat, sorted(field["images"])):
            contours = field["contours"][r]
            status = result["status_by_round"][r]
            kept = result["kept_indices_by_round"][r]
            layers = {}
            if not args.hide_rejected:
                for label, color in (("not_persistent", NONPERSISTENT_COLOR), ("no_passing_match", NO_PASS_COLOR),
                                     ("not_evaluable", NOT_EVALUABLE_COLOR)):
                    rejected = np.flatnonzero(status == label)
                    if len(rejected):
                        layers[label] = dict(contours=[contours[i] for i in rejected], color=color,
                                             alpha=1, fill_alpha=0.45, line_width=2.5,
                                             line_width_mode="absolute", render_scale=args.render_scale)
            rendered = mvt.render_contours(
                display_dapi(field["images"][r], args), [contours[i] for i in kept],
                colors=[result["colors"][i] for i in result["persistent_ids"]], markers=layers,
                fill_alpha=args.fill_alpha, line_width=args.line_width, line_width_mode="absolute",
                render_scale=args.render_scale, intensity_range=(0, 255))
            ax.imshow(rendered)
            title = f"Round {r}" + (" (reference)" if r == result["reference_round"] else "")
            title += f"\n{count} kept"
            if not args.hide_rejected:
                title += (f" · {np.count_nonzero(status == 'not_persistent')} nonpersistent"
                          f" · {np.count_nonzero(status == 'no_passing_match')} no pass")
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        mode = "Persistent contours only" if args.hide_rejected else "Rejected contours shown"
        fig.suptitle(f"{field['label']}\n{mode} · reference Round {result['reference_round']} · "
                     f"IoU > {result['iou_threshold']:g} in every round · {count}/{n_reference} IDs kept", fontsize=12)
        handles = [Patch(color=result["colors"][i], label=f"Persistent ID {i}") for i in result["persistent_ids"][:3]]
        if not args.hide_rejected:
            handles += [Patch(color=NONPERSISTENT_COLOR, label="Nonpersistent ID (passes here)"),
                        Patch(color=NO_PASS_COLOR, label="No passing IoU match here / unmatched"),
                        Patch(color=NOT_EVALUABLE_COLOR, label="ID touches an edge (excluded)")]
        fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.94))
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
        fig.savefig(temporary, dpi=args.dpi, facecolor="white")
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def assignment_record(result):
    field = result["field"]
    rounds = {}
    for r in sorted(field["images"]):
        ids, status = result["ids_by_round"][r], result["status_by_round"][r]
        kept = result["kept_indices_by_round"][r]
        rounds[r] = dict(image_path=field["paths"][r], contour_path=field["contour_paths"][r],
                         reference_ids=ids.tolist(), statuses=status.tolist(), is_edge=result["is_edge_by_round"][r].tolist(),
                         ious=[float(v) if np.isfinite(v) else None for v in result["ious_by_round"][r]],
                         kept_contour_indices=kept.tolist(), kept_ids=ids[kept].tolist(),
                         counts={label: int(np.count_nonzero(status == label))
                                 for label in ("persistent", "not_persistent", "no_passing_match", "not_evaluable")})
    return dict(patch_key="__".join(field["key"]), reference_round=result["reference_round"],
                iou_threshold=result["iou_threshold"], threshold_rule=">", shape=list(field["shape"]),
                reference_count=len(field["contours"][result["reference_round"]]),
                persistent_ids=result["persistent_ids"].tolist(), id_colors=result["colors"],
                error_colors=dict(not_persistent=NONPERSISTENT_COLOR, no_passing_match=NO_PASS_COLOR, not_evaluable=NOT_EVALUABLE_COLOR),
                missing_rounds_by_id=result["missing_rounds_by_id"], rounds=rounds,
                index_note="Contour indices are zero-based; IDs are one-based reference indices; ID 0 is unmatched. "
                           "Kept indices are sorted by ID. Reference/unmatched IoUs are null.")


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retention_statistics(total, kept):
    """Fractions are undefined for an empty denominator, represented by None."""
    if not 0 <= kept <= total:
        raise ValueError(f"Invalid retention counts: kept={kept}, original={total}")
    filtered = total - kept
    return dict(kept_count=kept, filtered_count=filtered,
                kept_proportion=kept / total if total else None,
                kept_percentage=100 * kept / total if total else None,
                filtered_proportion=filtered / total if total else None,
                filtered_percentage=100 * filtered / total if total else None)


def patch_columns(patch_key):
    tokens = patch_key.split("__")
    return {name: next((token for token in tokens if token.startswith(prefix)), "")
            for name, prefix in (("plate", "plate-"), ("replicate", "rep-"), ("well", "well-"),
                                 ("field", "field-"), ("cohort", "cohort-"), ("patch", "patch_"))}


def write_csv(path, columns, rows):
    temporary = path.with_suffix(".tmp.csv")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_statistics(output_dir, summary):
    """Export a consistent snapshot; only the parent process writes these files."""
    if summary['settings'].get('registration_based_filtering', False):
        return write_registration_statistics(output_dir, summary)
    settings = summary["settings"]
    round_numbers = settings.get("rounds", ROUNDS)
    reference_round = int(settings["reference_round"])
    iou_threshold = float(settings["iou_threshold"])
    binary_mode = settings.get("use_binary_mask_reference", False)
    filter_mode = "binary_mask_reference" if binary_mode else "hungarian"
    score_metric = "reference_coverage" if binary_mode else "iou"
    min_other_rounds = int(settings.get("min_other_rounds", 2)) if binary_mode else None
    run_status = summary.get("status", "unknown")
    field_rows, round_rows = [], []
    reference_total, kept_total = 0, 0

    def base_row(entry, status):
        return dict(**patch_columns(entry["patch_key"]), patch_key=entry["patch_key"],
                    reference_round=reference_round, iou_threshold=iou_threshold,
                    filter_mode=filter_mode, score_metric=score_metric, min_other_rounds=min_other_rounds,
                    status=status, run_status=run_status)

    for entry in sorted(summary.get("completed", []), key=lambda item: item["patch_key"]):
        row = base_row(entry, "complete")
        original, kept = entry["reference_count"], entry.get("kept_count", entry.get("persistent_count"))
        field_rows.append(dict(row, reference_count=original, **retention_statistics(original, kept)))
        reference_total += original
        kept_total += kept
        counts_by_round = entry.get("round_counts")
        if counts_by_round is None:
            # Older runs already saved enough information; no image reads needed.
            record = json.loads((output_dir / entry["assignments"]).read_text(encoding="utf-8"))
            record_kept = record["kept_reference_ids"] if binary_mode else record["persistent_ids"]
            if (record["patch_key"] != entry["patch_key"] or
                    record["reference_round"] != reference_round or
                    record["iou_threshold"] != iou_threshold or
                    record["reference_count"] != original or len(record_kept) != kept or
                    record.get("filter_mode", "hungarian") != filter_mode or
                    (binary_mode and record["min_other_rounds"] != min_other_rounds)):
                raise ValueError(f"Assignment data disagree with summary.json for {entry['patch_key']}")
            counts_by_round = {r: data["counts"] for r, data in record["rounds"].items()}
        counts_by_round = {int(r): counts for r, counts in counts_by_round.items()}
        if set(counts_by_round) != set(round_numbers):
            raise ValueError(f"Expected {len(round_numbers)} round counts for {entry['patch_key']}")
        for r in round_numbers:
            counts = counts_by_round[r]
            if binary_mode:
                n_pass, n_fail = counts["coverage_pass"], counts["coverage_fail"]
                if (counts["reference_count"] != original or counts["kept"] != kept or
                        counts["filtered"] != original - kept or n_pass + n_fail != original or
                        min(n_pass, n_fail, counts["detected_count"]) < 0):
                    raise ValueError(f"Inconsistent binary coverage counts in Round {r} for {entry['patch_key']}")
                round_rows.append(dict(row, round=r, is_reference=int(r == reference_round),
                                       total_count=original, **retention_statistics(original, kept),
                                       detected_count=counts["detected_count"], coverage_pass_count=n_pass,
                                       coverage_fail_count=n_fail, edge_count=counts.get("edge_count", 0),
                                       coverage_pass_proportion=n_pass / original if original else None,
                                       coverage_pass_percentage=100 * n_pass / original if original else None))
                continue
            n_kept = counts["persistent"]
            n_nonpersistent = counts["not_persistent"]
            n_no_pass = counts["no_passing_match"]
            n_unknown = counts.get("not_evaluable", 0)
            total = n_kept + n_nonpersistent + n_no_pass + n_unknown
            if (n_kept != kept or min(n_kept, n_nonpersistent, n_no_pass) < 0 or
                    (r == reference_round and (total != original or n_no_pass != 0))):
                raise ValueError(f"Inconsistent counts in Round {r} for {entry['patch_key']}")
            round_rows.append(dict(row, round=r, is_reference=int(r == reference_round),
                                   total_count=total, **retention_statistics(total, n_kept),
                                   nonpersistent_count=n_nonpersistent, no_passing_match_count=n_no_pass, not_evaluable_count=n_unknown))

    for entry in sorted(summary.get("failed", []), key=lambda item: item["patch_key"]):
        field_rows.append(dict(base_row(entry, "failed"), error=entry["error"]))
    incomplete = summary.get("discovery", {}).get("incomplete_patches", [])
    for entry in sorted(incomplete, key=lambda item: item["patch_key"]):
        field_rows.append(dict(base_row(entry, "skipped_incomplete"),
                               missing_rounds=";".join(map(str, entry["missing_rounds"]))))
    field_rows.sort(key=lambda row: row["patch_key"])
    identity_columns = ["plate", "replicate", "well", "field", "cohort", "patch", "reference_round", "iou_threshold",
                        "filter_mode", "score_metric", "min_other_rounds"]
    metric_columns = ["kept_count", "filtered_count", "kept_proportion", "kept_percentage",
                      "filtered_proportion", "filtered_percentage"]
    write_csv(output_dir / "field_summary.csv",
              identity_columns + ["reference_count"] + metric_columns +
              ["status", "run_status", "missing_rounds", "error", "patch_key"], field_rows)
    write_csv(output_dir / "round_summary.csv",
              identity_columns + ["round", "is_reference", "total_count"] + metric_columns +
              ["nonpersistent_count", "no_passing_match_count", "not_evaluable_count", "edge_count", "detected_count", "coverage_pass_count",
               "coverage_fail_count", "coverage_pass_proportion", "coverage_pass_percentage",
               "status", "run_status", "patch_key"], round_rows)
    completed_count, failed_count = len(summary.get("completed", [])), len(summary.get("failed", []))
    metadata = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), source_run_status=run_status,
        source_summary=str(output_dir / "summary.json"), reference_round=reference_round,
        iou_threshold=iou_threshold, selected_plots=summary.get("selected_plots"),
        filter_mode=filter_mode, score_metric=score_metric, min_other_rounds=min_other_rounds,
        threshold_rule=">", sample_size=settings.get("n_images"), seed=settings.get("seed"),
        completed_patches=completed_count, failed_patches=failed_count,
        pending_or_unprocessed_patches=max(0, summary.get("selected_plots", completed_count + failed_count) - completed_count - failed_count),
        skipped_incomplete_patches=len(incomplete), field_rows=len(field_rows), round_rows=len(round_rows),
        field_denominator="reference_count; one row per image patch, no aggregation or deduplication across patches",
        round_denominator=("reference_count in every panel; detected_count separately reports current-round instances"
                           if binary_mode else "total_count in that particular round before filtering"),
        kept_definition=(f"Reference nuclei with coverage > cutoff in at least {min_other_rounds} non-reference rounds"
                         if binary_mode else "Reference IDs with a passing Hungarian match in every round"),
        score_definition=("area(reference nucleus intersect round foreground) / area(reference nucleus)"
                          if binary_mode else "Intersection over union for each one-to-one contour match"),
        zero_denominator="Proportions and percentages are blank in CSV (null in JSON)",
        no_passing_match_definition=(None if binary_mode else
                                    "No passing assignment: low IoU, no counterpart, or one-to-one competition"),
        overall_reference_statistics=dict(reference_count=reference_total, **retention_statistics(reference_total, kept_total)),
    )
    if binary_mode:
        metadata["coverage_counts_definition"] = (
            "coverage_pass/fail count reference regions above/below the cutoff in that round alone; "
            "kept/filtered count the overall decision, identical across rounds. The reference round is not "
            "counted toward min_other_rounds. No DAPI intensity threshold is used.")
    write_json(output_dir / "statistics_metadata.json", metadata)
    print(f"Statistics: {completed_count} completed patches; {failed_count} failed; source run status={run_status}.", flush=True)
    print(f"  {output_dir / 'field_summary.csv'}", flush=True)
    print(f"  {output_dir / 'round_summary.csv'}", flush=True)
    return metadata


def process_patch(key, paths_by_round, args, timer=None):
    """One independent unit of work; no shared plot state or output filenames."""
    source = source_snapshot(*paths_by_round[args.reference_round])
    # The standalone CLI does not require the quantification package.
    timed = timer.call if timer is not None else lambda name, fn, *a, **k: fn(*a, **k)
    field = timed("qc_load_dapi", load_field, key, paths_by_round)
    if (len(source["source_line_numbers"]) != len(field["contours"][args.reference_round]) or
            sha256(paths_by_round[args.reference_round][1]) != source["reference_contour_sha256"]):
        raise ValueError("Reference contours changed while loading QC inputs")
    if args.registration_based_filtering:
        base = timed("qc_registration", registration_filter, field, args)
        name = args.registration_metric
        names = None if args.registration_compare_metrics else [name]
        comparisons = (timed("qc_scoring", compare_registration_metrics, base, args, names=names)
                       if args.registration_compare_metrics or name in registration_metric_specs(args) else {})
        result = comparisons.get(name, base)
        active_args = argparse.Namespace(**vars(args))
        active_args.registration_change_threshold = args.registration_active_threshold
        record = registration_assignment_record(result, active_args)
    elif args.use_binary_mask_reference:
        result = filter_binary_mask_reference(field, args.reference_round, args.iou_threshold, args.min_other_rounds)
        record = binary_assignment_record(result)
    else:
        result = filter_persistent_nuclei(field, args.reference_round, args.iou_threshold)
        record = assignment_record(result)
    record["reference_source"] = source
    stem = "__".join(key)
    plot_path = args.output_dir / "plots" / f"{stem}.png"
    assignment_path = args.output_dir / "assignments" / f"{stem}.json"
    extra_outputs = {}
    if args.registration_based_filtering:
        if not args.fast_mode:
            overlay_path = args.output_dir / "registration_overlays" / f"{stem}.png"
            map_path = (None if getattr(args, "skip_dense_maps", False) else
                        args.output_dir / "registration_maps" / f"{stem}.npz")
            score_path = args.output_dir / "registration_scores" / f"{stem}.png"
            timed("qc_rendering", save_registration_plots, result, plot_path, overlay_path, map_path, active_args)
            timed("qc_rendering", save_registration_score_plot, result, score_path, active_args)
            extra_outputs.update(registration_overlay=str(overlay_path.relative_to(args.output_dir)),
                                 registration_score_plot=str(score_path.relative_to(args.output_dir)))
            if map_path is not None:
                extra_outputs['registration_map'] = str(map_path.relative_to(args.output_dir))
            if getattr(args, "registration_alignment_panels", False):
                alignment_path = args.output_dir / "registration_alignment" / f"{stem}.png"
                alignment_path.parent.mkdir(exist_ok=True)
                timed("qc_rendering", save_registration_alignment_plot, result, alignment_path, active_args)
                extra_outputs['registration_alignment'] = str(alignment_path.relative_to(args.output_dir))
            if getattr(args, "myotube_contour_dir", None) is not None:
                myotube_path = args.output_dir / "myotube_overlays" / f"{stem}.png"
                myotube_path.parent.mkdir(exist_ok=True)
                _, translation, _, _ = registration_dependencies(args.tools_dir)
                overlay = timed("qc_myotube_overlays", save_myotube_overlay, result, myotube_path,
                                args, mvt, plt, np, translation.apply_translation)
                overlay['contour_sha256'] = sha256(overlay['contour_path'])
                record['myotube_overlay'] = overlay
                extra_outputs['myotube_overlay'] = str(myotube_path.relative_to(args.output_dir))
        extra_outputs['metric_comparisons'] = (timed("qc_comparison_exports", save_metric_comparisons, base, stem, args, results=comparisons)
                                               if args.registration_compare_metrics else {})
    elif not args.fast_mode:
        if args.use_binary_mask_reference:
            save_binary_plot(result, plot_path, args)
        else:
            save_plot(result, plot_path, args)
    write_json(assignment_path, record)
    retained = (dict(kept_count=len(record["kept_reference_ids"])) if (args.use_binary_mask_reference or args.registration_based_filtering)
                else dict(persistent_count=len(record["persistent_ids"])))
    return dict(patch_key=stem, label=field["label"], worker_pid=os.getpid(), **retained, **extra_outputs,
                reference_count=record["reference_count"], plot=None if args.fast_mode else str(plot_path.relative_to(args.output_dir)),
                assignments=str(assignment_path.relative_to(args.output_dir)),
                round_counts={r: data["counts"] for r, data in record["rounds"].items()})


def completed_patches(selected, indices, args):
    """Yield outcomes as they finish, with at most --workers patches in flight."""
    if args.workers == 1:
        for key in selected:
            try:
                result = process_patch(key, {r: indices[r][key] for r in args.rounds}, args)
            except Exception as exc:
                yield key, None, f"{type(exc).__name__}: {exc}"
            else:
                yield key, result, None
        return
    # spawn avoids inheriting Matplotlib or native-library thread state after fork.
    pool = ProcessPoolExecutor(max_workers=min(args.workers, len(selected)),
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=initialize_worker, initargs=(args.tools_dir,))
    pending, remaining = {}, iter(selected)

    def submit_next():
        key = next(remaining, None)
        if key is not None:
            future = pool.submit(process_patch, key, {r: indices[r][key] for r in args.rounds}, args)
            pending[future] = key

    try:
        for _ in range(min(args.workers, len(selected))):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                key = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    yield key, None, f"{type(exc).__name__}: {exc}"
                else:
                    yield key, result, None
                submit_next()
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    args = parse_args(argv)
    if args.summary_only:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        write_statistics(args.output_dir, summary)
        export_myonis_4i(args.output_dir, summary)
        return 0
    rule = (f"reference coverage > {args.iou_threshold} in at least {args.min_other_rounds}/{len(args.rounds)-1} other rounds"
            if args.use_binary_mask_reference else f"strict IoU > {args.iou_threshold} in every round (Hungarian)")
    if args.registration_based_filtering:
        prepare_registration_settings(args)
        rule = (f"{args.registration_metric}: area-mean {args.registration_aggregation}-map change <= {args.registration_active_threshold} "
                f"across {len(args.filter_rounds)-1} selected other rounds")
        print(f"Re-registration config: {args.registration_config}", flush=True)
        print("IoU/min-other-rounds do not apply in registration mode.", flush=True)
        print(f"Registration rounds: {args.rounds}; nuclear filter rounds: {args.filter_rounds}; "
              f"excluded from filtering: {sorted(set(args.rounds)-set(args.filter_rounds))}", flush=True)
    print(f"Run: {args.run_dir}\nReference: Round {args.reference_round}; {rule}\n"
          f"Output: {args.output_dir}", flush=True)
    if not args.dry_run:
        initialize_worker(args.tools_dir, verbose=True)
    indices, common, discovery = discover_patches(
        args.run_dir, reference_only=args.reference_round if args.registration_based_filtering else None,
        reference_round=args.reference_round, reference_contour_dir=args.reference_contour_dir, rounds=args.rounds)
    selected, n_fields = select_patches(common, args.patch_mode, args.seed, args.max_plots, args.n_images)
    print(f"{len(common)} complete patches across {n_fields} source fields. "
          f"Selected {len(selected)} patches ({args.patch_mode}); up to {min(args.workers, len(selected))} in parallel.", flush=True)
    if args.fast_mode:
        print("Fast mode: skipping all figures and dense pixel-map files; keeping assignments, QC statistics and myonis_4i metadata.", flush=True)
    if args.n_images is not None:
        print(f"Random sample: {args.n_images} patches without replacement; seed={args.seed}.", flush=True)
    if args.dry_run:
        return 0
    if args.myotube_contour_dir is not None and not args.fast_mode:
        if not args.myotube_contour_dir.is_dir():
            raise FileNotFoundError(args.myotube_contour_dir)
        reference_image = indices[args.reference_round][selected[0]][0]
        args.myotube_channels = discover_channels(args.run_dir, args.rounds, reference_image)
        if not any(channel['acquisition'] == args.myotube_acquisition for channel in args.myotube_channels):
            raise ValueError(f"Myotube source acquisition is missing: {args.myotube_acquisition}")
        print(f"Myotube overlays: {len(args.myotube_channels)} marker acquisitions across all registered rounds", flush=True)
    output_names = ["assignments"]
    if not args.fast_mode:
        output_names.append("plots")
        if args.registration_based_filtering:
            output_names += ["registration_overlays", "registration_scores"]
            if not args.skip_dense_maps:
                output_names.append("registration_maps")
            if args.registration_alignment_panels:
                output_names.append("registration_alignment")
    for name in output_names:
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = dict(started_at=datetime.now(timezone.utc).isoformat(), status="running", settings=settings,
                   library_path=str(mvt.__file__), discovery=discovery, selected_plots=len(selected),
                   selected_patch_keys=["__".join(key) for key in selected], completed=[], failed=[])
    # Preserve reference identities even when later image loading/registration
    # fails, so exported flags can remain explicitly not_assessed.
    summary["reference_sources"] = {}
    source_keys = set(selected) | {tuple(item["patch_key"].split("__")) for item in discovery["incomplete_patches"]}
    for key in sorted(source_keys):
        paths = indices[args.reference_round].get(key)
        if paths and paths[1] is not None:
            try:
                summary["reference_sources"]["__".join(key)] = source_snapshot(*paths)
            except (OSError, UnicodeError):
                pass  # Patch failure/missing input remains visible in its status.
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    started = time.monotonic()
    current = None
    try:
        with closing(completed_patches(selected, indices, args)) as outcomes:
            for number, (key, record, error) in enumerate(outcomes, 1):
                current = "__".join(key)
                if error is not None:
                    summary["failed"].append(dict(patch_key=current, error=error))
                    print(f"[{number}/{len(selected)}] FAILED {field_label(key)}: {error}", file=sys.stderr, flush=True)
                else:
                    summary["completed"].append(record)
                    kept = record.get("kept_count", record.get("persistent_count"))
                    print(f"[{number}/{len(selected)}] {record['label']}: kept {kept}/"
                          f"{record['reference_count']} reference IDs; saved {Path(record['plot'] or record['assignments']).name} "
                          f"({time.monotonic() - started:.0f}s elapsed)", flush=True)
                write_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       failed_patch=current, error=str(exc) or type(exc).__name__)
        raise
    else:
        summary["status"] = "complete_with_errors" if summary["failed"] else "complete"
    finally:
        summary["completed"].sort(key=lambda record: record["patch_key"])
        summary["failed"].sort(key=lambda record: record["patch_key"])
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
        write_statistics(args.output_dir, summary)
        export_myonis_4i(args.output_dir, summary)
    print(f"Finished {len(summary['completed'])} patches; {len(summary['failed'])} failed. Summary: {summary_path}", flush=True)
    print(f"Quantification QC metadata: {args.output_dir / 'myonis_4i'}", flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed outputs are saved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
