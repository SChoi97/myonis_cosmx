# Field-registered 4i seam scoring

This directory keeps **patch production** and **seam scoring** separate. It calls
the tested preprocessing pipeline in `../preprocessing/` for illumination
correction, whole-well field-level registration to measurement 1, quantitative
512-pixel patch generation, and display clipping. No IC, registration, or patching
algorithm is duplicated here.

## Patch cache already prepared

The completed field-level production run
`../preprocessing/results/runs/4i_production_20260823_200037_f6ecb24` already
uses exactly the 29 IC choices and clipping limits in the supplied table. Its
12 DAPI IC choices are Harmony-like, matching the existing approved workbook.
`marker_settings.csv` records the supplied 29 entries explicitly. The cache
validator checked all 40 acquisition choices, clipping values, field-level
registration, and all 120 well manifests; it recorded 631,840 quantitative
patch files in `results/cache_manifest.json`. This is a pointer and provenance
record, not a second copy of the images.

To validate another completed field-level run, run on Nemo:

```bash
PYTHON=/nemo/lab/tedescos/home/users/chois1/.conda/envs/illumination_correction_env/bin/python
HERE=/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/registration_beam_scoring
"$PYTHON" "$HERE/cache_patches.py" --run-dir /path/to/completed/field_level_run --output "$HERE/results/cache_manifest.json"
```

To deliberately start a **new** run with the same existing preprocessing
workers, execute `bash submit_field_patches.sh field_level_test` or
`bash submit_field_patches.sh field_level_production` on Nemo. This submits the
existing controller with field-level mode and prints its new run directory in
the submission log. Once it completes, validate it using `cache_patches.py`.
The current matching run makes a new production submission unnecessary.

## Scores

`score_seams.py` reads quantitative float32 patches and each well's
`registration_shifts.csv`. For each acquisition, the former source-field
boundaries are `1080*k + registration_shift_x/y` in reference coordinates;
the patch's `reference_x/y` gives the local line position. A line too close
to a patch edge is recorded as `insufficient_context` rather than assigned a
false low score.

The independent scores are:

* `duplicate_ncc`: strongest two-way-demeaned cross correlation between the
  strips on opposite sides of the old seam, searching widths 8–32 pixels and
  along-seam offsets ±3 pixels. `repeat_width_px` and
  `along_seam_offset_px` record the winning geometry.
* `duplicate_ncc_excess`: the seam NCC minus median NCC at nearby control cuts
  in the same patch. A high NCC that is also unusually high for this image is
  stronger evidence of repeated tissue.
* `matched_window_fraction`: fraction of 64-pixel seam windows whose strip
  correlation is at least 0.6. This distinguishes a local repeat from a repeat
  spanning much of the patch. Low-texture windows are not called matches.
* `matched_bright_runs` and `matched_bright_pixel_fraction`: rough burden of
  coincident bright DAPI signal in the two matched strips. Runs are **candidate
  repeated fragments, not a validated nucleus count**.
* `duplicate_foreground_fraction`: one copy of the matched bright pixels as a
  fraction of all bright pixels in the patch. This is a rough estimate of
  duplicated bright area, not an object-count error rate.
* `foreground_burden_fraction`: fraction of all bright patch pixels found in
  the strip around this seam. It measures how much measured signal is exposed
  to the join, regardless of whether it is duplicated.
* `jump_ratio`: 90th-percentile pixel jump at the seam relative to nearby
  control cuts. The denominator has a 1%-of-dynamic-range floor to avoid
  instability in empty fields. It is a brightness/misalignment comparator, not
  specific to duplicate objects.

The 14–20-pixel physical overlaps mean high NCC can be common even in patches
that look acceptable overall. Interpret it jointly with burden and visual
examples. Scores have **no automatic rejection threshold**. `seam_scores.csv` has one row
per old seam per patch per scored acquisition. `image_scores.csv` has one row
for every input patch image, with the maxima across its seams and an explicit
status when it has no source seam or no evaluable seam. `patch_scores.csv` has one row
per patch ID, including patches with no evaluable seam, and maxima across the
selected acquisitions. These tables are rerunnable without regenerating patches.
The score metadata hashes the source manifest, score code, inputs, and
measurement selection; identical results are reused automatically.

Pilot with four DAPI measurements and 20 patches in one well:

```bash
"$PYTHON" "$HERE/score_seams.py" \
  --well sc_rep5-6__r04_c02 \
  --measurements 1,6,9,12 \
  --max-patches 20 \
  --output-dir "$HERE/results/pilot_scores"
"$PYTHON" "$HERE/visualize_scores.py" \
  --score-dir "$HERE/results/pilot_scores" \
  --well sc_rep5-6__r04_c02 \
  --output-dir "$HERE/results/pilot_visualizations"
```

Full 12-DAPI scoring across all wells on Nemo:

```bash
bash "$HERE/submit_scores.sh"
```

This older launcher submits one scorer per well, then a gallery job.
Use `submit_ncc_gallery.sh` below for the bounded worker pool and DAPI round
comparisons. Direct `score_seams.py` calls select DAPI in all 12 measurements
unless an explicit acquisition selection is supplied.
The full-patch and closeup galleries rank low, median, and high examples for
each score. Every panel prints **all** scores beside the image and overlays the
old seam in cyan. Closeups center on the brightest section of that seam.
`illustrative_cutoff_sweep.csv` reports how many patch IDs would exceed several
candidate cutoffs; these are exploratory, not exclusion decisions. The
denominator is all scored patch IDs, while `patches_with_score` identifies
patches where the metric was evaluable. `illustrative_cutoff_sweep.png` plots
the same fractions.

## NCC excess galleries and patch retention curves

The filtering workflow scores **DAPI only**, in all 12 rounds. The default run
selects **500 spatial patch locations**, randomly and as equally as possible
across plates, then wells within each plate, and scores all 12 DAPI images at
each location: **6,000 TIFF images**. `SAMPLE_PATCHES=1000` means 1,000 locations
and **12,000 DAPI images**. The same locations across rounds make score
correlation and rejection-overlap comparisons possible. A recorded seed makes
the selection reproducible. Sampling is based on geometry and identity,
before computing any seam score. Marker-channel intensities do not contribute
to these filtering statistics.

The default cap is **12 child jobs total: 11 scoring workers plus 1 gallery**.
Each worker processes a group of wells sequentially. This caps the total number
of array tasks, rather than merely restricting concurrency of 120 tasks.
The launcher can be run with `bash` from a Nemo login node, or submitted with
`sbatch`; its own allocation requests 1 CPU and 4 GB. Workers explicitly use
the `ncpu` partition (override using `SCORING_PARTITION`).

Activate your Python environment before launching or submitting the script.
The launcher resolves `python` from that environment and prints its path in
the parent log. Workers inherit the submission environment and use that same
Python executable; they do not activate Conda or retrieve a fresh login
environment. `FOURI_PYTHON=/path/to/python` remains an optional override.
Run settings are passed as script arguments. The child submissions use plain
`--export=ALL` to inherit the current environment, with no custom variable list;
the launcher clears an inherited `SBATCH_GET_USER_ENV` setting before submitting
children. This avoids the implicit login-environment retrieval caused by
`--export=ALL,VAR=...` (see the [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html)).

```bash
bash "$HERE/submit_ncc_gallery.sh"
```

Or, from the same activated shell:

```bash
sbatch "$HERE/submit_ncc_gallery.sh"
```

The fix applies to newly submitted jobs. An existing held array keeps its original
submission environment settings. Submitting the same sample and seed again reuses
matching completed well scores.

Configure the sample and job cap independently:

```bash
SAMPLE_PATCHES=1000 SAMPLE_SEED=123 MAX_JOBS=12 sbatch "$HERE/submit_ncc_gallery.sh"
```

To score DAPI for every spatial patch in the complete dataset (189,552 DAPI
TIFF images across 15,796 locations):

```bash
SAMPLE_PATCHES=0 MAX_JOBS=12 bash "$HERE/submit_ncc_gallery.sh"
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `SAMPLE_PATCHES` | `500` | Spatial locations sampled; each contributes 12 DAPI images; `0` selects all locations |
| `SAMPLE_SEED` | `42` | Reproducible random selection |
| `MAX_JOBS` | `12` | Total child jobs, including the gallery; minimum 2 |
| `MAX_CONCURRENT` | Number of scoring workers | Optional further limit on simultaneous scoring workers |
| `NCC_EXAMPLES_PER_BIN` | `5` | Number of visual examples; independent of the scored sample |
| `ROUND_COMPARISON_CUTOFFS` | `0.2:0.4:0.6:0.8` | Illustrative NCC excess cutoffs for rejection-overlap matrices and shared patch flags |
| `DRY_RUN` | `0` | `1` writes the plan and prints counts without submitting jobs |

The launcher writes `scoring_plan_<run>.json` and its `.distribution.csv`
inside the score directory. They record every selected patch/image, the plate
and well allocation, expected counts, and worker assignments. The gallery
checks every well's metric tables against this plan before plotting.
Sample and full runs use separate output directories. Matching per-well
metric tables are reused when the same sample is submitted again.

The parent log (`slurm-<job>.out`), scoring logs
(`score_<array-job>_<task>.out`), and gallery log
(`ncc_gallery_<job>.out`) are written directly in this directory.
Their `.err` files are also written here.
`tqdm` progress bars for wells, image scoring, and gallery rendering are written
to `.out`, with unbuffered Python output. If `tqdm` is unavailable in the
environment, periodic progress counts are printed instead.

The default samples five seam examples per NCC-excess bin, with edges at
0, 0.2, 0.4, 0.6, 0.8, and 1.0, including open-ended bins below 0 and at or
above 1.0. Set the number and bin edges when submitting:

```bash
NCC_EXAMPLES_PER_BIN=20 NCC_BIN_EDGES='0:0.1:0.2:0.3:0.4:0.5:0.6:0.8:1.0' \
  bash "$HERE/submit_ncc_gallery.sh"
```

For the default sample, open
`results/ncc_level_gallery_dapi_sample_patch500_seed42_v3/index.html`.
The HTML page shows all retention plots and the annotated seam examples.
`score_<job>_<task>.out` is a worker progress log; the separate
`ncc_gallery_<job>.out` reports gallery creation, and its `.err` records any
plotting failure. Completed scoring does not imply the gallery job succeeded.
The gallery reads the original display limits from the completed production
run's `manifests/clipping/*.json`, so this run does not require `openpyxl` for
plotting. Older runs without those JSON files use the saved Excel workbook.
Metric tables are in `results/scores_dapi_sample_patch500_seed42_v3/wells/<well>/`.
Each example shows its
full patch and seam closeup with scores and cyan seam position. Each bin samples
distinct patch IDs using a recorded random seed, so repeated rounds of one
patch do not fill the same bin. `sampled_seams.csv` gives the exact examples;
`bin_summary.csv` gives seam and unique-patch counts in each bin.

`image_retention_curve.png` and CSV count each individual scored TIFF image.
`patch_retention_curve.png` and `retention_curve.csv` count spatial patch IDs.
For each spatial patch,
the rule uses the **maximum** evaluable NCC excess across all selected seams
and acquisitions. A patch is rejected when that maximum is greater than or
equal to the cutoff. Patches without an evaluable seam are included in the
retained total, and their number is reported separately. The companion
`retention_by_acquisition*.png` and CSV show one curve per acquisition, using
the number of images actually scored in that acquisition as its denominator.
In sample mode, all retention curves describe the selected sample. The primary
launcher always uses matched spatial locations and all 12 DAPI rounds.

### Comparing rounds and discarding the same patch everywhere

The shared rule is **reject the spatial patch in all rounds when any DAPI round
has a score at or above the cutoff**. Equivalently, compare the maximum DAPI
NCC excess across all seams and all 12 rounds with that cutoff. Transfer the
same `(well_slug, patch_id)` rejection to all marker channels. The scripts
report candidate decisions without deleting image files or choosing a cutoff.

* `dapi_round_score_correlations.png`: Pearson and Spearman heatmaps for the
  same spatial patches across rounds. Each cell gives its paired sample count.
  Correlations use finite scores in both rounds, with average ranks for ties.
  Missing or unevaluable scores are excluded. Fewer than three pairs or
  constant scores produce `NA`, not zero correlation.
* `dapi_round_rejection_overlap.png`: at each illustrative cutoff, rows are
  the round already flagged and columns are the round checked. A cell reports
  the fraction also flagged in that target round, with the number of eligible
  flagged source patches. No source seam is unflagged; missing images and
  unevaluable seams are excluded. The matrix is directional and can be
  asymmetric. Pink lines separate rounds 1–8 and 9–12 in both figures.
* `dapi_shared_filter_retention.png`: compares independent per-image rejection
  with rejecting all rounds of a patch if any DAPI round is flagged. Its
  `dapi_shared_filter_retention.csv` reports spatial patches kept/discarded,
  DAPI images kept/discarded, and the additional image loss for consistency.
  For 1,000 complete locations, discarding one location removes 12 DAPI images.
* `dapi_shared_patch_filter_sweep.csv`: one flag per well + patch ID + candidate
  cutoff, with the DAPI rounds that triggered it. `dapi_patch_round_scores.csv`
  contains all round scores/statuses and the shared maximum. Patches without
  an evaluable DAPI score are reported separately and remain retained by this
  exploratory score rule; they are not verified artifact-free.

All figures and CSV links are included in `index.html`. Adjust the candidate
cutoffs independently of scoring or image-example selection:

```bash
ROUND_COMPARISON_CUTOFFS=0.3:0.5:0.7 SAMPLE_PATCHES=1000 sbatch "$HERE/submit_ncc_gallery.sh"
```

To regenerate just these comparisons from saved tables, without loading TIFFs:

```bash
python "$HERE/compare_dapi_rounds.py" \
  --score-dir "$HERE/results/scores_dapi_sample_patch1000_seed123_v3" \
  --output-dir "$HERE/results/ncc_level_gallery_dapi_sample_patch1000_seed123_v3" \
  --cutoffs 0.3,0.5,0.7
```

To regenerate visual examples with a different sample size after scoring,
run only the gallery script. This reads the complete metric tables and opens
TIFF files only for sampled examples:

```bash
"$PYTHON" "$HERE/gallery_ncc_levels.py" \
  --score-dir "$HERE/results/scores_dapi_sample_patch500_seed42_v3" \
  --scoring-plan /path/to/the/saved/scoring_plan_RUN.json \
  --examples-per-bin 20 \
  --output-dir "$HERE/results/ncc_level_gallery_dapi_sample_patch500_seed42_v3"
```

For round-specific examples and retention counts, add repeated DAPI
`--acquisition-id` flags to that command. Round comparisons then use the selected
rounds; the complete 12-round run is needed for the shared 12-round filter.

For a quick one-well run against existing scores, run:

```bash
"$PYTHON" "$HERE/gallery_ncc_levels.py" \
  --score-dir "$HERE/results/pilot_scores" \
  --well sc_rep5-6__r04_c02 \
  --examples-per-bin 5 \
  --output-dir "$HERE/results/pilot_ncc_gallery"
```

## Extending scoring

Add a new metric in `score_join()` in `score_seams.py`, include its patch-level
aggregation in `summarize()`, then add it to `METRICS` in
`visualize_scores.py`. Re-run the scoring array. The existing quantitative
patches and their IC/registration metadata remain unchanged.
