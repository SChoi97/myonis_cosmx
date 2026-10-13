# Paired-plate 4i illumination-correction evaluation

`illumination_correction_eval.py` evaluates Original, BaSiC, and Harmony-like
images for the August 2026 4i experiment. The built-in list contains all 12
measurement-specific DAPI acquisitions plus the 28 non-DAPI markers on both
`sc_rep5-6` and `sc_rep7-8`.
Measurement 9 Alexa 488 (imaged 12 August 2026) is deliberately excluded due to
its reported low quality; measurement 9 Alexa 647 (`p-4EBP1`) remains included.

## Shared sampling guarantee

Before any model is fitted, the program indexes all 80 channel directories and
intersects their filename coordinates separately for plate slot 1 and plate
slot 2. It then makes one seeded sampling plan that every marker reuses:

- `--training-field-count 1024` selects 1024 identical `(plate, row, column,
  field)` coordinates, balanced as 512 fields from each plate.
- `--visualization-well-count 30` selects 30 identical complete wells, balanced
  as 15 wells from each plate. All fields 1–36 from those wells are evaluated.
- A single BaSiC model and a single Harmony-like model are fitted per marker
  using that marker's images at the shared coordinates from both plates.
- Each zoom evaluates all 60 internal borders in its 6 × 6 mosaic and displays
  the border with the largest method disparity.

The exact coordinates are saved to `shared_sampling_manifest.csv`. If there
are not enough coordinates shared by every marker, the program stops rather
than silently changing the sample for one marker.

## Running on the cluster

Edit the numeric controls near the top of `illumination_correction_eval.sh`,
then submit it:

```bash
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/illumination_correction
sbatch illumination_correction_eval.sh
```

The primary controls in the shell script are:

```bash
MAX_CONCURRENT_JOBS=8
TRAINING_FIELD_COUNT=1024
VISUALIZATION_WELL_COUNT=30
RANDOM_SEED=20260820
```

The submitted controller first creates the shared sampling manifest once. It
then launches one SLURM-array task per marker with `%MAX_CONCURRENT_JOBS`
throttling, so at most eight markers run simultaneously by default. As slots
finish, Slurm starts the remaining markers. A final dependent job combines all
validated results and writes `marker_completion_status.csv`. Per-marker logs
are named `illumination_marker_<array-job-id>_<marker-index>.out/.err`.

BaSiC now uses the safer non-time-series defaults: darkfield estimation is off
and intensity sorting is on. The flatfield and a real transformed image sample
must both be finite and nonzero before processing continues. Invalid fits stop
that marker rather than producing NaN score tables. `--basic-darkfield` remains
available as an opt-in and automatically retries without darkfield if its fit
fails validation.

The 40 paired paths and biological labels are defined in
`DEFAULT_MARKER_CHANNELS` and `DEFAULT_DATA_ROOT` near the top of
`illumination_correction_eval.py`. They can also be replaced at runtime with a
TSV/CSV or repeated CLI arguments:

```bash
python illumination_correction_eval.py \
  --marker-config marker_paths.example.tsv \
  --training-field-count 600 \
  --visualization-well-count 30 \
  --output-dir results/my_evaluation
```

```bash
python illumination_correction_eval.py \
  --marker /path/to/rep5-6/channel /path/to/rep7-8/channel \
  --marker /path/to/next_rep5-6/channel /path/to/next_rep7-8/channel \
  --training-field-count 600 \
  --visualization-well-count 30
```

Configuration files use the columns `name`, `display_label`, `plate_1_path`,
and `plate_2_path`. `name` may be blank to infer a folder name such as
`measurement_1_channel_2_alexa_488`.

## Output structure

```text
results/paired_plate_evaluation/
├── marker_inputs.csv
├── shared_sampling_manifest.csv
├── all_markers_average_metrics.csv
├── marker_completion_status.csv
└── measurement_1_channel_2_alexa_488/
    ├── metrics/
    │   ├── marker_metadata.csv        # includes biological name: eMYHC
    │   ├── per_field_metrics.csv
    │   ├── per_well_metrics.csv
    │   ├── average_metrics.csv
    │   ├── training_image_paths.csv
    │   ├── basic_model_diagnostics.json
    │   ├── harmony_model_diagnostics.json
    │   └── harmony_profiles.npz
    └── visualisation/
        ├── full_6x6/   # 30 three-panel full-well figures by default
        ├── zoom_2x2/   # 30 three-panel top-right 2 × 2 field figures
        └── zoom/       # 30 maximum-disparity edge figures by default
```

`zoom_2x2` shows fields 6 and 7 above fields 9 and 8 in the verified acquisition
layout. It uses the same figure size, full-well intensity limits, metric titles,
display downsampling, and DPI as `full_6x6`, making individual nuclei and cells
larger in each panel. These figures are generated automatically for every
visualization well when running `illumination_correction_eval.sh` or the Python
entry point.

`per_field_metrics.csv` contains every per-field EVEN edge-energy value.
`per_well_metrics.csv` contains aggregate edge energy, field summary values,
`P+`, `P-`, `P±`, x/y-axis prominence, the selected boundary discontinuity for
each method, and seam metadata. `average_metrics.csv` has only three rows—one
per method—and reports the mean and sample standard deviation of every metric
across the selected wells, plus the per-field mean and standard deviation.
Harmony split-half profile errors are retained in the Harmony-like row.
The biological `display_label` is included in the root input manifest, marker
metadata, training paths, every metric table, combined averages, console logs,
and figure titles.

All three metric tables are rewritten after every completed well so partial
results survive a later cluster-job failure.
