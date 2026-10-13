# Production 4i preprocessing

This package converts the two August 2026 Phenix Plus plates into aligned,
quantitative image patches and separate clipped PNG derivatives. The production
order is fixed:

1. evaluate and approve illumination correction;
2. run one corrected real-data smoke well on each plate;
3. apply the approved BaSiC or Harmony-like profile;
4. register corrected DAPI using the selected field-level or patch-level mode;
5. crop every acquisition to the same valid region and write quantitative
   float32 OME-TIFF patches;
6. apply the workbook display limits and write uint8 PNGs;
7. aggregate a browsable HTML report, a PDF report, and machine-readable CSV and
   JSON provenance.

PNG files are derived display/model inputs. Do not use them for intensity
quantification.

## 1. Evaluate all 40 acquisitions

The illumination evaluation now includes all 12 DAPI acquisitions and the 28
marker acquisitions. Measurement 9 Alexa 488 remains excluded, while
measurement 9 DAPI and p-4EBP1 remain included.

```bash
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/illumination_correction
sbatch illumination_correction_eval.sh
```

The aggregation job writes model artifacts and metrics used by the preprocessing
handoff. Review the compact workbook supplied with this pipeline:

```text
outputs/illumination_method_selection/illumination_method_choices.xlsx
```

`Marker Selections` contains exactly the 28 marker acquisitions. `DAPI Selections`
contains the 12 measurement-specific DAPI acquisitions required for
registration. Fill the yellow `chosen_method` cells with exactly `BaSiC` or
`Harmony-like`; the status changes to `approved` automatically. You may add a
free-text review note, but keep acquisition identity columns unchanged.

`submit_preprocessing.sh` points to this workbook through
`METHOD_SELECTION_WORKBOOK` near the top of the file. Before any Slurm arrays
are submitted, the controller validates all 40 choices and combines them with
the evaluated model paths and hashes in a generated detailed selection CSV.
The pipeline will not refit models and will not start if a workbook choice is
blank or invalid. Initialization also reloads and hash-validates every selected
model. For BaSiCPy 2.0 models it safely handles the upstream
`smoothness_darkfield: null` serialization bug while preserving the saved
flatfield, darkfield, and baseline arrays exactly; the load mode is recorded in
the run manifest and worker manifests.

## 2. Fill the clipping workbook

Edit [config/clip_limits.xlsx](config/clip_limits.xlsx) only when custom display
clipping is wanted. When both cells are blank, the pipeline uses the full
16-bit range `0-65535` (no custom clipping) and records
`limits_source=default_full_16bit`. Otherwise, enter both a finite `min_16bit`
and `max_16bit`, with max greater than min. Filling only one limit remains an
error. Keep the acquisition identity columns unchanged; the pipeline validates
all 40 rows before creating a run.

## 2a. Optional low-storage cleanup

The corrected FOV stores and registered common-crop stores are reproducible
intermediates. To remove them safely after their consumers finish, set these
two values in `config/pipeline_config.json`:

```json
"storage": {
  "cleanup_corrected_after_registration": true,
  "cleanup_registered_after_patching": true
}
```

Both are enabled by default in the supplied configuration. With cleanup
enabled, a stage-gated Slurm job verifies
all well manifests before deleting only the matching direct child of the
timestamped run directory:

- `corrected/` is removed after every registration task succeeds;
- `registered_crops/` is removed after every patch task succeeds.

The quantitative float32 OME-TIFF patches, PNGs, QC figures, manifests, reports,
model artifacts, and raw data are retained. Cleanup reduces retained storage but
also removes the ability to resume registration or patch generation directly;
rerunning those stages requires regenerating their deleted prerequisites.
Keeping the quantitative patches is recommended because they are the final
unclipped analysis inputs.

## 3. Select registration and run scope

Set `RUN_MODE` near the top of `submit_preprocessing.sh` to one of:

```bash
RUN_MODE=field_level_test
RUN_MODE=field_level_production
RUN_MODE=patch_level_test
RUN_MODE=patch_level_production
```

`test` and `production` remain accepted as backward-compatible aliases for
`field_level_test` and `field_level_production` respectively.

- `field_level_*` is the original workflow. It assembles the 36 corrected FOVs
  into one 6480 x 6480 image, registers every measurement to measurement 1,
  finds one common whole-well crop, and cuts that crop into 512 patches.
- `patch_level_*` never assembles the 6 x 6 mosaic for production output. Each
  original 1080 x 1080 Phenix field is registered independently in two
  microscope-stable cohorts: measurements 1-8 use measurement 1 DAPI, while
  measurements 9-12 use measurement 9 DAPI. Each field/cohort receives its own
  maximal common valid crop, center-trimmed to fit as many 512 x 512 patches as
  possible.

Patch-level coordinates are shared only within their cohort; they are not
object-aligned across the maintenance boundary between measurements 8 and 9.
Patch IDs and filenames include both cohort and source field to prevent
accidental cross-cohort matching. Manifests retain the cohort reference,
source field, field-local and nominal mosaic coordinates, and independent DAPI
shift for every output.

## 4. Run the four-well end-to-end test

Choose `RUN_MODE=field_level_test` or `RUN_MODE=patch_level_test`. After the illumination
workbook and clipping workbook are complete, submit directly from this folder:

```bash
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/preprocessing
sbatch submit_preprocessing.sh
```

The test uses four physical wells total—two per plate, covering repeats 5–8:

```text
sc_rep5-6 r04_c02   sc_rep5-6 r04_c07
sc_rep7-8 r04_c02   sc_rep7-8 r04_c07
```

It runs the genuine pipeline for all 40 acquisitions and writes all corrected
stores, registration figures, common crops, quantitative patches, PNGs,
manifests, HTML/PDF reports, and provenance snapshots under
`results/test_runs/<run-id>/`. Only the selected spatial wells are shortened;
algorithms and output formats are identical to production. The test graph is:

```text
2 plate smoke-test tasks
  -> 40 illumination tasks (four wells across both plates)
  -> 4 plate/well registration tasks
  -> corrected-store cleanup (enabled or recorded as skipped)
  -> 4 plate/well patch tasks
  -> registered-crop cleanup (enabled or recorded as skipped)
  -> 40 acquisition PNG tasks
  -> one afterany report task
```

## 5. Submit the full production pipeline

After inspecting the test report, change only this line near the top of
`submit_preprocessing.sh`:

```bash
RUN_MODE=field_level_production
```

Use `RUN_MODE=patch_level_production` for the independent-field, two-cohort
workflow.

Then submit the same command. A production controller creates a new immutable
run under `results/runs/<run-id>/` and launches:

```text
2 plate smoke-test tasks
  -> 40 illumination tasks
  -> 120 plate/well registration tasks
  -> corrected-store cleanup (enabled or recorded as skipped)
  -> 120 plate/well patch tasks
  -> registered-crop cleanup (enabled or recorded as skipped)
  -> 40 acquisition PNG tasks
  -> one afterany report task
```

In both modes the report task runs even when an earlier task fails, so a partial
run still has a readable record. Test and production run IDs include their mode
and the test report displays the exact selected coordinates.

`reportlab` is included in `environment.yml` for PDF generation. If it is
missing from an existing environment, reporting still exits successfully and
writes the HTML report and all CSV/JSON manifests, recording the PDF as skipped.

The smoke-test coordinate is configured with `smoke_row` and `smoke_column` in
`config/pipeline_config.json` (default `r04_c02`). Each of the two smoke tasks
applies all 40 approved correction models to that plate's real 36-field well,
executes the selected registration level for all 12 corrected DAPI
measurements, checks the resulting crop regions, and writes QC figures. The
correction array is released only when both pass.

## Registration and quarantine

Each measurement receives one whole-well subpixel translation, estimated from
its corrected DAPI mosaic. If the whole-well estimate fails QC, corresponding
DAPI fields provide a robust median-consensus fallback for one replacement
whole-well shift. Fields are never warped independently.

After applying the global shift, every displayed field region receives a local
residual and correlation score. Failed field regions quarantine only the output
patch IDs that intersect them. Those patch IDs are removed from all 40
acquisitions for the well, preserving complete marker stacks. An entire well is
quarantined only if no valid global shift or one-patch common crop remains.

That description applies to field-level mode. In patch-level mode, each source
field receives its own subpixel translation. There is no 6 x 6 assembly and
therefore no internal source-tile boundary inside an output patch. A
field/cohort is quarantined if a required DAPI registration fails, its shared
valid overlap falls below the configured threshold, or the overlap cannot hold
one complete output patch. Other fields from the same physical well continue.

Patch-level defaults are under `registration` in
`config/pipeline_config.json`:

```json
"patch_level_max_shift_px": 128.0,
"patch_level_min_overlap_fraction": 0.80
```

Every run contains an initially empty
`provenance/registration_overrides.csv`. Supported actions are:

- `accept`: accept the calculated shift despite automatic global QC;
- `replace`: supply explicit `shift_y` and `shift_x`;
- `exclude`: quarantine that measurement, which quarantines the paired well.

In patch-level mode, fill `field_id` to target one original field. Leaving
`field_id` blank applies the override to all 36 fields for that measurement.
Field-level mode uses only override rows with a blank `field_id`, preserving
the original well/measurement override behavior.

After editing an override, rerun the relevant registration array task and its
downstream patch/PNG tasks. The normal `--resume` option compares task hashes,
detects the changed override, and recomputes instead of accepting stale output.
Any replaced patch or PNG files are recorded in
`manifests/aggregate/replaced_outputs.csv`.

## Run outputs

```text
results/runs/<run-id>/
├── corrected/                 # optional per-FOV intermediate; may be cleaned
├── registered_crops/          # optional common-crop intermediate; may be cleaned
├── quantitative_patches/      # traceable float32 OME-TIFF patches
├── png_8bit/                  # clipped uint8 derivatives
├── manifests/
│   ├── illumination/
│   ├── smoke/
│   ├── wells/
│   ├── clipping/
│   ├── cleanup/
│   ├── replaced_outputs/
│   └── aggregate/
├── provenance/                # exact config, selections, workbook, git diff
├── reports/wells/             # linked per-well registration reports
├── run_manifest.json
├── run_report.html
└── run_report.pdf
```

The HTML report is the easiest entry point. It links to every well, registration
overlay, field-QC heatmap, aggregate table, quarantine record, and patch
manifest. The PDF includes configuration, correction selections, all
registration shifts, all failed field observations, quarantine rows, and
flagged-well figures.

## Environment

For a fresh environment:

```bash
conda env create -f environment.yml
conda activate illumination_correction_env
python -c 'import numpy, torch; print(torch.from_numpy(numpy.zeros((2,2), dtype=numpy.float32)))'
```

The final command explicitly tests the NumPy-PyTorch bridge required by
BaSiCPy.

## Tests

```bash
pytest -q tests
```

The suite checks the exact Phenix layout, synthetic subpixel registration,
field-consensus fallback, common-crop geometry, patch quarantine, clipping,
registry completeness, NGFF/OME-TIFF round trips, HTML/PDF report generation,
and BaSiCPy model serialization when BaSiCPy is available.
