# MYONIS 4i quantification

Measure nuclei, supplied myotube objects, and segmented substructures across registered 4i acquisitions. The primary result is **`myonis_4i_data.h5ad`**, containing all object measurements and their relationships. By default the run also writes **`myonis_4i_data_measurements.h5ad`** for nucleus/myotube analysis, plus a three-page **`run_summary.pdf`** (with an exclusion appendix when fields fail). No re-segmentation or cross-patch merging is performed.

The example YAML now starts each field worker with **patch-level DAPI re-registration and SSIM nucleus QC**, then quantifies marker channels and only the reference DAPI using those same saved transforms. QC defaults to **fast mode**: registration and scoring still run; QC figures and dense score-map exports are skipped. The separate quantification intensity-audit visualisations remain controlled by `visualisation`.

## Install and run

For the direct test launcher, use your existing Python 3.10+ analysis environment. `run_quantification.py` reads this checkout's `src/fouri_quantification` and the neighbouring `MYONIS-VisTools/myonis_vis` directly; neither repository needs a package installation. Existing scientific libraries are reused. See the single-submission instructions below.

For the optional installed CLI commands shown next, install from `4i/quantification` with `python -m pip install -e '.[test]'`. `conda env create -f environment.yml` is an alternative when a separate environment is wanted. Activate the chosen environment before local or Slurm execution.

```bash
fouri-quantify validate --config /absolute/path/dataset.yaml
fouri-quantify run --config /absolute/path/dataset.yaml --workers 4
fouri-quantify run --config /absolute/path/dataset.yaml --run-dir /absolute/path/existing_run --resume
fouri-quantify report --input /absolute/path/myonis_4i_data.h5ad
```

`configs/dataset.example.yaml` targets the supplied preprocessing test run and early registration cohort. `configs/dataset.pilot.yaml` selects one field for a small real-data check. Copy a configuration before changing it. The optional structure directories are intentionally null, and aberration is disabled until the actual prediction paths/source acquisition are supplied. Those configurations produce nucleus/myotube image measurements and explicitly mark unavailable structure measurements; they do not claim complete substructure coverage.

## Configure a dataset: YAML walkthrough

Keep a personal dataset configuration and marker configuration beside the fixed plate layout:

```text
configs/
  dataset.nemo.yaml
  markers.nemo.yaml
  plate_layout.yaml
```

Copy `configs/dataset.example.yaml` and `configs/markers.yaml`, then set `marker_config: markers.nemo.yaml` in your dataset copy. The top-level `dataset.example.yaml` is a Nemo-path example and instead refers to `configs/markers.yaml` and `configs/plate_layout.yaml` relative to its own location.

| YAML setting | Meaning |
|---|---|
| `schema_version: 1` | Leave unchanged. |
| `dataset_id` | Stable dataset identity used in object IDs; keep it unchanged when comparing settings on the same data. |
| `preprocessing.root` | Preprocessing run containing manifests, quantitative TIFFs and DAPI PNGs. |
| `image_manifest` | Index of quantitative TIFF patches. Quantitative intensity features do not use 8-bit display PNGs. |
| `display_manifest` | Acquisition-wide display/texture quantization bounds; does not clip quantitative intensities. |
| `marker_config` | Marker images, requested compartments and optional links to structure segmentations. |
| `plate_layout` | Fixed plate/repeat and physical well-row-to-cell-line mapping. |
| `output_root` | Parent for automatically named runs; CLI `--run-dir` supplies an explicit run destination. |
| `selection` | Cohorts, plates, wells, repeats, fields and optional marker selection; `max_fields` limits actual quantification. |
| `registration_qc` | First-stage DAPI registration and nucleus QC; see the next section. |
| `measurement.overlap_policy` | `include` measures shared pixels in each peer; `exclude` removes them from every overlapping peer. |
| `geometry.pixel_size_um` | Verified calibration, or `null` for pixel-only measurements. |
| `geometry.rim_width_px`, `perinuclear_width_px` | Compartment widths for markers requesting these compartments. |
| `geometry.edge_threshold` | Extra margin for quantification geometry flags; not an automatic object filter. The original QC script uses its own boundary-touching edge rule. |
| `geometry.merge_patch_objects` | Keep `false`; source fragments are not reconstructed across patches. |
| `segmentations` | Reference nuclei, MyHC myotubes and optional structures, each with its source acquisition and segmentation version. |
| `nucleus_myotube_assignment.overlap_threshold` | Existing best-overlap assignment; `0.8` requires **more than** 80% of nuclear area in its selected myotube. |
| `nucleus_qc` | Optional external QC-table import when integrated `registration_qc.enabled` is false. |
| `myhc_filter` | Optional original metadata-based myotube filter. Its threshold of 15 uses the supplied legacy 8-bit metadata, not quantitative TIFF intensity. |
| `nuclear_aberration` | Enable and supply prediction directories plus their source DAPI acquisition to measure sigmoid maps. |
| `texture` | Default on/off switch, quantization levels, pixel distances and minimum mask/pair support. Marker-level options can override the default switch. |
| `execution.workers` | Number of field workers for local `run`; Slurm array concurrency is controlled separately. |
| `execution.allow_partial` | Keep `false` to reject required-input errors during preparation. Failed field jobs are still explicitly reported as partial/failed outputs. |
| `execution.hash_inputs` | Keep `true` for checksummed inputs and safe resume. |
| `visualisation` | Which quantified markers/patches receive overlays, exact ROI exports and numerical intensity audits. |
| `outputs` | PDF, older lightweight overlays and optional additional nucleus/myotube H5AD exports. |

Relative image-manifest, segmentation, prediction, QC and registration-config paths use `preprocessing.root`. Relative `marker_config`, `plate_layout`, `output_root` and an explicit `registration_qc.tools_dir` use the dataset YAML's directory. Absolute Nemo paths are accepted. On the Mac mount, use `/Volumes/lab-tedescos/...`; on Nemo, use `/nemo/lab/tedescos/...`.

`max_fields: 50` selects up to 50 **field/cohort units globally**, including all their patches. It is not 50 per well, and not just a visualization limit. `field_sampling: random` with `field_seed: 0` is reproducible but not stratified across cell lines/repeats. Filename metadata and the fixed layout retain plate, repeat, physical well, cell line, field and patch IDs. Rows 2–7 map to NCRM1, NCRM5, K32Del, R249W, L302P and L302P-Corr respectively.

`cohorts: [m01_08]` covers rounds 1–8. Round 9 p-4EBP1 and later acquisitions require a separate `m09_12` run with reference segmentations in that cohort. Integrated registration uses every DAPI round in the selected cohort, including earlier rounds than the reference. It never registers across the cohort boundary.

### eMYHC is an image-only marker by default

All supplied dataset templates omit eMYHC from `segmentations.structures`, and `configs/markers.yaml` keeps it as an ordinary image marker. DAPI nuclear masks and **MyHC-derived myotube masks** measure its intensity and texture. Its default marker entry is:

```yaml
include_dapi: true
markers:
  emyhc:
    acquisition_id: measurement_01_488_emyhc
    nuclear_compartments: [whole_nucleus]
    myotube_compartments: [whole_myotube, cytoplasm]
  # Keep your other marker entries here.
```

For an older personal YAML, remove `structure_source`, `structure_parent` and `structure_overlap_threshold` from the eMYHC marker entry, and remove `emyhc` from the dataset YAML's `segmentations.structures`. Removing only the segmentation entry while leaving `structure_source: emyhc` is a validation error. Ordinary myotube measurements automatically use `segmentations.myotubes`, whose source is `measurement_02_555_myhc`.

eMYHC has nuclear, whole-myotube and myotube-cytoplasm intensity/texture features. Cytoplasm excludes nuclear masks. Original nucleus and MyHC myotube morphology remains available; separate eMYHC-positive-object counts, morphology and occupancy are not computed.

For SC35/53BP1/BAF/cGAS, keep the marker's structure link and set its contour directory under `segmentations.structures`. Individual structures retain morphology, intensity, texture and parent IDs; SC35/53BP1 also produce counts, intensity summaries and occupied area per nucleus, with inherited myotube ancestry. A valid empty contour file produces zero counts; a missing/unconfigured file produces unknown (`NaN`) counts. `include_dapi: true` quantifies only `segmentations.nuclei.acquisition_id` (for example `dapi_r02`). Other DAPI rounds are still used for re-registration and persistence QC, but have no intensity/texture feature columns.

## Step 1: integrated DAPI re-registration and nucleus QC

The newest QC implementation now lives in `src/fouri_quantification/qc/run_4i_nucleus_segmentation_qc.py`, with its export helper and standalone `.sh` launcher alongside it. The old `downstream_analysis/4i_analysis` paths are compatibility symlinks to these files, so there is one maintained implementation. The installed standalone entry point is `fouri-nucleus-qc`; its existing Hungarian, binary-coverage and registration options remain available. Integrated quantification uses its registration-based mode. The standalone `.sh` launcher also defaults to fast mode; set `QC_FAST_MODE=false` to restore its figures/maps. The standalone Python CLI retains its explicit `--fast-mode` flag.

```yaml
registration_qc:
  enabled: true
  fast_mode: true
  metric: ssim
  threshold: 0.35
  aggregation: max
  smoothing_sigma_px: 1.0
  min_valid_fraction: 0.90
  local_window: 11
  texture_floor: 0.01
  comparison_gain_limits: [0.01, 100.0]
  compare_metrics: false
  registration_config: null
  tools_dir: null
nucleus_qc:
  directory: null
  require_all: false
```

The reference round is derived from `segmentations.nuclei.acquisition_id`; there is no second reference setting that could disagree with it. The supplied examples now use round 2. For each selected patch, the first stage:

1. Reads that patch's exported DAPI PNG in every cohort round and its original reference nucleus contours. Only the reference round needs nucleus segmentation.
2. Calls the **same preprocessing `register_patch_field` and `apply_translation` helpers used by the original QC script**, with the preprocessing run's saved registration settings. It estimates a separate residual translation for each patch/round, directly to the chosen reference. Existing preprocessing shifts and illumination correction are not reapplied.
3. Scores the aligned DAPI with the selected metric and writes metadata for **every** original reference object. The default remains SSIM change with threshold `0.35` and max aggregation.
4. Saves the translations, registration diagnostics, reference/source identities, checksums and valid image bounds. Quantification reuses these transforms for every marker channel in that round/patch, without independently registering marker channels.
5. Translates myotube/structure contours from their declared source acquisition before nucleus→myotube and structure→nucleus assignment. Reference nuclear contours stay fixed. Input files, source line numbers and object UIDs are preserved.

`fast_mode: true` skips QC figures and dense `.npz` score maps, including comparison figures. It **does not skip registration**, SSIM maps needed for calculation, per-object scores, per-round evidence, QC tables or transform exports. `false` adds the original QC diagnostics for the sampled visualisation fields; other fields remain in fast mode. Quantification overlays/intensity audits are independent and remain available under `visualisation.enabled` even when QC fast mode is on.

`metric` accepts `ssim`, `relative`, `relative_calibrated`, `pixel_max`, `pixel_mean`, `zncc_signal` or `gradient`. `threshold` applies to the chosen metric. For SSIM it is a **change score** (`1 - clip(SSIM, 0, 1)`), not a raw similarity cutoff. `aggregation: max` takes the maximum change across rounds at each pixel, then averages those pixels within a nucleus; a score at or below `0.35` passes. `pixel_mean`/`pixel_max` require matching aggregation. Edge, low-signal and insufficient-valid-area objects remain unevaluable rather than being called disappeared. `min_valid_fraction` controls the required evaluable nuclear fraction; local SSIM windows also exclude invalid borders.

`local_window`, `smoothing_sigma_px`, `texture_floor` and `comparison_gain_limits` preserve the original scoring controls. DAPI background/gain/smoothing adjustments apply **only to QC scoring**, never to marker intensity measurements. `compare_metrics: true` retains exploratory comparisons, but imported flags still use the selected main metric. Registration settings come from `preprocessing.root/provenance/pipeline_config.json`, falling back to `4i/preprocessing/config/pipeline_config.json`; `registration_config` can supply another JSON file containing its `registration` section. `tools_dir` is auto-detected from this repository and can be overridden with the directory containing `myonis_cosmx` and `MYONIS-VisTools`.

Every quantitative TIFF is translated with linear interpolation into float32, retaining its native intensity scale. Shifted borders are missing (`NaN`), not zero-valued observations. Existing intensity coverage columns report missing pixels; a partially missing ROI has undefined intensity statistics. Texture uses finite in-mask pairs and records pair counts. No automatic crop, intensity normalization, object deletion or fallback to unregistered images occurs. If DAPI registration fails, that field fails and is recorded in the final run status.

Sigmoid ensembles retain the existing uint8 conversion, available-model mean and complete-assigned-nuclei aggregation rules. When their source round differs from the reference, the same translation is scaled to the native probability grid and applied to floating probabilities without requantizing to PNG. Subpixel interpolation can change intensities and texture, which is why all parameters and transforms are retained. Source polygon/raster morphology is kept; translated geometry controls measurement regions and parent assignment. Thus enabling registration can legitimately change parent assignments, while switching only the overlap policy does not.

The worker sequence is **QC/registration → quantification** for each field, allowing independent fields to run in parallel. Preparation only validates/indexes inputs and freezes settings. There is no need to submit a separate QC job or manually supply its output folder. Keep `nucleus_qc.directory: null` when integrated QC is enabled; external QC import and integrated QC cannot be enabled together. Old YAML files without a `registration_qc` block retain their previous behavior (integration disabled); the new examples explicitly enable it.

```text
run_directory/
  registration_qc/
    transforms.csv                 # Combined table, one row per patch and round
    summary.json
    myonis_4i/                     # Combined reference-object QC exchange
      manifest.json
      nucleus_qc.csv
      round_qc.csv
      patch_qc.csv
    <field_task_id>/
      transforms.csv
      transforms.json             # Exact source→reference translation matrices
      complete.json               # Cache checksums
      assignments/*.json
      summary.json
      field_summary.csv
      round_summary.csv
      nucleus_summary.csv
      myonis_4i/
      # QC plots/maps only when fast_mode: false
  myonis_4i_data.h5ad
  run_summary.pdf
  qc_visualisations/index.html
```

Failed fields retain their available QC diagnostics under their task directory; combined exports cover accepted field checkpoints. QC caches and transforms are checksummed and can be reused when a downstream field computation needs rerunning. Changed inputs, settings, package code or preprocessing registration helpers require a new run. Missing/corrupt QC artifacts trigger recomputation rather than silent reuse.

In the main H5AD, use `.uns['registration']['transforms']` for translations and diagnostics, `.uns['qc']['nucleus_qc']` for per-object/per-round QC evidence, `.uns['geometry']` for geometry in the measurement frame, and `.uns['source_geometry']` for original source contours. Nuclear `.obs` retains `qc_pass`, nullable `is_persistent`, `is_edge`, scores and reasons. Myotubes retain total and QC-stratified nucleus counts. No nucleus is removed because it failed SSIM or touched an edge. Filtering nuclear rows later does not automatically recalculate myotube aggregates.

Transform `shift_y`/`shift_x` values are in image pixels, positive down/right. `matrix_json` maps homogeneous source coordinates `[x, y, 1]` to reference coordinates. Each row is keyed by the full `field_key`, `spatial_key` and round, so local field/patch numbers from different wells or repeats cannot collide.

Detailed quantification audit NPZs include `source_values` when registration is enabled: these are the exact translated quantitative pixels used for measurement. Their JSON companions record the transform and source TIFF. Compare exported ROIs with `source_values`, not directly with an unshifted TIFF.

## Submit an integrated run to Nemo

Run these commands on Nemo after logging in through your usual method. The `ncpu` partition below matches the existing nucleus-QC launcher; memory/time allocations are starting values to check with a pilot.

### Run the configured test dataset with one submission

The main-folder `dataset.example.yaml` contains your supplied Nemo paths for nuclei, MyHC myotubes, 53BP1, BAF, SC35, cGAS and round-3 nuclear-aberration predictions. It enables registration/SSIM QC in fast mode, selects all fields matching its cohort/plate/well filters (`selection.max_fields: null`), and uses 120-dpi PNG visualisations for one patch per sampled field, with mask/audit companion files disabled. Set `selection.max_fields` to a number for a pilot. Its marker configuration is `configs/markers.yaml`, with eMYHC as an ordinary image marker.

Using your existing analysis environment, submit the launcher beside that YAML:

```bash
conda activate myonis_env_2
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification
sbatch --export=ALL run_quantification_test.sh
```

`run_quantification_test.sh` requests one `ncpu` job with 16 CPUs, 32 GB RAM and 24 hours, and runs 12 field workers. Its Slurm header is followed by this single command:

```bash
python -u "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_quantification.py" \
    --config "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/dataset.example.yaml" \
    --workers 12
```

`--config` selects your YAML, and `--workers` sets the number of fields processed in parallel. Edit these arguments directly in the script. Keep the worker count at or below `#SBATCH --cpus-per-task`. The command uses Python from your activated environment. To change the field count, paths, registration/QC settings or measurements, edit the YAML.

`run_quantification.py` loads `src/fouri_quantification` relative to its own location and loads MYONIS-VisTools directly from the neighbouring repository. Neither local package needs installing. It can be called by absolute path from any working directory; spawned field workers use the same source paths. If the neighbouring MYONIS-VisTools checkout is absent, an installed copy can be used. The scientific dependencies listed in `pyproject.toml` are still required in your active environment; install a missing dependency only if needed. If the older `python -m fouri_quantification.cli` command reports `ModuleNotFoundError`, use this direct script.

This one command performs input validation, preparation, registration/QC, quantification and final reporting. A new `run_<timestamp>_<fingerprint>` folder under the YAML's `output_root` holds the H5AD, PDF, transforms and visualisation audits. The Slurm log is `myonis_4i_test-<job_id>.out` in the submission directory. Submit either this single-job script or the batch script below for a given analysis.

### Multiple jobs, with several workers per job (no package installation)

Use the separate main-folder `run_quantification_batch.sh` to distribute the same calculations across several Slurm jobs:

```bash
conda activate myonis_env_2
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification
sbatch --export=ALL run_quantification_batch.sh
```

The script contains one explicit Python command. Edit these arguments directly:

```bash
python -u "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_quantification.py" submit-batch \
    --config "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/dataset.example.yaml" \
    --jobs 4 \
    --workers 12 \
    --partition ncpu \
    --log-dir "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification" \
    --memory 128G \
    --time 48:00:00 \
    --finalize-workers 4 \
    --finalize-memory 128G \
    --finalize-time 10:00:00 \
    --troubleshoot_timing
```

| Setting | Meaning |
|---|---|
| `--jobs 4` | Split selected fields into four jobs; at most four compute jobs run simultaneously. Slurm may schedule fewer. If fewer than four fields exist, fewer jobs are submitted. |
| `--workers 12` | Each job requests 12 CPUs and processes up to 12 fields concurrently: up to 48 fields across four jobs. Overrides YAML `execution.workers` for this run. |
| `--memory 128G` | Memory for each whole compute job, shared by its workers. Four jobs may therefore request 512 GB in total. |
| `--time 48:00:00` | Time limit for each compute job. |
| `--log-dir /path/to/folder` | Folder for worker and finalisation Slurm output, including prints, progress and timings. Defaults to `SLURM_SUBMIT_DIR`, or the current directory outside Slurm. Relative paths use that same base. |
| `--finalize-workers` | Parallel checkpoint readers and CPUs for the final assembly job; default 4. This does not repeat field calculations. |
| `--finalize-memory`, `--finalize-time` | Resources for final assembly/reporting. Memory is for the whole job; it must hold the combined tables. |
| `--require-complete` | Optional strict mode: refuse final output if any field is missing or invalid. By default batch submission assembles valid fields and reports exclusions. |
| `--troubleshoot_timing` | Stage timings plus the patch/image timing summary in the final job's log. Replace with `--troubleshoot` for the existing full diagnostics. |
| Optional `--visualisation_fields 5` | Five sampled fields across the entire dataset, not five per job. If omitted, the YAML visualisation limit applies. This never limits quantified fields. |

The shell script's `#SBATCH` header applies **only to preparation** (currently 16 CPUs, 128 GB, four hours). The Python arguments control the subsequent compute and assembly jobs. Each uses the exact Python executable from your activated environment and the local source launcher. There are no additional Python dependencies.

The sequence is:

1. **Prepare once:** validate/index inputs, freeze configuration, choose fields and the global visualisation sample.
2. **Compute in batches:** distribute complete field/cohort tasks by round-robin index. With 50 fields and four jobs, groups contain 13, 13, 12 and 12 fields. A field's patches, nuclei, myotubes and substructures stay together. Every job calls the same field calculation code and writes the same independent field checkpoints as the single-job pipeline.
3. **Assemble once:** wait for every batch job to finish, verify all expected checkpoints, then use the existing aggregation code to write `myonis_4i_data.h5ad`, the PDF and the visualisation index. This preserves the nucleus/myotube/structure tables and relationships; it does not concatenate only the nuclear AnnData matrix.

Calculations, registration transforms, reference-only DAPI, overlap policy and diagnostic contents are unchanged. The split and maximum concurrent worker count are recorded in `run_manifest.json`; job IDs, log paths and resources are in `batch_submission.json`. `selection.max_fields` controls how many fields are quantified: the main-folder YAML now uses `null` for all eligible fields. `fields: all` alone does not cancel a numerical `max_fields` cap. `--jobs` and `--workers` control parallelism, not dataset selection. Already prepared runs retain their frozen selection; submit a new run after changing the YAML.

Preparation prints the run folder, submitted job IDs and exact log paths to `myonis_4i_prepare-<job_id>.out`. With the supplied `--log-dir`, worker logs appear beside it in the main quantification folder as `myonis_4i_batch-<array_id>_<index>.out`; final assembly and the patch/image timing summary appear in `myonis_4i_finalize-<job_id>.out`. Both stdout and stderr use these files, so prints, progress bars and errors stay together. Detailed `logs/timings.csv` and structured error records remain inside the results folder. Older submissions retain their original `results/<run>/logs/batch-...out` and `finalize-...out` locations; changing this setting does not redirect jobs already submitted. Batch timings retain per-worker averages; they do not treat scheduler queue time as image-processing time.

The final job uses an `afterany` dependency and explicitly permits a partial archive. Missing, failed or invalid fields are excluded, and their identities and reasons are saved in `excluded_fields.csv`, `.uns['qc']['excluded_fields']`, `run_manifest.json`, and a PDF appendix. The archive retains status `partial`; an explicitly requested partial finalisation exits successfully when its output/report checks pass. A run with no valid fields still fails without writing an empty result. Add `--require-complete` to `submit-batch` to restore strict output: missing checkpoints are then detected before loading HDF5 payloads. Optional input omissions still follow the separate `complete_with_omissions` rules. Slurm concurrency caps and whole-array dependencies follow the [official array documentation](https://slurm.schedmd.com/job_array.html).

For recovery, after the original jobs have finished, rerun the failed batch in a compute allocation using the direct launcher:

```bash
python /absolute/path/4i/quantification/run_quantification.py worker-batch \
    --run-dir /absolute/path/to/the/existing/run \
    --batch-index 0 --batch-count 4 --workers 8
```

Batch indices start at zero. Keep the original `--batch-count`; valid checkpoints in that batch are reused. After all retries finish, run `python /absolute/path/4i/quantification/run_quantification.py finalize --run-dir /absolute/path/to/the/existing/run --require-complete` in a compute allocation with sufficient memory. Do not rerun the top-level submission script for recovery: it starts a new analysis. A submission failure retains the prepared run and any accepted job IDs in `batch_submission.json`; check those jobs before resubmitting.

### Rerun only final assembly

For the existing full run, use the main-folder `run_quantification_finalize.sh`. Activate the **same environment as the original run** (see `batch_submission.json` -> `python`) and submit:

```bash
cd /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification
sbatch --export=ALL run_quantification_finalize.sh
```

The script points to `test_run_r1-8/run_20261001_232056_07f18205` and calls:

```bash
python -u "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_quantification.py" finalize \
    --run-dir "/nemo/lab/tedescos/home/users/chois1/4i/4i_experiment_0826/4i_quantification/test_run_r1-8/run_20261001_232056_07f18205" \
    --workers 4 \
    --allow-partial \
    --allow-finalizer-update
```

Only the saved checkpoints are assembled; no image indexing, registration or feature computation is repeated. The original YAML snapshot supplies analysis settings. Do not run `submit-batch` again for this recovery, because that prepares a new analysis. The finalisation script's absolute Slurm log path places its output beside the other main-folder logs.

- `--allow-partial` explicitly permits missing/failed fields, reports every exclusion, and gives a successful exit status if the partial archive and requested reports are written successfully. Without it, the legacy manual `finalize` still writes a partial archive but returns nonzero; `--require-complete` prevents writing one.
- `--workers 4` reads four checkpoints concurrently using separate processes. Match `#SBATCH --cpus-per-task=4`. Pending copies/results are bounded by that worker count and output ordering is deterministic.
- Each checkpoint is streamed to temporary local storage while its SHA-256 is verified, then HDF5 reads its many internal tables locally. Temporary copies are deleted after each read; the full dataset is not copied at once. Registration/QC artifact checks remain enabled. `--scratch-dir /local/path` overrides `SLURM_TMPDIR` or the system temporary directory. It must exist and fit one checkpoint per reader; final H5AD writing also needs local temporary space for the output.
- `--allow-finalizer-update` is specifically for assembling existing checkpoints after an assembly/reporting code update. It permits changes only in the finalisation, CLI/Slurm, pipeline, output and PDF modules. Measurement, object-assignment, registration, configuration, dependencies and renderer checks remain pinned. Original computation hashes/settings stay intact; current assembly hashes and allowed differences are recorded separately in `run_manifest.json['finalization']` and embedded H5AD provenance. This flag does not permit rerunning field computations with changed code.

Finalisation still has to read every valid checkpoint and write the combined archive. Local HDF5 reads and parallel readers reduce shared-filesystem overhead; they do not guarantee a particular runtime. The PDF uses the already assembled archive in memory, avoiding a second full H5AD load.

For a partial archive, access the exclusions with:

```python
excluded = adata.uns["qc"]["excluded_fields"]
print(excluded[["plate_name", "repeat", "cell_line", "well_id", "field_id", "reason"]])
```

Each record identifies the entire excluded field and the number of omitted spatial patches. The specific failing patch/round is retained when available. Valid patches inside a field with no checkpoint are also absent; no fabricated measurements are added. `.uns['sample_metadata']` records selected, completed and excluded field counts per sample. The PDF keeps its three core pages and appends all excluded identities/reasons, adding pages as necessary.


### 1. Optional installed CLI setup for the array workflow

The direct single-job and batch launchers above do not need these package-install steps. The older one-field-per-job array workflow below remains available and uses the installed `fouri-quantify` command.

```bash
repo_path=/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx
quant_dir="$repo_path/4i/quantification"
cd "$quant_dir"

conda env create -f environment.yml
conda activate myonis_4i_quantification
python -m pip install -e /nemo/lab/tedescos/home/users/chois1/computational_tools/MYONIS-VisTools

command -v fouri-quantify
python -c "import fouri_quantification, myonis_vis; print('Imports OK')"
```

Skip environment creation if it already exists. After pulling code into an existing environment, `python -m pip install -e '.[test]'` from `4i/quantification` also registers the new `fouri-nucleus-qc` entry point. Integrated QC needs MYONIS-VisTools even if quantification visualisations are disabled, because it retains that package's original image/contour readers and masks.

### 2. Copy, edit and validate the YAML

```bash
cp configs/dataset.example.yaml configs/dataset.nemo.yaml
cp configs/markers.yaml configs/markers.nemo.yaml
```

Edit those copies with your reference contours, optional structure/prediction paths and selection; set `marker_config: markers.nemo.yaml`. The reference acquisition must describe the actual nucleus contours. Begin with `selection.max_fields: 2`, `visualisation.max_patches_per_field: 1` and a few visualization markers before requesting 50 fields. These visualization limits do not reduce quantified channels/patches.

```bash
config_path="$quant_dir/configs/dataset.nemo.yaml"
fouri-quantify validate --config "$config_path"
```

Review selected fields, marker names, DAPI availability and missing-input messages. Optional `directory: null` structure sources remain explicitly unavailable; required errors must be resolved before proceeding.

### 3. Prepare a new run

```bash
run_dir="$quant_dir/results/run_$(date +%Y%m%d_%H%M%S)"
fouri-quantify prepare --config "$config_path" --run-dir "$run_dir"
```

Do not create that run directory beforehand. Preparation creates it and its `logs` directory, snapshots the configuration and hashes selected inputs, including the QC DAPI PNGs. Registration/scoring execute in the subsequent compute jobs.

### 4. Submit the field array

```bash
task_count=$(python -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
  "$run_dir/tasks.json")

worker_job=$(sbatch --parsable \
  --partition=ncpu --export=ALL \
  --cpus-per-task=1 --mem=8G --time=04:00:00 \
  --array="0-$((task_count - 1))%8" \
  --output="$run_dir/logs/worker-%A_%a.out" \
  "$quant_dir/slurm/quantification_worker.sh" "$run_dir")

echo "Worker array: ${worker_job%%;*}"
```

For 50 selected fields this submits 50 tasks, with at most eight active simultaneously. Each task performs registration/QC, then quantification for all of its field's patches. The YAML's `execution.workers` applies to local execution, not this array. Extra CPUs do not add parallelism within one array task. The current scripts assume the activated environment is on `PATH`; `--export=ALL` passes that environment into the jobs.

### 5. After successful array submission, submit aggregation/reporting

```bash
sbatch \
  --partition=ncpu --export=ALL \
  --cpus-per-task=1 --mem=16G --time=02:00:00 \
  --dependency="afterany:${worker_job%%;*}" \
  --output="$run_dir/logs/finalize-%j.out" \
  "$quant_dir/slurm/aggregate_and_report.sh" "$run_dir"
```

`afterany` waits for every array task, including failures, so the reducer can report an explicitly partial/failed run. A generated H5AD is not proof that all fields succeeded: check `run_manifest.json`, the PDF and job exit codes. Array concurrency and dependency behavior follow the [Slurm array documentation](https://slurm.schedmd.com/job_array.html).

```bash
squeue -u "$USER"
sacct -j "${worker_job%%;*}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Use pilot runtime and `MaxRSS` to adjust later allocations. Figure count is sampled fields × visualised patches per field × requested marker views. Non-reference DAPI is excluded from quantification views. Registration fast mode is independent of these views.

The existing `slurm/submit_quantification.sh CONFIG NEW_RUN_DIR` helper performs preparation and both submissions automatically. Run it with `bash`, not `sbatch`; configure its worker/reducer resource directives for your cluster first. Its default array has no concurrency cap. Use either that helper or the explicit sequence above, once per run. Registration is automatically included in either workflow when enabled in YAML.

Both nucleus- and myotube-level measurements remain in `myonis_4i_data.h5ad` with `outputs.object_h5ad: false`. Setting it to `true` additionally exports conventional `exports/nuclei.h5ad` and `exports/myotubes.h5ad` files.

## Quantify all fields, visualise a random subset

Quantification and visualisation have **independent field limits**. For a full run with views from 100 random fields:

```yaml
selection:
  cohorts: [m01_08]
  fields: all
  max_fields: null           # Quantify every eligible field and all its patches.
  field_sampling: random
  field_seed: 0
visualisation:
  enabled: true
  max_fields: 100            # Visualise at most 100 random field/cohort units.
  field_seed: 0              # Independent, reproducible visualisation seed.
  markers: all
  max_patches_per_field: 1   # First patch per sampled field; null = every patch.
  dpi: 120
  max_labels_per_panel: 40
  export_masks: false
  export_audits: false
execution:
  workers: 12
  troubleshoot: false
  troubleshoot_timing: false
```

`selection.max_fields` limits actual quantification. `visualisation.max_fields` limits only visualisation, after the quantification selection. A limit of 100 means 100 fields, not 100 individual PNGs: each sampled field gets the requested marker/patch views. Set it to `0` (or `visualisation.enabled: false`) for no figures, or `null` to visualise every quantified field. Sampling uses full plate/repeat/well/field/cohort identities, never a local field number alone. It is reproducible across worker counts and manifest order, independent of `selection.field_seed`, and does not stratify by cell line/repeat. Fewer than 100 eligible fields means every available field is sampled.

Quantification identities are saved in `inputs/selected_fields.json`; the separate visualisation sample is in `inputs/visualisation_fields.json` and the run manifest. Your supplied main-folder test YAML retains its quantification limit of 50; set that limit to `null` yourself when ready for a full dataset.

The direct Python launcher loads MYONIS-VisTools from its neighbouring checkout, without installing it. Renderer source hashes are still recorded for reproducibility. Each sampled view shows the quantitative channel, nucleus/myotube/structure overlays and the actual measurement regions coloured by saved mean. Enabled nuclear-aberration predictions get their own native-grid views. Empty and missing structure inputs remain distinct.

Readable paths now look like:

```text
qc_visualisations/
  plate-sc_rep5-6__rep-05__cell-K32Del__well-r04c02__field-f03__cohort-m01_08/
    patch_0001__pr01-pc01__y25-x18/
      plate-sc_rep5-6__rep-05__cell-K32Del__well-r04c02__field-f03__cohort-m01_08__patch_0001__pr01-pc01__y25-x18__marker-baf.png
```

Normal mode writes PNGs plus the shared gallery/index, H5AD and report; per-view TSV/JSON/NPZ companions are off by default. Internal numerical intensity audits still run for rendered views, with their pass/mismatch counts in the H5AD/gallery. Required checkpoints, transforms, QC metadata and provenance are retained for valid results and safe resume.

## Progress and troubleshooting

`tqdm` displays field completion, elapsed time and an estimated remaining time; checkpoint aggregation has a separate progress bar. Slurm logs update at a slower cadence to avoid excessive output. The first field must finish before a useful field ETA is available. If `tqdm` is absent from your existing environment, the run continues with periodic counts rather than requiring installation.

For timing only, edit the command in `run_quantification_test.sh` to read:

```bash
python -u "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_quantification.py" \
    --config "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/dataset.example.yaml" \
    --workers 12 \
    --troubleshoot_timing
```

For full diagnostics and timing on five random fields, replace the final `--troubleshoot_timing` line with:

```bash
    --troubleshoot \
    --visualisation_fields 5
```

`--troubleshoot` (alias `--optimise`) enables **all quantified markers/predictions and all patches in the sampled fields**, full DAPI registration/filtering figures, comparison metrics, dense QC maps, exact measurement masks and TSV/JSON intensity audits. Unsampled fields retain fast registration/QC and produce no figures. The visualisation field limit still applies; this flag never silently generates full diagnostics for an entire large dataset. All rounds remain available in registration figures, while only reference DAPI is quantified. This diagnostic mode intentionally takes longer than a normal run.

`--troubleshoot_timing` prints stage summaries and writes one `logs/timings.csv` containing input indexing/checking, registration, scoring, contour loading, assignments, morphology, image reads/shifts, intensity, texture, rendering, aggregation and H5AD/PDF writing. Timings are wall-clock seconds per worker, not CPU time. The legacy field-duration summary covers registration and quantification; the timing CSV additionally exposes indexing, verification and output overhead. Stages ending in `_total` include their sub-stages; do not add both when estimating totals. Parallel-worker durations cannot be summed to obtain whole-run elapsed time. Remove the flag to stop detailed timing output. Neither flag changes measurement definitions or feature selection.

Both timing flags also print a compact end-of-run summary, saved under `timing_summary` in the existing `run_manifest.json` (no additional output file):

- **Seconds per spatial patch:** registration/QC and measurement work across all quantified channels, with diagnostic exports and recorded verification/checkpoint overhead shown separately. Fields with and without diagnostic exports are summarised separately.
- **Seconds per quantified channel image:** measurement work divided by successfully measured channel images. Shared work such as segmentation morphology, relationships and sigmoid features is amortised into this estimate. An image is one channel of one patch; only reference DAPI counts as a quantified DAPI image. The other DAPI rounds remain registered, but neither those registration-only images nor sigmoid maps inflate this denominator.
- **Effective wall time per patch/image:** total pipeline elapsed time, including preparation, diagnostics and final outputs, divided by completed patches/images across all workers. This measures throughput, not the latency of an individual image. It is reported only for a fresh complete run; resumed/partial runs and standalone finalization retain worker averages without claiming full-run throughput.

Patch/image worker averages are estimates from field timings, not separate stopwatches for each image. Missing channel images are listed separately and excluded from the image denominator. Checkpoints created without timing are omitted from these averages, and the summary states how many completed fields have timings. Full troubleshooting output is unchanged; no new flag is needed.

Compartment bounds are now computed once per patch and reused across markers so intensity/texture functions scan only the object's crop. Original masks, overlap rules, parent assignments and numerical calculations are preserved. Acquisition matching also uses indexed lookups rather than repeatedly scanning a dataframe for each contour file. Actual Nemo bottlenecks should be assessed from the timing output; local tests do not establish full-dataset cluster performance.

All display contours use `smoothing="none"`. Pixel fills and exported ROIs use the exact boolean masks, preserving holes and overlaps. Intensity/display limits are fixed per acquisition and do not change quantitative measurements. White pixels indicate overlapping displayed ROIs; separate object masks retain each ROI. On-panel labels are capped for readability, while all objects remain in the audit tables and mask archives.

Open `qc_visualisations/index.html` for an offline, filterable index. With `--troubleshoot` (or explicit audit/mask exports), each view has:

- `.png`: source, segmentations and measurement regions.
- `.tsv`: full IDs/parents, feature names, pixel counts, saved means/sums, independently recalculated means/sums and comparison results. `integrated_stored=false` means that feature has no stored sum to compare; the pixel sum is still provided.
- `.npz`: packed boolean masks, with one `mask_key` per audit row and the image shape. No pickle is used.
- `.json`: source paths, spatial identity, display limits and mask encoding.

```python
import pandas as pd
import tifffile
import numpy as np
from fouri_quantification.reporting.visualisation import read_qc_mask

audit = pd.read_csv('/path/to/view.tsv', sep='\t')
row = audit.iloc[0]
roi = read_qc_mask('/path/to/view.npz', row.mask_key)
with np.load('/path/to/view.npz', allow_pickle=False) as archive:
    image = archive['source_values'] if 'source_values' in archive else tifffile.imread('/path/to/source_quantitative.ome.tif')
print(row.object_uid, row.saved_mean, image[roi].mean(dtype='float64'))
```

For registered quantitative views, `source_values` contains the exact translated TIFF pixels, with missing borders. For sigmoid views, the NPZ also contains `source_values`: the actual available-model mean probability map on its native prediction grid. Use this array rather than a quantitative TIFF for the numerical check. The H5AD `.uns['qc']['visualisations']` indexes every view using paths relative to the run folder. Render failures and arithmetic mismatches are recorded and cause a nonzero run exit; successfully computed measurements remain saved. Resume checks visualisation file checksums and regenerates a field when a required QC artifact is missing.

Detailed rendering runs inside the existing parallel field workers. Limit `visualisation.max_fields`, `markers` and `max_patches_per_field` for routine runs. None reduces quantified fields, markers or patches. Packed masks imply TSV/JSON audits so exported mask keys remain interpretable. The older `outputs.qc_overlays` defaults to zero and is also restricted to sampled fields.

Historical pilot (before reference-only DAPI): the visualisation pilot is saved in `results/visualisation_qc_20261001/`, configured by `configs/dataset.visualisation-pilot.yaml`. It quantifies all four patches of one real field and renders its first patch across 26 acquisitions. All 2,004 ROI intensity audits passed; nucleus/myotube feature tables were unchanged by enabling QC. The gallery is `qc_visualisations/index.html`. Missing structure paths are explicitly labelled in these real-data examples. Synthetic tests exercise the supplied-structure and sigmoid views, including overlapping structures, parent clipping, native prediction grids, resume and parallel rendering.

## Include or exclude overlapping measurement pixels

```yaml
measurement:
  overlap_policy: include  # include (default) or exclude
```

`include` preserves the existing numerical behaviour: each object measures its own mask, so pixels shared by two objects contribute to both. `exclude` removes every pixel shared by two or more peer objects from **all** those objects. For two overlapping nuclei A and B, their measurement masks become `A & ~B` and `B & ~A`; no winner is chosen. Triple overlaps and identical masks follow the same rule. Overlap is evaluated separately within each patch for nuclei, retained myotubes, and each structure segmentation source. A nucleus inside a myotube, or an SC35 object inside a nucleus, is normal containment and is not removed on that basis. This option does not merge or deduplicate objects across patches or fields.

The policy affects marker intensity, masked texture, N/C intensity ratios, structure pixel measurements and their parent summaries, and nuclear-aberration sigmoid measurements. Rims, interiors and perinuclear territories are built from original masks and then trimmed. Removed nuclear pixels never become cytoplasm, and cut boundaries do not create new nuclear rims. Parent-clipped structure measurements also exclude ambiguous pixels in the parent nucleus or myotube cytoplasm. Whole-structure measurements exclude overlaps with structures from the same source; nuclear-parent overlap is additionally removed in the parent-clipped measurements.

Source contours, IDs, original morphology, parent assignments, nucleus/structure counts, and geometric summaries such as `nuclear_area_fraction` and `assigned_nuclei_area_px2` stay unchanged. Structure counts still include objects whose parent-clipped measurement becomes empty; counts represent detections, not the number of measurable objects. Structure occupancy and coverage use the retained measurement masks. Equal-object intensity summaries remain NaN if any contributing object has an undefined mean. Resident-nucleus intensity/texture summaries reflect the selected policy.

Each nucleus, myotube and measured structure records `overlap_area_px2`, `overlap_fraction`, `measurement_area_px2` and `measurement_empty`. These refer to the whole object on the quantitative image grid; individual compartments also retain their own `n_pixels` columns. Completely overlapped objects keep their IDs and counts, have zero retained area, and undefined intensity/texture values (NaN). Sigmoid sums retain the existing empty-mask convention of zero with a NaN mean. Sigmoid overlap is evaluated on the **native prediction grid**, and its area-normalized summaries use retained nuclear area; exclude mode additionally records `nuclear_aberration_measurement_area_px2` on that grid. Missing predictions still produce NaN, and the complete-assigned-nuclei rule is preserved.

The chosen policy is saved in `.uns['measurement']['overlap_policy']`, the resolved configuration, run manifest, summary PDF, and QC figure/audit metadata. QC fills and exported masks show the actual selected measurement regions, while the segmentation panel retains original contours. Run each mode into a **different output directory** and compare rows by `object_uid`; changing the policy invalidates resume into an existing run. Configurations without this setting continue to use `include`.

## Inputs and paths

Required inputs:

- A preprocessing `patch_manifest.csv` and its quantitative TIFFs.
- Canonical YOLOv8-seg contour directories for nuclei and MyHC myotubes, each with its source acquisition ID and segmentation version.
- `markers.yaml`, mapping stable marker keys to acquisitions and measurement options.

Optional inputs:

- One contour directory each for SC35, 53BP1, BAF and cGAS.
- One sigmoid PNG directory per nuclear-aberration model. Enable `nuclear_aberration` and specify the source DAPI acquisition. Maps retain the existing uint8-to-[0,1] conversion and mean-ensemble semantics.
- A `myonis_4i` metadata directory from `downstream_analysis/4i_analysis/run_4i_nucleus_segmentation_qc.py`, as described below.
- The original MyHC intensity metadata CSV if the existing MyHC filter is wanted. Its default threshold of 15 is on that metadata's original 8-bit scale, not the quantitative TIFF scale.
- Verified `geometry.pixel_size_um`. Pixel measurements are available without it. Physical-unit ring widths require it.
- An optional supplemental sample CSV keyed by `plate_name,well_id`. It cannot contradict filename-derived metadata.

All relative image-manifest, segmentation, prediction and MyHC-filter paths resolve under `preprocessing.root`. `marker_config`, `plate_layout`, optional `asset_map`, optional `sample_metadata`, and `output_root` resolve relative to the dataset YAML. Absolute paths are accepted. Image paths containing `quantitative_patches/` are rebased under the supplied preprocessing root, so original `/nemo/...` manifests work on a mounted `/Volumes/...` run.

For renamed segmentation exports, an optional `asset_map.csv` supplies `source_key,path,image_name`. `image_name` is the complete canonical original image filename; `path` points to the actual asset. Source keys are `nuclei`, `myotubes`, `structures.sc35`, `structures.53bp1`, etc., and `aberration.0`, `aberration.1` for models. Source acquisition and coordinates are still validated. A source listed in this CSV uses the explicit mapping instead of directory discovery.

A valid empty contour file means zero detected objects. An absent file means unavailable segmentation. Invalid rows, duplicate spatial/acquisition keys and incompatible coordinates are errors. A missing required nucleus/myotube input prevents execution by default. `execution.allow_partial: true` permits preparation with required-input errors; it is separate from finalisation policy. Failed field-cohorts remain recorded. Explicit `finalize --allow-partial` treats a successfully written partial archive as successful; ordinary full-run failures still return nonzero.

## Reference-nucleus persistence and edge QC

Quantify **all original reference objects**, then attach QC metadata. The QC script now creates this folder under its existing `--output-dir`, without deleting, filtering or renumbering any contours:

```text
myonis_4i/
  manifest.json       # Active method/settings, table checksums, definitions and run status
  nucleus_qc.csv      # One row per original reference object, including rejected/unknown objects
  round_qc.csv        # Per-reference-object, per-round evidence
  patch_qc.csv        # Completed, failed, incomplete and unavailable-identity patches
```

These are dataset-wide tables for the selected QC run; each row retains full plate/repeat/well/field/cohort/patch identity in its canonical reference image name. Per-field files are unnecessary: quantification reads the tables once during preparation and sends each worker only its relevant object rows. Only the **main active method** is exported. The optional exploratory metric-comparison outputs are not silently mixed into it.

For **external QC import**, first set `registration_qc.enabled: false`. Use the **same DAPI reference round and identical original YOLO files** in both pipelines. The current example and standalone QC default to Round 2. For example:

```bash
python 4i/quantification/src/fouri_quantification/qc/run_4i_nucleus_segmentation_qc.py \
  --datapath /path/to/preprocessing/run \
  --reference-round 2 \
  --reference-contour-dir /path/to/preprocessing/run/segmentation/multiround_nuclei/dapi_r2/4i_phenix_myonis_segmentation_nuclei_r2/02_nuclei_deduplication/contours \
  --registration_metric ssim --registration-aggregation max \
  --registration-ssim-threshold 0.35 --workers 12 \
  --output-dir /path/to/qc_run
```

Keep your chosen QC method/thresholds; this integration does not change their calculations. `is_persistent` means **passes that configured method**, not an independently verified biological presence label. Hungarian mode requires matches in all selected rounds. Binary coverage needs `--min-other-rounds 7` to require all seven comparisons in an eight-round run; its existing default of 2 allows some rounds to fail. Registration mode uses the recorded image-change rule. The standalone script defaults to rounds 1–8; `--rounds 1 2 3 4 5 6 7 8 9 10 11 12` includes all 12, excluding the reference from scoring and including earlier rounds. `--n_images` samples patches: omit it for full coverage. In the older Slurm launcher, `QC_N_IMAGES=all` disables its existing 500-patch production sample; `REFERENCE_ROUND` controls its reference (default remains 2).

### Standalone 12-round SSIM figures

For the `4i_production_20260823_200037_f6ecb24` dataset, use `run_registration_qc_12rounds.sh` in the quantification folder. It runs DAPI re-registration followed by the existing SSIM filter, without marker feature quantification or H5AD assembly. The supplied original nucleus and MyHC contours identify **round 2** as their source frame. The launcher registers all 12 rounds, but **excludes rounds 7, 8 and 9 from nuclear filtering** because they have no nuclear markers. It also displays the MyHC contours over every registered marker acquisition, including those excluded rounds.

```bash
sbatch --export=ALL /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_registration_qc_12rounds.sh
```

The launcher selects **500 complete reference patches**, not 500 fields or 500 independent round images. Every selected patch is matched across all 12 DAPI rounds using its full spatial filename identity. This dataset uses well-level mosaic patch names, so field/cohort tokens are not required by standalone QC. Missing image rounds or missing reference contours are recorded as incomplete and excluded from the eligible sampling pool. The seed is 42; edit `--n_images` and `--seed` in the script or override them after its name. Sampling is uniform across eligible patches, not stratified by cell line/repeat.

`--rounds` lists rounds to register and display. `--filter-rounds 1 2 3 4 5 6 10 11 12` lists rounds used for nuclear decisions, including the reference. The reference is not scored, giving eight SSIM comparisons. Omit `--filter-rounds` to filter using every registered round. Excluded rounds contribute neither SSIM scores nor restrictions to the common valid-pixel area used for scoring, while their transforms remain available for marker overlays. Their image registration must still succeed.

The output folder is `<preprocessing run>/qc/ssim_ref02_exclude07-09_myotubes_random500_seed42/`:

| Folder/file | Contents |
| --- | --- |
| `registration_alignment/` | One before/after overlay panel per patch, showing every non-reference round. Reference DAPI is magenta, moving DAPI green, and overlap white. Titles record the applied shifts, registration correlation and filtering exclusions. |
| `plots/` | All 12 registered DAPI rounds with fixed reference contours: blue = kept, orange = high SSIM change, grey = edge/not evaluable. Rounds 7–9 are labelled `EXCLUDED FROM DAPI FILTERING`. |
| `registration_overlays/` | Eight scored SSIM change maps plus mean/max maps, on a fixed 0–1 scale. |
| `registration_scores/` | Nucleus mean/max scores and decisions with reference IDs. |
| `myotube_overlays/` | One overview per patch with original MyHC segmentation, reference DAPI and all registered marker channel images. Cyan outlines show myotubes. Rounds 7–9 are included and labelled `EXCLUDED FROM DAPI FILTERING`. |
| `round_summary.csv` | Per-patch/round shifts, correlation, `used_for_filtering` and filtering counts. Excluded rounds are recorded as not scored. |
| `nucleus_summary.csv`, `myonis_4i/` | Object decisions, edge flags and per-round QC evidence. |
| `assignments/`, `summary.json` | Detailed transforms, original object IDs, settings and the selected patch list. |

This creates five PNG panels per successful patch (up to 2,500 PNGs for 500 patches), grouping all marker channels into one myotube overview. It retains the current SSIM change threshold of 0.35 and pixelwise `max` aggregation across the eight scored comparisons, followed by the area mean within each nucleus. The per-round QC table retains all 12 rounds: excluded rounds have blank scores and `is_filter_round=0`. `--skip-dense-maps` avoids `.npz` archives; it does not change scoring. The launcher omits `--fast_mode` because fast mode disables all figures. It also omits exploratory metric comparisons to avoid extra figure sets. No source contours or images are altered.

`--myotube-contour-dir` enables myotube figures; `--myotube-acquisition measurement_02_555_myhc` identifies the image frame where those contours were created. If its round differs from the DAPI reference, the source round's DAPI transform shifts the contours into the reference frame once. Each marker image receives its own round's saved DAPI transform. These plots use the actual marker PNGs with percentile display contrast; they do not apply the DAPI brightness correction to marker intensities. Missing marker images are shown as unavailable and recorded in the assignments JSON.

For a small figure check, use a separate output directory:

```bash
sbatch --export=ALL /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_registration_qc_12rounds.sh \
  --n_images 10 --output-dir /absolute/path/to/ssim_10_patch_pilot
```

Add `--dry-run` to check matching and sample counts without loading images or generating figures. Both the preprocessing run root and its `png_8bit` folder are accepted as `--datapath`. Figures automatically expand to show every selected round.

Then configure quantification:

```yaml
nucleus_qc:
  directory: /path/to/qc_run/myonis_4i
  require_all: false
```

Relative directories resolve under `preprocessing.root`. `require_all: false` permits partial/sampled QC; absent or failed QC remains **not assessed**, with unknown persistence. `true` requires completed QC for every selected reference patch, including valid empty patches. It does not require every nucleus to pass QC. A null directory disables the input and leaves persistence unknown.

The join uses canonical reference image name + **original nonblank contour's actual file line number**, verified against the entire contour-file SHA-256 captured during QC. Blank lines do not shift IDs incorrectly. Absolute Nemo/Mac paths are provenance only, so moving files between mounts works. A mismatched round, grid, contour content/order, duplicate key, truncated table or corrupted export is rejected. Do not pass filtered/renumbered contours to quantification. QC input files participate in fingerprint/resume checks.

Nuclear `.obs` includes:

| Column | Meaning |
|---|---|
| `qc_pass` | Boolean convenience filter: evaluated, passes the active QC method, and not an edge |
| `is_persistent` | 1 = passes, 0 = evaluated failure, NaN = unknown/unevaluable; never cast directly to bool |
| `is_edge` | QC edge decision when available; otherwise the existing quantification geometry flag |
| `is_edge_geometry`, `qc_is_edge` | Original quantification edge flag and separate imported QC flag; the latter is 0/1/NaN |
| `qc_status` | `passed`, `failed`, `not_evaluable`, or `not_assessed` |
| `qc_exclusion_reason` | Edge, high change, insufficient coverage, missing match, unavailable QC, etc. |
| `qc_reference_round`, `qc_reference_id` | Source reference round and original one-based QC ID |
| `qc_filter_mode`, `qc_score_metric`, `qc_score_aggregation`, `qc_threshold` | Recorded active method/settings |
| `qc_score`, `qc_mean_change_score`, `qc_max_change_score`, `qc_valid_area_fraction` | Available numerical evidence; undefined values remain NaN |

Edge/empty/low-signal/insufficient-valid-area objects are **not** relabelled as disappeared nuclei. New failed or incomplete patches with an available reference retain rows with `qc_status=not_assessed`. Missing reference identities are recorded in the patch table; unmatched quantification nuclei also remain not assessed. `--summary-only` can regenerate exports from **new** saved QC results without source images or contours. Older assignments without captured checksums are marked `legacy_identity_missing` and require a fresh QC run before safe import.

```python
import anndata as ad
from fouri_quantification.access import get_nuclei

run = ad.read_h5ad('/path/to/myonis_4i_data.h5ad')
nuclei = get_nuclei(run)
good_nuclei = nuclei.loc[nuclei.qc_pass].copy()
round_evidence = run.uns['qc']['nucleus_qc']['rounds']
```

All morphology, intensity, texture, sigmoid values, parent assignments and original counts are computed on the full reference-object set. Structures inherit `parent_nucleus_qc_status` and `parent_nucleus_qc_pass`; direct cytoplasmic structures have no nuclear parent. Myotubes additionally report `n_nuclei_qc_passed`, `n_nuclei_qc_failed`, `n_nuclei_qc_not_evaluable` and `n_nuclei_qc_not_assessed`, which reconcile to `n_nuclei`.

**Filtering nuclear rows does not recalculate myotube metrics or filter linked `.uns` tables.** Existing myotube resident summaries and structure counts describe all reference nuclei. For analyses of passing nuclei only, aggregate the filtered nuclear/structure tables explicitly; direct myotube intensities still describe the original myotube measurement mask. The overlap-exclusion option also continues to consider all original peer objects, independent of these QC flags.

## Fixed metadata

Metadata comes from the canonical filename, including plate, repeat, physical well row/column, field, cohort, patch, patch row/column, acquisition and reference coordinates. Redundant field/cohort tokens must agree. Within a spatial patch, all acquisitions must share geometry.

| Physical row | Cell line | Condition |
|---:|---|---|
| 2 | NCRM1 | Control 1 |
| 3 | NCRM5 | Control 2 |
| 4 | K32Del | K32Del |
| 5 | R249W | R249W |
| 6 | L302P | L302P |
| 7 | L302P-Corr | L302P-Corr |

`sc_rep5-6` is plate 1 (repeats 5/6); `sc_rep7-8` is plate 2 (repeats 7/8). Repeat is parsed directly, not inferred from the column. `well-r04c02` maps to K32Del even when the earlier manifest's `cell_line_row` is 3; the older value is retained only as provenance. The fixed layout is in `configs/plate_layout.yaml` and packaged with the library. The resolved copy is embedded in every result.

IDs include dataset, plate, well, repeat, cohort, field, patch, segmentation version and source row. A registry captures source identity before filtering; existing post-filter local IDs are retained as compatibility columns. Unassigned structures retain all source metadata.

## Compatibility and coordinate scope

`src/fouri_quantification/compatibility` contains versioned, verbatim snapshots of selected functions from `generate_h5ad/utils`. Original source SHA-256 hashes are recorded in `source_manifest.json`. Function-body parity is tested against this checkout; changes to the original scripts do not silently alter this package. Only the aberration module's import path is adapted.

Preserved calculations include:

- Maximum nucleus/myotube intersection divided by nucleus area, strictly greater than the configured threshold; ties retain input order.
- Existing polygon area, perimeter, circularity and maximum vertex-distance `major_axis_length_px`.
- Existing metadata-based MyHC filtering and per-patch local-ID assignment after filtering.
- Existing sigmoid conversion, available-model mean ensemble, rasterization and all seven nuclear/myotube aberration feature names and missing-value rules.

Objects are processed in sorted spatial/file order and original YOLO row order; this order is saved. Rasterizer/software versions are recorded. The existing geometry routines receive manifest-derived field offsets, replacing CosMx grid assumptions. Extra overlap/ambiguity flags do not change legacy assignment decisions.

Measurements 1-8 and 9-12 are separate coordinate systems in the supplied data. The default config selects `m01_08`; it does not attach p-4EBP1 or later-cohort images to early-cohort objects. To analyze the second cohort, supply reference segmentations within that cohort and a separate selected run. Cross-cohort registration and patch-object merging are outside this pipeline's scope. Myotubes crossing patch/crop boundaries remain flagged fragments, not automatically reconstructed complete biological myotubes.

## Feature definitions

Intensity is measured on unclipped quantitative TIFFs. It includes mean, median, integrated intensity, standard deviation, p10/p90/p95, minimum, maximum, pixel count and finite-pixel coverage. Nonfinite masked pixels invalidate intensity statistics rather than becoming zeros. No automatic background correction, exposure normalization or marker positivity threshold is introduced.

All selected markers default to whole-nucleus, whole-myotube and myotube-cytoplasm measurements. Marker configuration can add nuclear rim/interior, local perinuclear cytoplasm and resident nuclear union. Cytoplasm excludes all nuclear masks. Perinuclear territories are disjoint within each myotube, assigned by nearest nuclear-mask distance with a finite radius. These are local sampling territories, not independently segmented cells. N/C ratios require finite values and a denominator above the configured floor.

Texture uses fixed acquisition-wide quantization bounds from `texture_range` in the marker config, otherwise the preprocessing clipping manifest, otherwise the explicit fallback range. This quantization affects texture only. Symmetric GLCMs pool valid within-mask pairs over four directions at configured pixel offsets. Contrast, homogeneity, energy, correlation and base-2 histogram entropy are saved, along with pair counts. Tiny masks and constant-marginal correlation are undefined (NaN). Texture is not run across exterior zero-filled bounding-box pixels.

SC35/53BP1 structures are assigned using intersection divided by structure area. They retain individual morphology/intensity/texture and source/parent IDs. Nuclear summaries include counts, original polygon-area statistics, equal-object mean within-parent intensity, mean and integrated intensity over the union of structure pixels inside the nucleus, and union occupancy. Overlapping structures count separately; in include mode shared pixels contribute once to union measurements, while exclude mode removes shared pixels before forming the union. Myotube ancestry is inherited from the assigned nucleus. BAF/cGAS can use `nucleus_then_myotube`; objects without a nuclear parent can be assigned directly to a myotube and measured in its cytoplasm. eMYHC uses the reference nucleus and MyHC myotube compartments for image measurements and has no structure source in the supplied configuration.

Myotubes retain direct image measurements, nuclei count, nuclear area fraction, area per nucleus, nuclear density, nearest-neighbour distances and resident-nucleus feature summaries. Resident summaries include mean, median, population SD and `n_valid`. Nuclear-structure and direct-cytoplasmic summaries are separate; their combined count counts each assigned object once. A valid empty segmentation gives count zero and undefined mean object intensity. Missing segmentation gives NaN counts. Existing aberration aggregation retains its stricter complete-assigned-nuclei rule.

The H5AD `feature_dictionary` provides names, formulas, units and storage locations. Additional morphology is separate from the four legacy values. Skeleton pixel count is named as such; it is not reported as physical skeleton length.

## H5AD access

Finalisation writes two H5AD files by default. `myonis_4i_data.h5ad` is the complete archive. `myonis_4i_data_measurements.h5ad` keeps **all nuclei and myotubes with their existing morphology, marker intensity, texture, nuclear aberration, structure counts/summaries, parent IDs, sample metadata, and compact field QC**. It omits individual SC35/53BP1/BAF/cGAS rows, contours, the duplicate object registry, relationship audits, per-round QC evidence and input manifests. The `feature_dictionary` in this view lists only retained features. Set `outputs.measurements_h5ad: false` to skip this second file in future runs.

To create the measurements file from an already completed run without repeating quantification or finalisation:

```bash
python /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/make_measurements_h5ad.py \
  --input "/absolute/path/to/run/myonis_4i_data.h5ad"
```

The default output is `/absolute/path/to/run/myonis_4i_data_measurements.h5ad`; `--output` overrides it. The script copies HDF5 groups without loading the full archive and replaces the destination only after its copy succeeds. Use the measurements file for nucleus- and myotube-level plots. Use the full file for individual structure objects, contour visualisations and detailed audit trails. The existing myotube resident-nucleus averages include **all assigned reference nuclei**; subsetting nuclei by `qc_pass` does not recompute those averages.

For separate files, add `--split`. This writes `myonis_4i_data_nuclei_measurements.h5ad` (nuclear `.obs`, `.X` and `.obsm`) and `myonis_4i_data_myotubes_measurements.h5ad` (one myotube per `.obs` row, with whole-myotube channel means in `.X`). `--split --compact-myotubes` keeps every direct myotube measurement and each resident-nucleus feature's `mean` and `n_valid`, but removes resident `median`/`std` and repeated `source_path`/`image_name` columns. The estimated sizes for this run are about 1.5 GiB for nuclei and around 2 GiB for compact myotubes; the actual sizes are printed after export. The standard combined file and unsplit myotube features remain unchanged.

The cluster launcher exports repeats 5, 6, 7 and 8 separately, in sequence. It accepts those flags after its name:

```bash
sbatch --export=ALL /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/run_quantification_generate_measurements_h5ad.sh --split --compact-myotubes
```

Use `--repeat N` with the Python exporter to export only one biological replicate from the completed archive. For example, this writes `myonis_4i_data_measurements_rep_5.h5ad` with only replicate 5 nuclei and myotubes, and filters the saved sample and field QC tables to replicate 5:

```bash
python /nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx/4i/quantification/make_measurements_h5ad.py \
  --input "/absolute/path/to/run/myonis_4i_data.h5ad" \
  --repeat 5
```

Change to `--repeat 6` for `myonis_4i_data_measurements_rep_6.h5ad`. It also works with `--split --compact-myotubes`; those outputs are named `myonis_4i_data_nuclei_measurements_rep_5.h5ad` and `myonis_4i_data_myotubes_measurements_rep_5.h5ad`. To limit the cluster launcher to specific replicates, edit its `for repeat in 5 6 7 8` list. Parent myotube IDs remain available in the nucleus file. The original full H5AD and the all-replicate measurements file are unchanged.

The nucleus file retains `parent_myotube_uid`, `qc_pass`, cell line, repeat, well, field and patch metadata. A nonempty parent ID identifies a nucleus assigned to a MyHC-segmented myotube under the configured overlap threshold; it is the operational myonucleus filter:

```python
import anndata as ad
nuclei = ad.read_h5ad('/absolute/path/myonis_4i_data_nuclei_measurements.h5ad')
is_myonucleus = nuclei.obs['parent_myotube_uid'].fillna('').ne('')
passing_myonuclei = nuclei.obs[is_myonucleus & nuclei.obs['qc_pass']]
```

This assignment is distinct from QC persistence. The myotube file uses the same object IDs as its row index, so the parent ID also joins each nucleus to its myotube.

```python
import anndata as ad
from fouri_quantification.access import get_nuclei, get_myotubes, get_structures, subset_by_myotubes

run = ad.read_h5ad('/absolute/path/myonis_4i_data.h5ad')
nuclei = get_nuclei(run)                       # Flat metadata + feature table
myotubes = get_myotubes(run)
sc35 = get_structures(run, 'sc35')
myotube_adata = get_myotubes(run, as_anndata=True)
counts = run.obsm['structure_features']['sc35__structures__count']
family = subset_by_myotubes(run, myotubes.index[:1])
```

| Location | Content |
|---|---|
| `.obs` | Nuclear identity, metadata, assignments and existing aberration columns |
| `.X`, `.var` | Whole-nucleus mean intensity by acquisition, with acquisition metadata |
| `.obsm['morphology_features']` | Existing four-column morphology matrix |
| `.obsm['additional_morphology_features']` | Separately named new morphology |
| `.obsm['intensity_features']` | Named intensity/compartment measurements |
| `.obsm['texture_features']` | Named texture measurements |
| `.obsm['structure_features']` | Structure summaries per nucleus |
| `.uns['myotubes']` | Complete myotube table |
| `.uns['structures']` | Individual structure tables keyed by source |
| `.uns['relationships']` | Accepted assignments and candidate evidence |
| `.uns['object_registry']`, `['geometry']` | Immutable source IDs, inclusion status and packed contours |
| `.uns['sample_metadata']`, `['metadata_mappings']` | Cell line/repeat and fixed plate layout |
| `.uns['feature_dictionary']` | Feature definitions and locations |
| `.uns['inputs']`, `['qc']`, `['provenance']` | Input index, quality/coverage, failures and exact configuration |

AnnData's observation axis is nuclei. Arbitrary nuclear slicing does not subset `.uns` or recompute myotube summaries. `subset_by_myotubes` selects complete parent families and keeps their valid aggregates; original run QC remains explicitly labelled. Other nuclear subsets should be analyzed through the returned tables with explicit reaggregation. Embedded child tables load into memory with ordinary `read_h5ad`; optional object-specific exports are useful for large datasets. `outputs.object_h5ad: true` writes conventional `exports/nuclei.h5ad` and `exports/myotubes.h5ad` views as well.

## Parallel jobs, recovery and reporting

Each worker owns one field/cohort, loads contours once and reads each marker image once per patch. Workers write independent standard-H5AD checkpoints; a single reducer validates parent references and counts, then writes the final file atomically. Input SHA-256 hashes, analysis configuration, numerical-library versions and package source hashes determine resume compatibility. Workers verify their inputs before processing or reusing a checkpoint; workers and the reducer reject changed code or software versions. Changing workers does not change numerical results. Changing inputs/configuration/code requires a new run. `hash_inputs: false` is an explicitly weaker size/mtime fingerprint option.

HDF5 files are constructed in local temporary storage (`SLURM_TMPDIR` when set), then copied to the destination and atomically renamed. Allow enough scratch space for a field checkpoint and, on the reducer, the combined archive. Wide tables retain standard AnnData encoding and can be opened with ordinary `read_h5ad`.

```bash
bash /absolute/path/4i/quantification/slurm/submit_quantification.sh \
  /absolute/path/dataset.yaml /absolute/path/new_run_directory
```

This older submitter prepares an input snapshot and dispatches one array task per field/cohort, followed by an `afterany` reducer/report job. Activate the environment before submission; adjust cluster resource directives as needed. Missing/failed tasks produce a partial result, failure records and a nonzero status. Individual prepared tasks can be rerun with `fouri-quantify worker --run-dir ... --task-index ...`, then `finalize` can regenerate the combined output. The recommended `run_quantification_batch.sh` above instead supports several workers per job, limits simultaneous jobs and assembles valid checkpoints with explicit exclusions by default.

The three-page core report (plus an exclusion appendix when needed) describes the actual run and cell-line/repeat coverage, each marker's measured compartments/features/availability, and how to read the output. It can be regenerated from the H5AD alone. A failed PDF is recorded without deleting measurements. `complete_with_omissions` means workers completed but some optional structure/image measurements were unavailable; it is distinct from a fully covered run.

## Tests

Run `python -m pytest tests` from this directory. Install MYONIS-VisTools as described above to exercise the rendering integration tests; those tests skip if the optional renderer is absent. Tests cover fixed metadata and redundant-token checks, path rebasing, invalid input pairing, exact legacy function-body preservation, strict overlap/tie behavior, union-based substructure summaries, masked texture, missing-versus-empty segmentations, sigmoid ensembles, H5AD round trips, ancestry/count consistency, parallel agreement, resume invalidation, and partial-job finalization. Additional tests cover an exact 50-field subset, seed/order stability, exported ROI equality, deliberately shifted masks, native sigmoid grids, rendering failures and recovery from deleted QC artifacts.

### Historical pilot (1 October 2026; before reference-only DAPI)

All 28 tests passed. The provided `dataset.pilot.yaml` completed one real field (four patches, 26 acquisitions) with 194 nuclei and 40 supplied myotube objects. Of these, 144 nuclei were assigned and 50 remained unassigned; saved parent counts reconciled exactly. The archive contains 758 nuclear feature columns across `.obsm`, plus the myotube table and feature dictionary. The three-page PDF was rendered and visually checked.

The pilot result is under `results/real_data_pilot_20261001_final/`. Its status is `complete_with_omissions`: optional structure paths and physical pixel size were not supplied, and aberration was explicitly disabled. Structure counts are NaN, not false zeros. The field computation took approximately 42 seconds in this environment, excluding input indexing and final output/report writing; this is not a full-dataset performance estimate. Local parallel execution and Slurm-style partition/finalization were tested; the Slurm scripts were syntax-checked but not submitted to a cluster.


### Historical integrated-registration pilot (1 October 2026; before reference-only DAPI)

`configs/dataset.registration-pilot.yaml` selects one real field with the round-2 reference and enables SSIM/max QC at change threshold 0.35, fast mode, and reference-DAPI and selected first-patch marker views. The historical saved pilot is `results/registration_qc_pilot_20261001/`.

- Four patches and all 26 selected acquisitions quantified; 178 nuclei and 40 myotubes retained, with 138 nuclei assigned to a myotube.
- 32 patch/round transforms saved (including four reference identities); largest absolute corrections were 2.35 pixels in y and 4.25 pixels in x.
- QC statuses: 129 passed, 9 failed, 40 not evaluable. These flags did not remove objects.
- Five MYONIS-VisTools views; all 335 ROI intensity audits passed. The three-page PDF was rendered and visually checked.
- Status `complete_with_omissions`: optional structure segmentation paths and physical calibration were not supplied; nuclear aberration was disabled. This is not a claim of complete substructure measurements.

The regression suite passed 67 existing tests and seven new registration tests, covering known patch-specific shifts, shared transforms across channels, contour-parent assignment, missing borders, sigmoid grids, fast/full score parity, failed registration, checksummed resume, and parallel processing in the later cohort. Slurm scripts were syntax-checked; no Nemo jobs were submitted for this local validation.


### Reference-only DAPI and lean-mode validation (1 October 2026)

A local run of `configs/dataset.registration-pilot.yaml` with timing enabled retained 178 nuclei and 40 myotubes across four patches. It quantified 19 acquisitions (18 markers plus reference DAPI), retained all 32 DAPI registration transforms, and generated four first-patch PNG views with no per-view TSV/JSON/NPZ companions. All 268 visual intensity checks passed; the PDF has three pages.

Against the earlier integrated-registration pilot, all 278 shared nuclear intensity columns, 253 shared nuclear texture columns and 3,336 shared numeric myotube columns matched exactly, including missing values. Non-reference DAPI columns and the former unconfigured eMYHC structure columns are intentionally absent.

Local timings were 75.8 s for indexing/input checks, 38.7 s for worker input/software verification, 14.2 s for registration/QC and 44.5 s for quantification. Within quantification, TIFF reads took 28.1 s, texture 3.9 s and the four visualisations 3.3 s. These are mounted-filesystem measurements, not a Nemo benchmark; optional substructure contours and aberration predictions were not supplied in this pilot. Use `--troubleshoot_timing` on the actual full configuration before drawing conclusions about cluster performance.

All 83 regression tests passed after these changes. Coverage includes reference-only DAPI with all-round registration, deterministic visualisation sampling independent of quantification, lean file output, diagnostic/normal measurement parity, timing exports, mask/texture crop equivalence, overlap exclusion, parallel workers and resume. The real pilot PDF was also rendered and visually checked.
