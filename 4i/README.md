# MYONIS 4i

The multiplexed-immunofluorescence workflow lives in this folder. Existing image preparation is followed by object quantification:

```text
4i/
├── illumination_correction/   # Existing illumination workflow
├── preprocessing/             # Existing registration, patches and manifests
└── quantification/
    ├── README.md              # Installation, paths, features and H5AD access
    ├── pyproject.toml
    ├── environment.yml
    ├── configs/
    │   ├── dataset.example.yaml
    │   ├── dataset.pilot.yaml
    │   ├── dataset.visualisation-pilot.yaml
    │   ├── markers.yaml
    │   └── plate_layout.yaml
    ├── src/fouri_quantification/
    │   ├── inputs/            # Filename metadata and input validation
    │   ├── compatibility/     # Preserved CosMx calculations
    │   ├── objects/           # Geometry, compartments and parent relationships
    │   ├── features/          # Intensity, morphology, texture and summaries
    │   ├── outputs/           # Linked H5AD tables and aggregation
    │   ├── reporting/         # Three-page PDF and QC overlays
    │   ├── access.py          # Nuclear/myotube/structure access helpers
    │   ├── pipeline.py        # Parallel jobs, checkpoints and resume
    │   └── cli.py
    ├── slurm/                 # Array workers and reducer submission
    ├── examples/
    ├── tests/
    └── results/               # Generated run folders; ignored by Git
```

See [the quantification guide](quantification/README.md) for configuration and execution. A completed run contains:

```text
run_directory/
├── myonis_4i_data.h5ad        # All linked measurements and provenance
├── run_summary.pdf           # Run, marker coverage and data-access guide
├── run_manifest.json         # Completion state, versions and checksums
├── config/                   # Resolved dataset and marker settings
├── inputs/                   # Validated index and source fingerprints
├── tasks/                    # One task per selected field/cohort
├── checkpoints/              # Independently resumable field results
├── qc_overlays/              # Nuclear/myotube assignment examples
├── qc_visualisations/        # MYONIS-VisTools figures, intensity audits, exact ROIs
│   ├── index.html            # Offline field/marker browser
│   ├── index.tsv
│   └── <field-task-id>/patch_0001/
│       ├── <marker>.png
│       ├── <marker>.tsv
│       ├── <marker>.npz
│       └── <marker>.json
├── logs/                     # Failures and Slurm output
└── exports/                  # Optional nucleus/myotube AnnData views
```

Cell line and condition use the physical well row; repeat and plate are parsed from the source filename. Global IDs explicitly include repeat. The supplied early and late registration cohorts require their own compatible reference segmentations.

The example quantification YAML limits the run to 50 reproducibly selected field/cohort units (`selection.max_fields`) and enables detailed visual checks for every selected patch and marker. Set the limit to `null` for all fields. The small visualisation pilot quantifies one field and renders its first patch across all marker acquisitions.
