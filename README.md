# Analysis of the relationship between nuclear morphology and transcriptome using spatial transcriptomics

<p align="center">
  <img src="images/myonis_cosmx_logo.jpg" alt="Logo" width="500">
</p>

This repository contains code for processing of data from the Nanostring CosMx assay. \
The main analysis sections are described below.

## generate_h5ad

This contains scripts for processing of segmented nuclei and myotubes AFTER running segmentation models. The main output of this folder are `h5ad` files used for downstream analysis.

## r_analysis

This contains scripts for conversion of `h5ad` to `rds` format and for downstream statistical analysis such as DGE using `edgeR`.

## 4i quantification

[`4i/quantification`](4i/quantification/README.md) measures multiplexed IF images at nucleus, myotube and substructure level, with stable object/parent IDs and cell-line/repeat metadata in `myonis_4i_data.h5ad`.

The example workflow first re-registers each patch's DAPI rounds to its reference DAPI and performs SSIM nucleus QC (change threshold 0.35; fast mode enabled). It then reuses each patch/round translation for its marker images and source contours before quantification. QC flags are retained for downstream filtering; nuclei are not deleted. The [quantification guide](4i/quantification/README.md) explains every YAML section, measuring markers such as eMYHC without their own segmentation, saved transforms and QC outputs, and step-by-step Nemo/Slurm submission.
