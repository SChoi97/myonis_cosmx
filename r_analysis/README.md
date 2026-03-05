# Analysis of the relationship between nuclear morphology and transcriptome using spatial transcriptomics

This repository contains code for processing of data from the Nanostring CosMx assay. \
The repository is split into 2 main sections: `generate_h5ad` and `r_analysis`

## generate_h5ad

This contains scripts for processing of segmented nuclei and myotubes AFTER running segmentation models. The main output of this folder are `h5ad` files used for downstream analysis.

## r_analysis

This contains scripts for conversion of `h5ad` to `rds` format and for downstream statistical analysis such as DGE using `edgeR`.