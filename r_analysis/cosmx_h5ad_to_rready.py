# --- Read, combine, and write R-ready h5ad files to SAVEPATH ---

import scanpy as sc
import anndata as ad
import pandas as pd

from pathlib import Path
from utils.preprocessing_utils import to_target_format

# Input paths
T5R5_NUCLEI_COMBINED_ADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t5r5/count_matrices/count_matrix_myotubes_greedy/combined_data/combined_nuclei.h5ad")
T5R5_MYOTUBE_COMBINED_ADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t5r5/count_matrices/count_matrix_myotubes_greedy/combined_data/combined_myotubes.h5ad")
T6R6_NUCLEI_COMBINED_ADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t6r6/count_matrices/count_matrix_myotubes_greedy/combined_data/combined_nuclei.h5ad")
T6R6_MYOTUBE_COMBINED_ADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t6r6/count_matrices/count_matrix_myotubes_greedy/combined_data/combined_myotubes.h5ad")

T5R5_METADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t5r5/nuclei_classification/resnet18_t5r5_simclr_severity_classes_e100_bf16_v2_finetuned/classifier_inference_predictions.csv")
T6R6_METADATA_PATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t6r6/nuclei_classification/resnet18_t5r5_simclr_severity_classes_e100_bf16_v2_finetuned/classifier_inference_predictions.csv")

prefix = "greedy_"

# Set this to your target output folder (ideally on a volume with free space)
SAVEPATH = Path("/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/r_ready")
SAVEPATH.mkdir(parents=True, exist_ok=True)

# Read
t5r5_nuclei_combined_adata = sc.read_h5ad(T5R5_NUCLEI_COMBINED_ADATA_PATH)
t5r5_myotube_combined_adata = sc.read_h5ad(T5R5_MYOTUBE_COMBINED_ADATA_PATH)
t6r6_nuclei_combined_adata = sc.read_h5ad(T6R6_NUCLEI_COMBINED_ADATA_PATH)
t6r6_myotube_combined_adata = sc.read_h5ad(T6R6_MYOTUBE_COMBINED_ADATA_PATH)

# Add slide labels and force unique cell IDs before concat
for adata_obj, slide in [
    (t5r5_nuclei_combined_adata, "T5R5"),
    (t6r6_nuclei_combined_adata, "T6R6"),
    (t5r5_myotube_combined_adata, "T5R5"),
    (t6r6_myotube_combined_adata, "T6R6"),
]:
    adata_obj.obs["Slide Name"] = slide
    adata_obj.obs_names = slide + "_" + adata_obj.obs_names.astype(str)
    adata_obj.obs_names_make_unique()
    adata_obj.var_names_make_unique()

# Concat
nuclei_combined_adata = ad.concat(
    [t5r5_nuclei_combined_adata, t6r6_nuclei_combined_adata],
    join="outer"
)
myotube_combined_adata = ad.concat(
    [t5r5_myotube_combined_adata, t6r6_myotube_combined_adata],
    join="outer"
)

# Final uniqueness pass
nuclei_combined_adata.obs_names_make_unique()
nuclei_combined_adata.var_names_make_unique()
myotube_combined_adata.obs_names_make_unique()
myotube_combined_adata.var_names_make_unique()

# Filter + add myonucleus flag
filtered_nuclei_adata = nuclei_combined_adata[nuclei_combined_adata.obs["is_edge"] == False].copy()
filtered_nuclei_adata.obs["is_myonucleus"] = (filtered_nuclei_adata.obs["myotube_id"] != -1).astype(int)
filtered_nuclei_adata.obs_names_make_unique()
filtered_nuclei_adata.var_names_make_unique()

# Write outputs to SAVEPATH
nuc_out = SAVEPATH / f"{prefix}nuclei_combined.rready.h5ad"
myo_out = SAVEPATH / f"{prefix}myotube_combined.rready.h5ad"
fnu_out = SAVEPATH / f"{prefix}filtered_nuclei.rready.h5ad"

nuclei_combined_adata.write_h5ad(nuc_out, compression="gzip")
myotube_combined_adata.write_h5ad(myo_out, compression="gzip")
filtered_nuclei_adata.write_h5ad(fnu_out, compression="gzip")

print("Wrote:", nuc_out)
print("Wrote:", myo_out)
print("Wrote:", fnu_out)

# Read in and process classifier metadata
t5r5_df = pd.read_csv(T5R5_METADATA_PATH)
t6r6_df = pd.read_csv(T6R6_METADATA_PATH)

t5r5_fmt = to_target_format(t5r5_df, "t5r5", include_labels=True)
t6r6_fmt = to_target_format(t6r6_df, "t6r6", include_labels=True)

final_df = pd.concat([t5r5_fmt, t6r6_fmt], ignore_index=True)
print(len(t5r5_fmt), len(t6r6_fmt), len(final_df))

final_df.to_csv(SAVEPATH / f"{prefix}classifier_metadata.csv", index=False)
print("Wrote:", SAVEPATH / f"{prefix}classifier_metadata.csv")
