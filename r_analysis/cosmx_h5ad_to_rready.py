# --- Read, combine, and write R-ready h5ad files to SAVEPATH ---

import argparse
import scanpy as sc
import anndata as ad
import pandas as pd

from pathlib import Path
from utils.preprocessing_utils import to_target_format

def parse_args():
    parser = argparse.ArgumentParser(
        description="Read, combine, and write R-ready h5ad files plus merged classifier metadata."
    )
    parser.add_argument("--t5r5_nuclei_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t5r5_myotube_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t6r6_nuclei_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t6r6_myotube_combined_adata_path", type=Path, required=True)
    parser.add_argument("--t5r5_metadata_path", type=Path, required=True)
    parser.add_argument("--t6r6_metadata_path", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--savepath", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.savepath.mkdir(parents=True, exist_ok=True)

    # Read
    t5r5_nuclei_combined_adata = sc.read_h5ad(args.t5r5_nuclei_combined_adata_path)
    t5r5_myotube_combined_adata = sc.read_h5ad(args.t5r5_myotube_combined_adata_path)
    t6r6_nuclei_combined_adata = sc.read_h5ad(args.t6r6_nuclei_combined_adata_path)
    t6r6_myotube_combined_adata = sc.read_h5ad(args.t6r6_myotube_combined_adata_path)

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

    # Write outputs to savepath
    nuc_out = args.savepath / f"{args.prefix}nuclei_combined.rready.h5ad"
    myo_out = args.savepath / f"{args.prefix}myotube_combined.rready.h5ad"
    fnu_out = args.savepath / f"{args.prefix}filtered_nuclei.rready.h5ad"

    nuclei_combined_adata.write_h5ad(nuc_out, compression="gzip")
    myotube_combined_adata.write_h5ad(myo_out, compression="gzip")
    filtered_nuclei_adata.write_h5ad(fnu_out, compression="gzip")

    print("Wrote:", nuc_out)
    print("Wrote:", myo_out)
    print("Wrote:", fnu_out)

    # Read in and process classifier metadata
    t5r5_df = pd.read_csv(args.t5r5_metadata_path)
    t6r6_df = pd.read_csv(args.t6r6_metadata_path)

    t5r5_fmt = to_target_format(t5r5_df, "t5r5", include_labels=True)
    t6r6_fmt = to_target_format(t6r6_df, "t6r6", include_labels=True)

    final_df = pd.concat([t5r5_fmt, t6r6_fmt], ignore_index=True)
    print(len(t5r5_fmt), len(t6r6_fmt), len(final_df))

    metadata_out = args.savepath / f"{args.prefix}classifier_metadata.csv"
    final_df.to_csv(metadata_out, index=False)
    print("Wrote:", metadata_out)


if __name__ == "__main__":
    main()
