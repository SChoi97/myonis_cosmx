import argparse
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is importable, so we can import utils
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cosmx.utils.cosmx_visualisation_utils import (
    numericalSort,
    create_fiji_lut,
    unpack_object_contours,
    visualise_labels,
)


def process_fov(fov_num: int, args) -> None:
    fov_str = str(fov_num).zfill(args.fov_pad)

    # Resolve I/O paths
    anndata_dir = Path(args.anndata_dir)
    dapi_dir = Path(args.dapi_dir)
    myhc_dir = Path(args.myhc_dir)
    out_base = Path(args.out_dir)

    out_dir = out_base / f"F{fov_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nuclei_adata_path = anndata_dir / f"F{fov_str}_nuclei.h5ad"
    myotube_adata_path = anndata_dir / f"F{fov_str}_myotubes.h5ad"

    if not nuclei_adata_path.exists() or not myotube_adata_path.exists():
        print(f"[WARN] F{fov_str}: Missing anndata files. nuclei={nuclei_adata_path.exists()}, myotube={myotube_adata_path.exists()} -> skipping")
        return

    # Locate patches for this FOV
    dapi_image_path_list = sorted(list(dapi_dir.glob(f"*F{fov_str}*.png")), key=numericalSort)
    myhc_image_path_list = sorted(list(myhc_dir.glob(f"*F{fov_str}*.png")), key=numericalSort)

    expected_patches = args.rows * args.cols
    if len(dapi_image_path_list) < expected_patches or len(myhc_image_path_list) < expected_patches:
        print(f"[WARN] F{fov_str}: Not enough patches (DAPI={len(dapi_image_path_list)}, MyHC={len(myhc_image_path_list)} expected={expected_patches}) -> skipping")
        return

    print(f"[INFO] F{fov_str}: Found {len(dapi_image_path_list)} DAPI and {len(myhc_image_path_list)} MyHC patches")

    # Load anndata and contours
    nuclei_adata = ad.read_h5ad(nuclei_adata_path)
    myotube_adata = ad.read_h5ad(myotube_adata_path)

    nuclei_contours_dict = nuclei_adata.uns["Object Contours"]
    myotube_contours_dict = myotube_adata.uns["Object Contours"]
    nuclei_contours, nuclei_contour_offsets = unpack_object_contours(nuclei_contours_dict)
    myotube_contours, myotube_contour_offsets = unpack_object_contours(myotube_contours_dict)

    dapi_clip_low, dapi_clip_high = args.dapi_clip
    myhc_clip_low, myhc_clip_high = args.myhc_clip
    dapi_lw = args.dapi_linewidth
    myhc_lw = args.myhc_linewidth

    # Two modes: full_fov (stitched) or patch (per-patch figures)
    mode = args.visualisation_mode
    patch_size = args.patch_size
    n_rows = args.rows
    n_cols = args.cols

    # Prepare output directories
    if getattr(args, "save_pngs_individually", False):
        plots_dir = out_dir / "plots"
        images_dir = out_dir / "images"
        plots_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
    else:
        plots_dir = out_dir
        images_dir = out_dir

    if mode == "full_fov":
        # Stitch all patches together into full FOV image
        full_height = patch_size * n_rows
        full_width = patch_size * n_cols

        dapi_full = np.zeros((full_height, full_width), dtype=np.uint16)
        myhc_full = np.zeros((full_height, full_width), dtype=np.uint16)

        for target_patch in range(expected_patches):
            row = target_patch // n_cols
            col = target_patch % n_cols
            y_start = row * patch_size
            x_start = col * patch_size
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            dapi_patch = np.array(imageio.imread(dapi_image_path_list[target_patch]))
            myhc_patch = np.array(imageio.imread(myhc_image_path_list[target_patch]))

            dapi_full[y_start:y_end, x_start:x_end] = dapi_patch
            myhc_full[y_start:y_end, x_start:x_end] = myhc_patch

        # Build composite "images" figure (DAPI, MyHC, merged) and save
        dapi_full_modified = np.clip(dapi_full, dapi_clip_low, dapi_clip_high)
        myhc_full_modified = np.clip(myhc_full, myhc_clip_low, myhc_clip_high)

        # Normalize for merged RGB
        dapi_den = max(1e-9, float(dapi_full_modified.max() - dapi_full_modified.min()))
        myhc_den = max(1e-9, float(myhc_full_modified.max() - myhc_full_modified.min()))
        dapi_norm = (dapi_full_modified - dapi_full_modified.min()) / dapi_den
        myhc_norm = (myhc_full_modified - myhc_full_modified.min()) / myhc_den

        merged_rgb = np.zeros((dapi_norm.shape[0], dapi_norm.shape[1], 3), dtype=np.float32)
        merged_rgb[:, :, 0] = myhc_norm
        merged_rgb[:, :, 1] = np.clip(dapi_norm + myhc_norm, 0, 1)
        merged_rgb[:, :, 2] = np.clip(dapi_norm + myhc_norm, 0, 1)

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(dapi_full_modified, cmap=create_fiji_lut("cyan"))
        ax[0].set_axis_off()
        ax[0].grid(False)

        ax[1].imshow(myhc_full_modified, cmap=create_fiji_lut("gray"))
        ax[1].set_axis_off()
        ax[1].grid(False)

        ax[2].imshow(merged_rgb)
        ax[2].set_axis_off()
        ax[2].grid(False)

        plt.tight_layout()
        (plots_dir / f"F{fov_str}_images.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / f"F{fov_str}_images.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # Save individual panels if requested
        if getattr(args, "save_pngs_individually", False):
            merged_rgb_u8 = (np.clip(merged_rgb, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(images_dir / "dapi_fluo.png", dapi_full_modified.astype(np.uint16))
            imageio.imwrite(images_dir / "myhc_fluo.png", myhc_full_modified.astype(np.uint16))
            imageio.imwrite(images_dir / "merged_fluo.png", merged_rgb_u8)

        # Create overlay visualizations for entire FOV
        nuclei_overlay_full = np.zeros((full_height, full_width, 3), dtype=np.uint8)
        myotube_overlay_full = np.zeros((full_height, full_width, 3), dtype=np.uint8)
        combined_overlay_full = np.zeros((full_height, full_width, 3), dtype=np.uint8)
        assignment_overlay_full = np.zeros((full_height, full_width, 3), dtype=np.uint8)

        for target_patch in range(expected_patches):
            row = target_patch // n_cols
            col = target_patch % n_cols
            y_start = row * patch_size
            x_start = col * patch_size
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            dapi_image = np.array(imageio.imread(dapi_image_path_list[target_patch]))
            myhc_image = np.array(imageio.imread(myhc_image_path_list[target_patch]))
            # scale MyHC to uint8 for nicer overlays
            myhc_image = (np.clip(myhc_image, myhc_clip_low, myhc_clip_high) / 2**16 * 255).astype(np.uint8)

            nuclei_patch_adata = nuclei_adata[nuclei_adata.obs["patch_idx"] == target_patch]
            object_ids_nuclei = nuclei_patch_adata.obs.index.astype(int)
            nuclei_patch_contours = [nuclei_contours[i] for i in object_ids_nuclei]

            myotube_patch_adata = myotube_adata[myotube_adata.obs["patch_idx"] == target_patch]
            object_ids_myotube = myotube_patch_adata.obs.index.astype(int)
            myotube_patch_contours = [myotube_contours[i] for i in object_ids_myotube]

            try:
                myotube_assignment_label_list = nuclei_patch_adata.obs["myotube_id"].tolist()
            except KeyError:
                myotube_assignment_label_list = None

            nuclei_overlay_patch = visualise_labels(
                image=dapi_image,
                contours=nuclei_patch_contours,
                alpha=1.0,
                linewidth=dapi_lw,
                image_edge_outline=False,
                image_edge_border_width=5,
            )

            myotube_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=myotube_patch_contours,
                alpha=1.0,
                linewidth=myhc_lw,
                image_edge_outline=False,
                image_edge_border_width=5,
            )

            combined_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=nuclei_patch_contours,
                alpha=2,
                linewidth=dapi_lw,
                myotube_contours=myotube_patch_contours,
                myotube_linewidth=myhc_lw,
                myotube_alpha=0.5,
                myotube_line_color=(211, 211, 211),
                myotube_fill_color=(200, 200, 200),
                myotube_color="#6E6E6E",
                nuclei_color="#00A0F7",
                image_edge_outline=False,
                image_edge_border_width=5,
            )

            assignment_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=nuclei_patch_contours,
                alpha=2,
                linewidth=dapi_lw,
                myotube_contours=myotube_patch_contours,
                myotube_linewidth=myhc_lw,
                myotube_alpha=0.3,
                myotube_line_color=(120, 120, 120),
                myotube_fill_color=(200, 200, 200),
                assignment_labels=myotube_assignment_label_list,
                image_edge_outline=False,
                image_edge_border_width=5,
            )

            nuclei_overlay_full[y_start:y_end, x_start:x_end] = nuclei_overlay_patch
            myotube_overlay_full[y_start:y_end, x_start:x_end] = myotube_overlay_patch
            combined_overlay_full[y_start:y_end, x_start:x_end] = combined_overlay_patch
            assignment_overlay_full[y_start:y_end, x_start:x_end] = assignment_overlay_patch

        fig2, ax2 = plt.subplots(1, 4, figsize=(40, 10))
        plt.subplots_adjust(wspace=0.02, hspace=0)
        overlays = [nuclei_overlay_full, myotube_overlay_full, combined_overlay_full, assignment_overlay_full]
        for a, img in zip(ax2, overlays):
            a.imshow(img)
            a.axis("off")

        plt.tight_layout()
        (plots_dir / f"F{fov_str}_annotations.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / f"F{fov_str}_annotations.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig2)

        # Save individual overlay panels if requested
        if getattr(args, "save_pngs_individually", False):
            imageio.imwrite(images_dir / "dapi_overlay.png", nuclei_overlay_full)
            imageio.imwrite(images_dir / "myhc_overlay.png", myotube_overlay_full)
            imageio.imwrite(images_dir / "combined_overlay.png", combined_overlay_full)
            imageio.imwrite(images_dir / "instance_overlay.png", assignment_overlay_full)

    else:  # mode == "patch"
        # Per-patch outputs: two figures per patch (images + annotations)
        patch_index_width = max(2, len(str(expected_patches - 1)))

        for target_patch in range(expected_patches):
            # Read patch images
            dapi_patch = np.array(imageio.imread(dapi_image_path_list[target_patch]))
            myhc_patch = np.array(imageio.imread(myhc_image_path_list[target_patch]))

            # Prepare "images" figure (DAPI, MyHC, merged) for this patch
            dapi_patch_mod = np.clip(dapi_patch, dapi_clip_low, dapi_clip_high)
            myhc_patch_mod = np.clip(myhc_patch, myhc_clip_low, myhc_clip_high)

            dapi_den = max(1e-9, float(dapi_patch_mod.max() - dapi_patch_mod.min()))
            myhc_den = max(1e-9, float(myhc_patch_mod.max() - myhc_patch_mod.min()))
            dapi_norm = (dapi_patch_mod - dapi_patch_mod.min()) / dapi_den
            myhc_norm = (myhc_patch_mod - myhc_patch_mod.min()) / myhc_den

            merged_rgb = np.zeros((dapi_norm.shape[0], dapi_norm.shape[1], 3), dtype=np.float32)
            merged_rgb[:, :, 0] = myhc_norm
            merged_rgb[:, :, 1] = np.clip(dapi_norm + myhc_norm, 0, 1)
            merged_rgb[:, :, 2] = np.clip(dapi_norm + myhc_norm, 0, 1)

            fig_p, ax_p = plt.subplots(1, 3, figsize=(18, 6))
            ax_p[0].imshow(dapi_patch_mod, cmap=create_fiji_lut("cyan"))
            ax_p[0].set_axis_off(); ax_p[0].grid(False)
            ax_p[1].imshow(myhc_patch_mod, cmap=create_fiji_lut("gray"))
            ax_p[1].set_axis_off(); ax_p[1].grid(False)
            ax_p[2].imshow(merged_rgb)
            ax_p[2].set_axis_off(); ax_p[2].grid(False)
            plt.tight_layout()
            fname_images = plots_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_images.png"
            plt.savefig(fname_images, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_p)

            if getattr(args, "save_pngs_individually", False):
                merged_rgb_u8 = (np.clip(merged_rgb, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_dapi_fluo.png", dapi_patch_mod.astype(np.uint16))
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_myhc_fluo.png", myhc_patch_mod.astype(np.uint16))
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_merged_fluo.png", merged_rgb_u8)

            # Prepare "annotations" figure (4 overlays) for this patch
            dapi_image = dapi_patch
            myhc_image = (np.clip(myhc_patch, myhc_clip_low, myhc_clip_high) / 2**16 * 255).astype(np.uint8)

            nuclei_patch_adata = nuclei_adata[nuclei_adata.obs["patch_idx"] == target_patch]
            object_ids_nuclei = nuclei_patch_adata.obs.index.astype(int)
            nuclei_patch_contours = [nuclei_contours[i] for i in object_ids_nuclei]

            myotube_patch_adata = myotube_adata[myotube_adata.obs["patch_idx"] == target_patch]
            object_ids_myotube = myotube_patch_adata.obs.index.astype(int)
            myotube_patch_contours = [myotube_contours[i] for i in object_ids_myotube]

            try:
                myotube_assignment_label_list = nuclei_patch_adata.obs["myotube_id"].tolist()
            except KeyError:
                myotube_assignment_label_list = None

            nuclei_overlay_patch = visualise_labels(
                image=dapi_image,
                contours=nuclei_patch_contours,
                alpha=1.0,
                linewidth=dapi_lw,
            )
            myotube_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=myotube_patch_contours,
                alpha=1.0,
                linewidth=myhc_lw,
            )
            combined_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=nuclei_patch_contours,
                alpha=2,
                linewidth=dapi_lw,
                myotube_contours=myotube_patch_contours,
                myotube_linewidth=myhc_lw,
                myotube_alpha=0.5,
                myotube_line_color=(211, 211, 211),
                myotube_fill_color=(200, 200, 200),
                myotube_color="#6E6E6E",
                nuclei_color="#00A0F7",
            )
            assignment_overlay_patch = visualise_labels(
                image=myhc_image,
                contours=nuclei_patch_contours,
                alpha=2,
                linewidth=dapi_lw,
                myotube_contours=myotube_patch_contours,
                myotube_linewidth=myhc_lw,
                myotube_alpha=0.3,
                myotube_line_color=(120, 120, 120),
                myotube_fill_color=(200, 200, 200),
                assignment_labels=myotube_assignment_label_list,
            )

            fig_a, ax_a = plt.subplots(1, 4, figsize=(28, 7))
            plt.subplots_adjust(wspace=0.02, hspace=0)
            overlays = [nuclei_overlay_patch, myotube_overlay_patch, combined_overlay_patch, assignment_overlay_patch]
            for a, img in zip(ax_a, overlays):
                a.imshow(img); a.axis("off")
            plt.tight_layout()
            fname_ann = plots_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_annotations.png"
            plt.savefig(fname_ann, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_a)

            if getattr(args, "save_pngs_individually", False):
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_dapi_overlay.png", nuclei_overlay_patch)
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_myhc_overlay.png", myotube_overlay_patch)
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_combined_overlay.png", combined_overlay_patch)
                imageio.imwrite(images_dir / f"F{fov_str}_P{target_patch:0{patch_index_width}d}_instance_overlay.png", assignment_overlay_patch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch CosMx patch images for FOVs and generate overlay visualizations."
    )
    parser.add_argument("--anndata-dir", required=True, type=Path, help="Directory containing F{FOV}_nuclei.h5ad and F{FOV}_myotubes.h5ad")
    parser.add_argument("--dapi-dir", required=True, type=Path, help="Directory containing DAPI patch PNGs")
    parser.add_argument("--myhc-dir", required=True, type=Path, help="Directory containing MyHC patch PNGs")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory base; per-FOV subdirs will be created")
    parser.add_argument("--fov-range", required=True, nargs=2, type=int, metavar=("START", "END"), help="Process FOVs in [START, END) with zero-padded IDs")
    parser.add_argument("--patch-size", default=1024, type=int, help="Patch size in pixels (default: 1024)")
    parser.add_argument("--rows", default=4, type=int, help="Number of patch rows per FOV (default: 4)")
    parser.add_argument("--cols", default=4, type=int, help="Number of patch cols per FOV (default: 4)")
    parser.add_argument("--dapi-clip", default=(20, 220), nargs=2, type=int, metavar=("LOW", "HIGH"), help="Clip range for DAPI channel (default: 20 220)")
    parser.add_argument("--myhc-clip", default=(1000, 15000), nargs=2, type=int, metavar=("LOW", "HIGH"), help="Clip range for MyHC channel (default: 1000 15000)")
    parser.add_argument("--fov-pad", default=5, type=int, help="Zero-pad width for FOV IDs (default: 5)")
    parser.add_argument("--dpi", default=300, type=int, help="DPI for saved figures (default: 300)")
    parser.add_argument("--visualisation_mode", choices=["full_fov", "patch"], default="full_fov", help="full_fov (stitched) or patch (per-patch outputs)")
    parser.add_argument("--dapi_linewidth", default=5, type=int, help="Line width for DAPI/nuclei outlines")
    parser.add_argument("--myhc_linewidth", default=7, type=int, help="Line width for MyHC/myotube outlines")
    parser.add_argument("--n_workers", default=1, type=int, help="Number of parallel workers across FOVs (default: 1)")
    parser.add_argument("--save_pngs_individually", action="store_true", help="If set, save each panel as an individual PNG under images/, and multi-panel figures under plots/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start, end = args.fov_range
    if end < start:
        print(f"[WARN] Invalid FOV range [{start}, {end}). Nothing to do.")
        sys.exit(0)

    total = max(0, end - start)
    if args.n_workers and args.n_workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = [ex.submit(process_fov, fov_num, args) for fov_num in range(start, end)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="FOVs"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] Worker exception: {e}")
    else:
        for fov_num in tqdm(range(start, end), desc="FOVs"):
            try:
                process_fov(fov_num, args)
            except Exception as e:
                fov_str = str(fov_num).zfill(args.fov_pad)
                print(f"[ERROR] F{fov_str}: {e}")