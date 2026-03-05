import argparse
from pathlib import Path

import os
import cv2
import numpy as np
import anndata as ad
import imageio.v2 as imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from utils.cosmx_visualisation_utils import unpack_object_contours
from utils.cosmx_single_nuclei_crop_utils import (
    ensure_rgb,
    generate_aligned_crop,
    safe_str,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate centered, axis-aligned single-nucleus crops from an .h5ad "
            "produced by cosmx_generate_count_matrix_h5ad.py."
        )
    )
    parser.add_argument("--h5ad_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--image_path", required=True, help="Directory containing original patch images (.png)")
    parser.add_argument("--savepath", required=True, help="Directory to save crops (will create raw/ and mask/)")
    parser.add_argument(
        "--filter_edge_objects",
        action="store_true",
        help="If set, drop objects with obs['is_edge']==True before processing",
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        default=512,
        help="Output square canvas size (pixels) for centered/aligned crops (default: 512)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers when h5ad_path is a directory (default: 1 = serial)",
    )
    parser.add_argument(
        "--size_filter",
        type=int,
        default=0,
        help="If >0, skip objects with contour area smaller than this pixel threshold",
    )
    parser.add_argument(
        "--downsampled_image_size",
        type=int,
        default=0,
        help=(
            "If >0, also write bilinearly resized crops (raw/mask) at this square "
            "size into raw_<size>/ and mask_<size>/ alongside the full-size outputs"
        ),
    )
    return parser.parse_args()


def _process_h5ad_file(
    h5ad_path: Path,
    image_dir: Path,
    raw_dir: Path,
    mask_dir: Path,
    canvas_size: int,
    filter_edge_objects: bool,
    size_filter: int,
    downsampled_size: int,
) -> Tuple[int, int, str]:
    """Process a single h5ad file and return (processed, skipped, name)."""
    print(f"Loading {h5ad_path}…", flush=True)
    data = ad.read_h5ad(h5ad_path)

    contours_list, offsets_list = unpack_object_contours(data.uns["Object Contours"])
    if len(contours_list) != len(data.obs):
        raise ValueError(
            f"Contours count ({len(contours_list)}) does not match obs rows ({len(data.obs)}) in {h5ad_path}."
        )

    obs = data.obs.copy()
    obs["__contour"] = contours_list
    obs["__offset"] = offsets_list

    if filter_edge_objects and "is_edge" in obs.columns:
        before = len(obs)
        obs = obs[~obs["is_edge"].astype(bool)]
        print(f"{h5ad_path.name}: filtered edge objects {before} -> {len(obs)}", flush=True)
    elif filter_edge_objects:
        print(f"{h5ad_path.name}: warning 'is_edge' column not found; skipping edge filter.", flush=True)

    processed = 0
    skipped_missing_image = 0
    for img_name, group in tqdm(obs.groupby("image_name"), desc=h5ad_path.name):
        img_path = image_dir / f"{img_name}.png"
        if not img_path.exists():
            skipped_missing_image += len(group)
            print(f"{h5ad_path.name}: missing image {img_path}, skipping {len(group)} objects.", flush=True)
            continue

        img = imageio.imread(img_path)
        img = ensure_rgb(img)

        for _, row in group.iterrows():
            contour_local = row["__contour"]
            if contour_local is None or len(contour_local) < 3:
                continue

            if size_filter and size_filter > 0:
                contour_arr = np.asarray(contour_local, dtype=np.float32).reshape(-1, 2)
                if cv2.contourArea(contour_arr) < float(size_filter):
                    continue

            try:
                raw_crop, mask_crop = generate_aligned_crop(
                    img, contour_local, canvas_size=canvas_size
                )
            except Exception as e:
                print(f"{h5ad_path.name}: failed cropping {row.name}: {e}", flush=True)
                continue

            field = safe_str(row.get("field", ""))
            patch_idx = safe_str(row.get("patch_idx", ""))
            cell_line = safe_str(row.get("Cell Line", ""))
            local_id = safe_str(row.get("local_id", row.name))

            fname = f"field_{field}_patch_{patch_idx}_cellline_{cell_line}_localid_{local_id}.png"

            imageio.imwrite(raw_dir / fname, raw_crop)
            imageio.imwrite(mask_dir / fname, mask_crop)

            if downsampled_size and downsampled_size > 0:
                resized_raw = cv2.resize(
                    raw_crop, (downsampled_size, downsampled_size), interpolation=cv2.INTER_LINEAR
                )
                resized_mask = cv2.resize(
                    mask_crop, (downsampled_size, downsampled_size), interpolation=cv2.INTER_LINEAR
                )
                imageio.imwrite(raw_dir.parent / f"raw_{downsampled_size}" / fname, resized_raw)
                imageio.imwrite(mask_dir.parent / f"mask_{downsampled_size}" / fname, resized_mask)
            processed += 1

    return processed, skipped_missing_image, h5ad_path.name


def main():
    args = parse_args()

    # Avoid thread oversubscription inside each worker (BLAS/numexpr/etc.)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_var, "1")

    h5ad_input = Path(args.h5ad_path)
    image_dir = Path(args.image_path)
    save_dir = Path(args.savepath)
    raw_dir = save_dir / "raw"
    mask_dir = save_dir / "mask"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    downsampled_size = args.downsampled_image_size
    if downsampled_size and downsampled_size > 0:
        (save_dir / f"raw_{downsampled_size}").mkdir(parents=True, exist_ok=True)
        (save_dir / f"mask_{downsampled_size}").mkdir(parents=True, exist_ok=True)

    if h5ad_input.is_dir():
        # Only process .h5ad files that include "nuclei" in the filename
        h5ad_files = sorted(
            f for f in h5ad_input.glob("*.h5ad") if "nuclei" in f.name
        )
    else:
        h5ad_files = [h5ad_input] if "nuclei" in h5ad_input.name else []

    if not h5ad_files:
        raise ValueError(f"No .h5ad files found at {h5ad_input}")

    total_processed = 0
    total_skipped = 0

    # Parallel across h5ad files when directory + n_workers>1, otherwise serial
    if len(h5ad_files) > 1 and args.n_workers and args.n_workers > 1:
        print(f"Parallel processing with {args.n_workers} workers… (files={len(h5ad_files)})", flush=True)
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = [
                ex.submit(
                    _process_h5ad_file,
                    h5ad_path,
                    image_dir,
                    raw_dir,
                    mask_dir,
                    args.canvas_size,
                    args.filter_edge_objects,
                    args.size_filter,
                    downsampled_size,
                )
                for h5ad_path in h5ad_files
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing H5AD Files"):
                try:
                    processed, skipped_missing_image, fname = fut.result()
                    total_processed += processed
                    total_skipped += skipped_missing_image
                    print(f"{fname}: saved {processed} crops. Skipped {skipped_missing_image} (missing images).", flush=True)
                except Exception as e:
                    print(f"Error in worker: {e}", flush=True)
    else:
        for h5ad_path in tqdm(h5ad_files, desc="Processing H5AD Files"):
            processed, skipped_missing_image, fname = _process_h5ad_file(
                h5ad_path,
                image_dir,
                raw_dir,
                mask_dir,
                args.canvas_size,
                args.filter_edge_objects,
                args.size_filter,
                downsampled_size,
            )
            total_processed += processed
            total_skipped += skipped_missing_image
            print(f"{fname}: saved {processed} crops. Skipped {skipped_missing_image} (missing images).", flush=True)

    print(f"All done. Saved {total_processed} crops to {save_dir} (raw/ and mask/).", flush=True)
    if total_skipped:
        print(f"Skipped {total_skipped} objects due to missing images across all files.", flush=True)
    print(
        "Filenames encode field/patch/cell line/local_id, enabling direct mapping back to obs rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()

