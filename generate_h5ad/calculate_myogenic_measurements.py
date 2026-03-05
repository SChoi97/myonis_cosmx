import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Dict, Optional

import pandas as pd

# Ensure project root is importable for utils
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.myogenic_visualisation_utils import numericalSort
from utils.myogenic_calculation_utils import process_anchor_path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute myogenic measurements and visualisations from YOLO contours (no patching)."
    )
    p.add_argument("--dapi-dir", default=None, type=Path, help="Directory with DAPI images (optional)")
    p.add_argument("--myhc-dir", default=None, type=Path, help="Directory with MyHC images (optional)")
    p.add_argument(
        "--dapi-contour-dir",
        default=None,
        type=Path,
        help="Directory with YOLO txt contours for nuclei (DAPI). Optional; defaults to --dapi-dir when omitted.",
    )
    p.add_argument(
        "--myhc-contour-dir",
        default=None,
        type=Path,
        help="Directory with YOLO txt contours for myotubes (MyHC). Optional; defaults to --myhc-dir when omitted.",
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory base")
    p.add_argument("--image-pattern", default="*.png", help="Glob pattern for images in provided image directories (default: *.png)")
    p.add_argument(
        "--name-match-threshold",
        default=0.9,
        type=float,
        help="Minimum SequenceMatcher ratio for fuzzy image/txt name matching when no exact stem is found; below this the field is skipped (default: 0.9)",
    )
    p.add_argument(
        "--channel_index_matching",
        action="store_true",
        help="Use strict channel-index pairing (chN<->chM) with exact non-channel stem matching; overrides fuzzy name matching when set.",
    )
    p.add_argument("--dapi-clip", default=(20, 220), nargs=2, type=int, metavar=("LOW", "HIGH"))
    p.add_argument("--myhc-clip", default=(1000, 15000), nargs=2, type=int, metavar=("LOW", "HIGH"))
    p.add_argument("--dpi", default=300, type=int, help="DPI for saved figures")
    p.add_argument("--dapi-linewidth", default=5, type=int, help="Line width for nuclei outlines")
    p.add_argument("--myhc-linewidth", default=7, type=int, help="Line width for myotube outlines")
    p.add_argument("--save-pngs-individually", action="store_true", help="If set, save each panel as individual PNGs")
    p.add_argument(
        "--n-workers",
        "--n_workers",
        dest="n_workers",
        default=1,
        type=int,
        help="Number of parallel workers across images (default: 1)",
    )
    p.add_argument(
        "--nuclei-overlap-threshold",
        default=0.8,
        type=float,
        help="Fractional overlap required to assign a nucleus to a myotube (default: 0.8)",
    )
    p.add_argument(
        "--edge-threshold",
        "--edge_threshold",
        dest="edge_threshold",
        default=0,
        type=int,
        help="Pixels from image edge to mark a nucleus as edge-touching (default: 0)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.dapi_dir is None and args.myhc_dir is None:
        print("[ERROR] At least one image directory must be provided: --dapi-dir and/or --myhc-dir")
        return

    if args.dapi_dir is not None and args.dapi_contour_dir is None:
        print("[WARN] --dapi-contour-dir not provided; using --dapi-dir for nuclei contour lookup")
        args.dapi_contour_dir = args.dapi_dir
    if args.myhc_dir is not None and args.myhc_contour_dir is None:
        print("[WARN] --myhc-contour-dir not provided; using --myhc-dir for myotube contour lookup")
        args.myhc_contour_dir = args.myhc_dir

    for arg_name, path in [
        ("dapi-dir", args.dapi_dir),
        ("myhc-dir", args.myhc_dir),
        ("dapi-contour-dir", args.dapi_contour_dir),
        ("myhc-contour-dir", args.myhc_contour_dir),
    ]:
        if path is not None and not path.exists():
            print(f"[ERROR] --{arg_name} does not exist: {path}")
            return

    out_dir = args.out_dir
    plots_dir = out_dir / "plots"
    images_dir = out_dir / "images" if args.save_pngs_individually else plots_dir
    measurements_dir = out_dir / "measurements"
    field_measurements_dir = measurements_dir / "field_measurements"
    combined_measurements_dir = measurements_dir / "combined_measurements"
    plots_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    field_measurements_dir.mkdir(parents=True, exist_ok=True)
    combined_measurements_dir.mkdir(parents=True, exist_ok=True)

    dapi_paths = sorted(args.dapi_dir.glob(args.image_pattern), key=numericalSort) if args.dapi_dir is not None else []
    myhc_paths = sorted(args.myhc_dir.glob(args.image_pattern), key=numericalSort) if args.myhc_dir is not None else []

    if dapi_paths:
        anchor_paths = dapi_paths
        anchor_modality = "dapi"
    elif myhc_paths:
        anchor_paths = myhc_paths
        anchor_modality = "myhc"
    else:
        print(f"[WARN] No images found with pattern {args.image_pattern}")
        return

    if args.dapi_dir is not None and not dapi_paths:
        print(f"[WARN] No DAPI images found in {args.dapi_dir} with pattern {args.image_pattern}")
    if args.myhc_dir is not None and not myhc_paths:
        print(f"[WARN] No MyHC images found in {args.myhc_dir} with pattern {args.image_pattern}")

    per_nucleus_columns = [
        "image_name",
        "nucleus_id",
        "assignment_idx",
        "area",
        "perimeter",
        "circularity",
        "major_axis_length",
        "average_width",
        "assigned_myotube_id",
        "is_edge",
    ]
    per_myotube_columns = [
        "image_name",
        "myotube_id",
        "assignment_idx",
        "area",
        "perimeter",
        "circularity",
        "major_axis_length",
        "average_width",
        "average_intensity",
        "median_intensity",
        "stdev_intensity",
        "n_nuclei",
    ]

    all_per_nucleus: List[dict] = []
    all_per_myotube: List[dict] = []
    all_per_field: List[dict] = []

    if args.n_workers < 1:
        print(f"[WARN] --n-workers must be >= 1; got {args.n_workers}. Using 1.")
        args.n_workers = 1

    if args.n_workers > 1 and len(anchor_paths) > 1:
        ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(anchor_paths)
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            future_to_idx = {
                ex.submit(
                    process_anchor_path,
                    anchor_path,
                    anchor_modality,
                    args,
                    field_measurements_dir,
                    plots_dir,
                    images_dir,
                    per_nucleus_columns,
                    per_myotube_columns,
                ): idx
                for idx, anchor_path in enumerate(anchor_paths)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Fields"):
                idx = future_to_idx[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    print(f"[ERROR] {anchor_paths[idx].stem}: {e}")

        for result in ordered_results:
            if result is None:
                continue
            all_per_nucleus.extend(result["per_nucleus_rows"])
            all_per_myotube.extend(result["per_myotube_rows"])
            if result["per_field_row"] is not None:
                all_per_field.append(result["per_field_row"])
    else:
        for anchor_path in tqdm(anchor_paths, desc="Fields"):
            try:
                result = process_anchor_path(
                    anchor_path,
                    anchor_modality,
                    args,
                    field_measurements_dir,
                    plots_dir,
                    images_dir,
                    per_nucleus_columns,
                    per_myotube_columns,
                )
            except Exception as e:
                print(f"[ERROR] {anchor_path.stem}: {e}")
                continue

            all_per_nucleus.extend(result["per_nucleus_rows"])
            all_per_myotube.extend(result["per_myotube_rows"])
            if result["per_field_row"] is not None:
                all_per_field.append(result["per_field_row"])

    # Write aggregated measurement tables
    if args.dapi_dir is not None and args.dapi_contour_dir is not None:
        pd.DataFrame(all_per_nucleus, columns=per_nucleus_columns).to_csv(
            combined_measurements_dir / "all_per_nucleus.csv", index=False
        )
    if args.myhc_dir is not None and args.myhc_contour_dir is not None:
        pd.DataFrame(all_per_myotube, columns=per_myotube_columns).to_csv(
            combined_measurements_dir / "all_per_myotube.csv", index=False
        )
    if args.dapi_dir is not None and args.myhc_dir is not None:
        if all_per_field:
            pd.DataFrame(all_per_field).to_csv(combined_measurements_dir / "all_per_field.csv", index=False)
        else:
            pd.DataFrame(
                columns=[
                    "image_name",
                    "total_myotubes",
                    "total_nuclei",
                    "nuclei_in_myotubes",
                    "differentiation_index",
                    "fusion_index_nuclei",
                    "fusion_index_assigned_nuclei",
                    "fused_myotube_fraction",
                    "mean_nuclei_per_myotube",
                ]
            ).to_csv(combined_measurements_dir / "all_per_field.csv", index=False)


if __name__ == "__main__":
    main()
