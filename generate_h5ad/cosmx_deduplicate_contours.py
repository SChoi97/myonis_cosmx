"""
Rule-based de-duplication of YOLOv8 segmentation contours.

Delegates the core logic to utils.cosmx_deduplication_utils.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local utils are importable when running as a script
_COSMX_DIR = Path(__file__).resolve().parent
if str(_COSMX_DIR) not in sys.path:
    sys.path.insert(0, str(_COSMX_DIR))

from utils.cosmx_deduplication_utils import run_deduplication


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based de-duplication of YOLOv8 segmentation contours."
    )
    parser.add_argument(
        "--contour_path",
        type=Path,
        required=True,
        help="Path to a directory of YOLOv8 seg txt files (or a single txt file).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.9,
        help="Containment threshold to consider a smaller object inside a larger one.",
    )
    parser.add_argument(
        "--object_threshold",
        type=int,
        default=3,
        help="From this number of enclosed objects onwards, remove the larger object.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1024,
        help="Patch size in pixels used to denormalize YOLO segmentation coords.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Optional directory containing source images for overlays.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".png",
        help="Preferred image extension when searching for overlays.",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=None,
        help="Optional output directory root. Will create 'contours' and 'visualisation' inside.",
    )
    args = parser.parse_args()
    run_deduplication(
        contour_path=args.contour_path,
        iou_threshold=args.iou_threshold,
        object_threshold=args.object_threshold,
        patch_size=args.patch_size,
        image_dir=args.image_dir,
        image_ext=args.image_ext,
        savepath=args.savepath,
    )


if __name__ == "__main__":
    main()
