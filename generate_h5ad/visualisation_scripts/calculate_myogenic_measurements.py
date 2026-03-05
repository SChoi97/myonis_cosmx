import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from shapely.geometry import Polygon, LineString
except ImportError:  # pragma: no cover - runtime guard
    print("shapely is required. Install via `pip install shapely`.", file=sys.stderr)
    sys.exit(1)

# Ensure project root is importable for utils
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cosmx.utils.cosmx_visualisation_utils import (
    numericalSort,
    create_fiji_lut,
    visualise_labels,
)
from scripts.cosmx.utils.cosmx_utils import compute_bbox, assign_nuclei_to_myotubes
from tqdm import tqdm


def load_yolo_segmentation_polygons(txt_path: Path, width: int, height: int) -> List[np.ndarray]:
    """
    Load YOLOv8 segmentation polygons (normalized coords) and convert to pixel space.
    """
    polygons: list[np.ndarray] = []
    if not txt_path.exists():
        return polygons
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # cls + at least 3 points
                continue
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            arr = np.array(coords, dtype=np.float32).reshape(-1, 2)
            arr[:, 0] *= float(width)
            arr[:, 1] *= float(height)
            polygons.append(arr)
    return polygons


def _compute_width_samples(poly_points: np.ndarray, min_samples: int = 20) -> List[float]:
    """
    Estimate width by intersecting the polygon with lines perpendicular
    to the major axis sampled along its length.
    """
    if poly_points is None or len(poly_points) < 3:
        return [0.0]

    poly = Polygon(poly_points)
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return [0.0]
    if poly.is_empty:
        return [0.0]
    if poly.geom_type == "MultiPolygon":
        # Pick the largest polygon component
        parts = list(poly.geoms)
        if not parts:
            return [0.0]
        poly = max(parts, key=lambda g: g.area)
        if poly.is_empty:
            return [0.0]

    pts = np.asarray(poly.exterior.coords, dtype=np.float64)
    centroid = np.array(poly.centroid.coords[0], dtype=np.float64)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_vec = eigvecs[:, np.argmax(eigvals)]
    axis_norm = np.linalg.norm(axis_vec) + 1e-12
    axis_vec = axis_vec / axis_norm
    perp_vec = np.array([-axis_vec[1], axis_vec[0]])

    projections = pts @ axis_vec
    min_p, max_p = projections.min(), projections.max()
    span = max_p - min_p
    if span <= 1e-6:
        return [0.0]

    num_samples = max(min_samples, 20)
    samples = np.linspace(min_p, max_p, num_samples)
    centroid_proj = centroid @ axis_vec
    line_half_len = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2.0 + 10.0

    widths: List[float] = []
    for s in samples:
        point_on_axis = centroid + axis_vec * (s - centroid_proj)
        line = LineString(
            [
                point_on_axis - perp_vec * line_half_len,
                point_on_axis + perp_vec * line_half_len,
            ]
        )
        inter = poly.intersection(line)
        if inter.is_empty:
            continue

        coords: List[tuple[float, float]] = []
        if inter.geom_type == "LineString":
            coords.extend(inter.coords)
        elif inter.geom_type == "MultiLineString":
            for seg in inter.geoms:
                coords.extend(seg.coords)
        elif inter.geom_type == "Point":
            coords.append(inter.coords[0])
        elif inter.geom_type == "GeometryCollection":
            for geom in inter.geoms:
                if geom.geom_type == "LineString":
                    coords.extend(geom.coords)
                elif geom.geom_type == "Point":
                    coords.append(geom.coords[0])

        if len(coords) < 2:
            continue

        proj = np.dot(np.array(coords, dtype=np.float64), perp_vec)
        width_val = float(proj.max() - proj.min())
        if width_val >= 0:
            widths.append(width_val)

    return widths if widths else [0.0]


def _compute_width_segments_for_overlay(poly_points: np.ndarray, min_samples: int = 20) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Similar to _compute_width_samples, but returns line segments (start, end) for overlay.
    Each segment is a tuple of two (x, y) integer points.
    """
    if poly_points is None or len(poly_points) < 3:
        return []

    poly = Polygon(poly_points)
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return []
    if poly.is_empty:
        return []
    if poly.geom_type == "MultiPolygon":
        parts = list(poly.geoms)
        if not parts:
            return []
        poly = max(parts, key=lambda g: g.area)
        if poly.is_empty:
            return []

    pts = np.asarray(poly.exterior.coords, dtype=np.float64)
    centroid = np.array(poly.centroid.coords[0], dtype=np.float64)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_vec = eigvecs[:, np.argmax(eigvals)]
    axis_norm = np.linalg.norm(axis_vec) + 1e-12
    axis_vec = axis_vec / axis_norm
    perp_vec = np.array([-axis_vec[1], axis_vec[0]])

    projections = pts @ axis_vec
    min_p, max_p = projections.min(), projections.max()
    span = max_p - min_p
    if span <= 1e-6:
        return []

    num_samples = max(min_samples, 20)
    samples = np.linspace(min_p, max_p, num_samples)
    centroid_proj = centroid @ axis_vec
    line_half_len = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2.0 + 10.0

    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for s in samples:
        point_on_axis = centroid + axis_vec * (s - centroid_proj)
        line = LineString(
            [
                point_on_axis - perp_vec * line_half_len,
                point_on_axis + perp_vec * line_half_len,
            ]
        )
        inter = poly.intersection(line)
        if inter.is_empty:
            continue

        geoms = []
        if inter.geom_type == "LineString":
            geoms = [inter]
        elif inter.geom_type == "MultiLineString":
            geoms = list(inter.geoms)
        elif inter.geom_type == "GeometryCollection":
            geoms = [g for g in inter.geoms if g.geom_type == "LineString"]
        else:
            continue

        for g in geoms:
            coords = list(g.coords)
            if len(coords) < 2:
                continue
            start = tuple(map(int, map(round, coords[0])))
            end = tuple(map(int, map(round, coords[-1])))
            segments.append((start, end))

    return segments


def draw_width_sampling_overlay(base_image: np.ndarray, myotube_contours: List[np.ndarray], color=(255, 0, 0), thickness: int = 1, min_samples: int = 20) -> np.ndarray:
    """
    Draw width sampling line segments on top of the provided overlay image.
    """
    overlay = base_image.copy()
    for poly in myotube_contours:
        segments = _compute_width_segments_for_overlay(poly, min_samples=min_samples)
        for (x1, y1), (x2, y2) in segments:
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
    return overlay


def find_matching_image(base: str, directory: Path, threshold: float = 0.9) -> Tuple[Optional[Path], float, str]:
    """
    Find a file whose stem matches `base`.
    Strategy:
      1) exact stem match (base.*)
      2) otherwise pick the file with the highest SequenceMatcher ratio above `threshold`
    Returns: (Path or None, score, mode ['exact'|'fuzzy'])
    """
    candidates = sorted(directory.glob(f"{base}.*"))
    for cand in candidates:
        if cand.is_file():
            return cand, 1.0, "exact"

    best_path: Optional[Path] = None
    best_score: float = threshold
    best_score_any: float = 0.0
    for cand in sorted(directory.iterdir()):
        if not cand.is_file():
            continue
        score = SequenceMatcher(None, cand.stem, base).ratio()
        if score > best_score_any:
            best_score_any = score
        if score > best_score:
            best_score = score
            best_path = cand

    mode = "fuzzy" if best_path is not None else "none"
    return best_path, best_score if best_path is not None else best_score_any, mode


def find_matching_txt(stem: str, directory: Path, threshold: float = 0.9) -> Tuple[Optional[Path], float, str]:
    """
    Find a .txt file whose stem matches `stem`, with the same exact/fuzzy strategy.
    """
    exact = directory / f"{stem}.txt"
    if exact.exists():
        return exact, 1.0, "exact"

    best_path: Optional[Path] = None
    best_score: float = threshold
    best_score_any: float = 0.0
    for cand in sorted(directory.glob("*.txt")):
        score = SequenceMatcher(None, cand.stem, stem).ratio()
        if score > best_score_any:
            best_score_any = score
        if score > best_score:
            best_score = score
            best_path = cand

    mode = "fuzzy" if best_path is not None else "none"
    return best_path, best_score if best_path is not None else best_score_any, mode


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute myogenic measurements and visualisations from YOLO contours (no patching)."
    )
    p.add_argument("--dapi-dir", required=True, type=Path, help="Directory with DAPI images")
    p.add_argument("--myhc-dir", required=True, type=Path, help="Directory with MyHC images")
    p.add_argument("--dapi-contour-dir", required=True, type=Path, help="Directory with YOLO txt contours for nuclei (DAPI)")
    p.add_argument("--myhc-contour-dir", required=True, type=Path, help="Directory with YOLO txt contours for myotubes (MyHC)")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory base")
    p.add_argument("--image-pattern", default="*.png", help="Glob pattern for images in dapi-dir (default: *.png)")
    p.add_argument(
        "--name-match-threshold",
        default=0.9,
        type=float,
        help="Minimum SequenceMatcher ratio for fuzzy image/txt name matching when no exact stem is found; below this the field is skipped (default: 0.9)",
    )
    p.add_argument("--dapi-clip", default=(20, 220), nargs=2, type=int, metavar=("LOW", "HIGH"))
    p.add_argument("--myhc-clip", default=(1000, 15000), nargs=2, type=int, metavar=("LOW", "HIGH"))
    p.add_argument("--dpi", default=300, type=int, help="DPI for saved figures")
    p.add_argument("--dapi-linewidth", default=5, type=int, help="Line width for nuclei outlines")
    p.add_argument("--myhc-linewidth", default=7, type=int, help="Line width for myotube outlines")
    p.add_argument("--save-pngs-individually", action="store_true", help="If set, save each panel as individual PNGs")
    p.add_argument(
        "--nuclei-overlap-threshold",
        default=0.8,
        type=float,
        help="Fractional overlap required to assign a nucleus to a myotube (default: 0.8)",
    )
    return p.parse_args()


def main():
    args = parse_args()

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

    dapi_paths = sorted(args.dapi_dir.glob(args.image_pattern), key=numericalSort)
    if not dapi_paths:
        print(f"[WARN] No DAPI images found in {args.dapi_dir} with pattern {args.image_pattern}")
        return

    all_per_object: List[dict] = []
    all_per_field: List[dict] = []

    for dapi_path in tqdm(dapi_paths, desc="Fields"):
        base = dapi_path.stem
        myhc_path, myhc_score, myhc_mode = find_matching_image(base, args.myhc_dir, threshold=args.name_match_threshold)
        if myhc_path is None:
            print(f"[WARN] Missing MyHC image for {base} (no exact/fuzzy match) -> skipping")
            continue
        if myhc_mode == "fuzzy":
            print(f"[WARN] Using fuzzy MyHC match for {base}: {myhc_path.name} (score={myhc_score:.3f})")

        myhc_stem = myhc_path.stem

        dapi_contour_path, dapi_contour_score, dapi_contour_mode = find_matching_txt(
            base, args.dapi_contour_dir, threshold=args.name_match_threshold
        )
        myhc_contour_path, myhc_contour_score, myhc_contour_mode = find_matching_txt(
            myhc_stem, args.myhc_contour_dir, threshold=args.name_match_threshold
        )

        if dapi_contour_path is None or myhc_contour_path is None:
            print(
                f"[WARN] Missing contour txt for {base}: nuclei={'found' if dapi_contour_path else 'missing'} "
                f"myotube={'found' if myhc_contour_path else 'missing'} -> skipping"
            )
            continue
        if dapi_contour_mode == "fuzzy":
            print(f"[WARN] Using fuzzy nuclei contour match for {base}: {dapi_contour_path.name} (score={dapi_contour_score:.3f})")
        if myhc_contour_mode == "fuzzy":
            print(f"[WARN] Using fuzzy myotube contour match for {myhc_stem}: {myhc_contour_path.name} (score={myhc_contour_score:.3f})")

        dapi_img = np.array(imageio.imread(dapi_path))
        myhc_img = np.array(imageio.imread(myhc_path))
        if dapi_img.shape[:2] != myhc_img.shape[:2]:
            print(f"[WARN] Shape mismatch for {base} (DAPI {dapi_img.shape}, MyHC {myhc_img.shape}) -> skipping")
            continue

        height, width = dapi_img.shape[0], dapi_img.shape[1]
        nuclei_polys = load_yolo_segmentation_polygons(dapi_contour_path, width, height)
        myotube_polys = load_yolo_segmentation_polygons(myhc_contour_path, width, height)

        nuclei_objects = []
        for poly in nuclei_polys:
            if len(poly) < 3:
                continue
            nuclei_objects.append({"polygon": poly, "bbox": compute_bbox(poly)})

        myotube_objects = []
        for poly in myotube_polys:
            if len(poly) < 3:
                continue
            myotube_objects.append({"polygon": poly, "bbox": compute_bbox(poly)})

        assignment = assign_nuclei_to_myotubes(
            nuclei_objects, myotube_objects, overlap_threshold=args.nuclei_overlap_threshold, verbose=False
        )

        myotube_n_nuclei = np.zeros(len(myotube_objects), dtype=int)
        for n_idx, m_idx in assignment.items():
            if 0 <= m_idx < len(myotube_n_nuclei):
                myotube_n_nuclei[m_idx] += 1

        # Measurements per myotube
        per_object_rows = []
        for i, poly in enumerate(myotube_polys):
            area = Polygon(poly).area if len(poly) >= 3 else 0.0
            widths = _compute_width_samples(poly)
            width_mean = float(np.mean(widths)) if widths else 0.0
            per_object_rows.append(
                {
                    "image_name": base,
                    "object_id": i,
                    "width": width_mean,
                    "area": area,
                    "n_nuclei": int(myotube_n_nuclei[i]) if i < len(myotube_n_nuclei) else 0,
                }
            )
        per_object_df = pd.DataFrame(per_object_rows)
        per_object_path = field_measurements_dir / f"{base}_per_object.csv"
        per_object_df.to_csv(per_object_path, index=False)
        all_per_object.extend(per_object_rows)

        # Per-field measurements
        total_nuclei = len(nuclei_objects)
        nuclei_in_myotubes = len(assignment)
        diff_index = float(nuclei_in_myotubes) / float(total_nuclei) if total_nuclei > 0 else 0.0
        counts_counter = Counter(myotube_n_nuclei.tolist()) if len(myotube_n_nuclei) > 0 else Counter()
        max_nuclei_count = int(myotube_n_nuclei.max()) if len(myotube_n_nuclei) > 0 else 0

        per_field_row = {
            "image_name": base,
            "total_myotubes": len(myotube_objects),
            "total_nuclei": total_nuclei,
            "nuclei_in_myotubes": nuclei_in_myotubes,
            "differentiation_index": diff_index,
        }
        for nuclei_count in range(max_nuclei_count + 1):
            per_field_row[f"n_nuclei_{nuclei_count}"] = counts_counter.get(nuclei_count, 0)

        per_field_df = pd.DataFrame([per_field_row])
        per_field_path = field_measurements_dir / f"{base}_per_field.csv"
        per_field_df.to_csv(per_field_path, index=False)
        all_per_field.append(per_field_row)

        # Visual outputs (matches structure of previous script)
        dapi_clip_low, dapi_clip_high = args.dapi_clip
        myhc_clip_low, myhc_clip_high = args.myhc_clip
        dapi_mod = np.clip(dapi_img, dapi_clip_low, dapi_clip_high)
        myhc_mod = np.clip(myhc_img, myhc_clip_low, myhc_clip_high)

        dapi_den = max(1e-9, float(dapi_mod.max() - dapi_mod.min()))
        myhc_den = max(1e-9, float(myhc_mod.max() - myhc_mod.min()))
        dapi_norm = (dapi_mod - dapi_mod.min()) / dapi_den
        myhc_norm = (myhc_mod - myhc_mod.min()) / myhc_den

        merged_rgb = np.zeros((dapi_norm.shape[0], dapi_norm.shape[1], 3), dtype=np.float32)
        merged_rgb[:, :, 0] = myhc_norm
        merged_rgb[:, :, 1] = np.clip(dapi_norm + myhc_norm, 0, 1)
        merged_rgb[:, :, 2] = np.clip(dapi_norm + myhc_norm, 0, 1)

        fig_img, ax_img = plt.subplots(1, 3, figsize=(18, 6))
        ax_img[0].imshow(dapi_mod, cmap=create_fiji_lut("cyan"))
        ax_img[0].set_axis_off()
        ax_img[1].imshow(myhc_mod, cmap=create_fiji_lut("gray"))
        ax_img[1].set_axis_off()
        ax_img[2].imshow(merged_rgb)
        ax_img[2].set_axis_off()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{base}_images.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_img)

        if args.save_pngs_individually:
            merged_rgb_u8 = (np.clip(merged_rgb, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(images_dir / f"{base}_dapi_fluo.png", dapi_mod.astype(np.uint16))
            imageio.imwrite(images_dir / f"{base}_myhc_fluo.png", myhc_mod.astype(np.uint16))
            imageio.imwrite(images_dir / f"{base}_merged_fluo.png", merged_rgb_u8)

        # Overlays
        myhc_for_overlay = (myhc_mod / (myhc_clip_high or 1)).clip(0, 1) * 255.0
        myhc_for_overlay = myhc_for_overlay.astype(np.uint8)
        assignment_labels = [-1 for _ in nuclei_polys]
        for n_idx, m_idx in assignment.items():
            if 0 <= n_idx < len(assignment_labels):
                assignment_labels[n_idx] = m_idx

        nuclei_overlay = visualise_labels(
            image=dapi_mod,
            contours=nuclei_polys,
            alpha=1.0,
            linewidth=args.dapi_linewidth,
        )
        myotube_overlay = visualise_labels(
            image=myhc_for_overlay,
            contours=myotube_polys,
            alpha=1.0,
            linewidth=args.myhc_linewidth,
        )
        combined_overlay = visualise_labels(
            image=myhc_for_overlay,
            contours=nuclei_polys,
            alpha=2,
            linewidth=args.dapi_linewidth,
            myotube_contours=myotube_polys,
            myotube_linewidth=args.myhc_linewidth,
            myotube_alpha=0.5,
            myotube_line_color=(211, 211, 211),
            myotube_fill_color=(200, 200, 200),
            myotube_color="#6E6E6E",
            nuclei_color="#00A0F7",
        )
        assignment_overlay = visualise_labels(
            image=myhc_for_overlay,
            contours=nuclei_polys,
            alpha=2,
            linewidth=args.dapi_linewidth,
            myotube_contours=myotube_polys,
            myotube_linewidth=args.myhc_linewidth,
            myotube_alpha=0.3,
            myotube_line_color=(120, 120, 120),
            myotube_fill_color=(200, 200, 200),
            assignment_labels=assignment_labels,
        )

        fig_overlay, ax_overlay = plt.subplots(1, 4, figsize=(28, 7))
        plt.subplots_adjust(wspace=0.02, hspace=0)
        overlays = [nuclei_overlay, myotube_overlay, combined_overlay, assignment_overlay]
        for a, img in zip(ax_overlay, overlays):
            a.imshow(img)
            a.axis("off")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{base}_annotations.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_overlay)

        if args.save_pngs_individually:
            imageio.imwrite(images_dir / f"{base}_nuclei_overlay.png", nuclei_overlay)
            imageio.imwrite(images_dir / f"{base}_myotube_overlay.png", myotube_overlay)
            imageio.imwrite(images_dir / f"{base}_combined_overlay.png", combined_overlay)
            imageio.imwrite(images_dir / f"{base}_assignment_overlay.png", assignment_overlay)

        # Width sampling visualization (saved to images_dir only)
        width_overlay = draw_width_sampling_overlay(
            base_image=combined_overlay,
            myotube_contours=myotube_polys,
            color=(255, 0, 0),
            thickness=1,
            min_samples=20,
        )
        imageio.imwrite(images_dir / f"{base}_width_sampling.png", width_overlay)

    # Write aggregated measurement tables
    if all_per_object:
        pd.DataFrame(all_per_object).to_csv(combined_measurements_dir / "all_per_object.csv", index=False)
    if all_per_field:
        pd.DataFrame(all_per_field).to_csv(combined_measurements_dir / "all_per_field.csv", index=False)


if __name__ == "__main__":
    main()

