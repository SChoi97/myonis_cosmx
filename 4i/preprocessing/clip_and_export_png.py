#!/usr/bin/env python3
"""Stage 4: apply acquisition-specific display clipping and write uint8 PNGs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

from utils.clipping import clip_to_uint8, load_clip_limits
from utils.manifests import (
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    write_csv,
    write_json,
)
from utils.provenance import start_task, task_environment, verify_code_snapshot
from utils.registry import ACQUISITIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def png_name(tiff_name: str) -> str:
    return tiff_name[:-8] + ".png" if tiff_name.endswith(".ome.tif") else Path(tiff_name).with_suffix(".png").name


def main() -> int:
    args = build_parser().parse_args()
    if args.count_tasks:
        print(len(ACQUISITIONS))
        return 0
    if args.task_index is None or not 0 <= args.task_index < len(ACQUISITIONS):
        raise ValueError(f"--task-index must be in 0..{len(ACQUISITIONS) - 1}.")
    run_dir = args.run_dir.expanduser().resolve()
    verify_code_snapshot(run_dir)
    task_started, monotonic_start = start_task()
    acquisition = ACQUISITIONS[args.task_index]
    status_path = run_dir / "manifests" / "clipping" / f"{acquisition.acquisition_id}.json"
    run_manifest = read_json(run_dir / "run_manifest.json")
    workbook_path = Path(run_manifest["snapshots"]["clip_limits"])
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    limits_by_id = load_clip_limits(workbook_path)
    expected = {item.acquisition_id for item in ACQUISITIONS}
    if set(limits_by_id) != expected:
        raise ValueError(
            f"Clipping workbook must contain exactly 40 acquisitions; "
            f"missing={sorted(expected - set(limits_by_id))}, "
            f"unexpected={sorted(set(limits_by_id) - expected)}"
        )
    limits = limits_by_id[acquisition.acquisition_id]
    patch_rows = []
    patch_manifest_paths = sorted(
        (run_dir / "manifests" / "wells").glob("*/patches.csv")
    )
    for path in patch_manifest_paths:
        patch_rows.extend(
            row for row in read_csv(path) if row["acquisition_id"] == acquisition.acquisition_id
        )
    task_hash = sha256_json(
        {
            "stage": "clip_and_export_png",
            "acquisition_id": acquisition.acquisition_id,
            "clip_workbook_sha256": sha256_file(workbook_path),
            "patch_manifest_hashes": {
                str(path.relative_to(run_dir)): sha256_file(path)
                for path in patch_manifest_paths
            },
        }
    )
    output_manifest_path = (
        run_dir / "manifests" / "clipping" / f"{acquisition.acquisition_id}.csv"
    )
    if args.resume and status_path.is_file():
        previous = read_json(status_path)
        prior_rows = read_csv(output_manifest_path)
        if (
            previous.get("task_hash") == task_hash
            and previous.get("execution_mode") == "production"
            and previous.get("status") == "complete"
            and len(prior_rows) == int(previous.get("written_png_count", -1))
            and all(Path(row["png_path"]).is_file() for row in prior_rows)
        ):
            print(f"{acquisition.acquisition_id}: resumed verified PNG outputs")
            return 0
    replaced_outputs = []
    if not args.validate_only:
        png_root = (run_dir / "png_8bit").resolve()
        for prior in read_csv(output_manifest_path):
            prior_path = Path(prior.get("png_path", "")).resolve()
            if prior_path.is_file() and prior_path.is_relative_to(png_root):
                prior_path.unlink()
                replaced_outputs.append(
                    {
                        "stage": "clip_and_export_png",
                        "acquisition_id": acquisition.acquisition_id,
                        "path": str(prior_path),
                        "reason": "task_hash_changed_or_incomplete_previous_run",
                    }
                )
        write_csv(
            run_dir
            / "manifests"
            / "replaced_outputs"
            / f"{acquisition.acquisition_id}__png.csv",
            replaced_outputs,
        )
    output_rows = []
    if not args.validate_only:
        try:
            import tifffile
            from PIL import Image
        except ImportError as error:
            raise ImportError("tifffile and Pillow are required for PNG export.") from error
    for row in patch_rows:
        input_path = Path(row["quantitative_path"])
        output_path = (
            run_dir
            / "png_8bit"
            / acquisition.acquisition_id
            / png_name(input_path.name)
        )
        if not args.validate_only:
            image = np.asarray(tifffile.imread(input_path), dtype=np.float32)
            clipped = clip_to_uint8(
                image, limits["min_16bit"], limits["max_16bit"]
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.stem}.", suffix=".png", dir=output_path.parent
            )
            os.close(descriptor)
            try:
                Image.fromarray(clipped, mode="L").save(
                    temporary_name, format="PNG", optimize=True
                )
                os.replace(temporary_name, output_path)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        output_rows.append(
            {
                **row,
                "min_16bit": limits["min_16bit"],
                "max_16bit": limits["max_16bit"],
                "limits_source": limits["limits_source"],
                "custom_clipping": limits["custom_clipping"],
                "png_path": str(output_path),
            }
        )
    write_csv(output_manifest_path, output_rows)
    write_json(
        status_path,
        {
            "stage": "clip_and_export_png",
            "status": "validated" if args.validate_only else "complete",
            "acquisition": acquisition.as_row(),
            "clip_limits": limits,
            "input_patch_count": len(patch_rows),
            "written_png_count": 0 if args.validate_only else len(output_rows),
            "replaced_previous_output_count": len(replaced_outputs),
            "task_hash": task_hash,
            "execution_mode": "validation_only" if args.validate_only else "production",
            "environment": task_environment(task_started, monotonic_start),
        },
    )
    print(
        f"{acquisition.acquisition_id}: {len(output_rows)} PNGs at "
        f"[{limits['min_16bit']}, {limits['max_16bit']}] "
        f"({limits['limits_source']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
