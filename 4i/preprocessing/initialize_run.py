#!/usr/bin/env python3
"""Create an immutable timestamped run directory and snapshot all inputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from utils.config import config_as_dict, load_config, with_execution_mode
from utils.clipping import load_clip_limits
from utils.correction import load_model, model_load_mode
from utils.manifests import approved_selections
from utils.provenance import initialize_run
from utils.registry import ACQUISITIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-workbook", type=Path, required=True)
    parser.add_argument("--clip-limits", type=Path, required=True)
    parser.add_argument(
        "--patch-size",
        type=int,
        help="Square output patch size in pixels; overrides the config for this run.",
    )
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument(
        "--registration-level",
        choices=("field_level", "patch_level"),
        default="field_level",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.patch_size is not None:
        config = replace(config, patch_size=args.patch_size)
    config = with_execution_mode(
        config,
        test_mode=args.test_mode,
        registration_level=args.registration_level,
    )
    selections = approved_selections(args.selection)
    expected = {item.acquisition_id for item in ACQUISITIONS}
    if set(selections) != expected:
        raise ValueError(
            f"Selection must contain exactly 40 acquisitions; "
            f"missing={sorted(expected - set(selections))}, "
            f"unexpected={sorted(set(selections) - expected)}"
        )
    if not args.clip_limits.is_file():
        raise FileNotFoundError(args.clip_limits)
    if not args.selection_workbook.is_file():
        raise FileNotFoundError(args.selection_workbook)
    clip_limits = load_clip_limits(args.clip_limits)
    if set(clip_limits) != expected:
        raise ValueError(
            f"Clipping workbook must contain valid limits for exactly 40 acquisitions; "
            f"missing={sorted(expected - set(clip_limits))}, "
            f"unexpected={sorted(set(clip_limits) - expected)}"
        )
    model_validation = {}
    for acquisition in ACQUISITIONS:
        method, model = load_model(selections[acquisition.acquisition_id])
        model_validation[acquisition.acquisition_id] = {
            "method": method,
            "load_mode": model_load_mode(model),
        }
        del model
    run_dir = initialize_run(
        args.runs_root,
        args.config.resolve(),
        args.selection.resolve(),
        args.selection_workbook.resolve(),
        args.clip_limits.resolve(),
        Path(__file__).resolve().parents[2],
        config_snapshot=config_as_dict(config),
        model_load_validation=model_validation,
    )
    # This command's stdout is consumed by the submission shell script.
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
