#!/usr/bin/env python3
"""Attach submitted Slurm job IDs to the immutable run manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.manifests import read_json, write_json
from utils.provenance import timestamp_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-job", required=True)
    parser.add_argument("--illumination-job", required=True)
    parser.add_argument("--registration-job", required=True)
    parser.add_argument("--corrected-cleanup-job", required=True)
    parser.add_argument("--patch-job", required=True)
    parser.add_argument("--registered-cleanup-job", required=True)
    parser.add_argument("--clipping-job", required=True)
    parser.add_argument("--report-job", required=True)
    args = parser.parse_args()
    path = args.run_dir / "run_manifest.json"
    manifest = read_json(path)
    manifest["submitted"] = timestamp_values()
    manifest["slurm_job_graph"] = {
        "real_data_smoke_test": args.smoke_job,
        "illumination": args.illumination_job,
        "registration": args.registration_job,
        "cleanup_corrected": args.corrected_cleanup_job,
        "patch_generation": args.patch_job,
        "cleanup_registered_crops": args.registered_cleanup_job,
        "clipping": args.clipping_job,
        "report": args.report_job,
    }
    write_json(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
