#!/usr/bin/env python3
"""Repair sweep run folders accidentally created with carriage returns in paths."""

import argparse
import json
from pathlib import Path


CONFIG_NAME = "myonis_scvi_config.json"
SUMMARY_NAME = "sweep_run_summary.json"
KNOWN_ROOT_FILES = {
    "adata.h5ad",
    "model.pt",
    CONFIG_NAME,
    SUMMARY_NAME,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename sweep run directories/files whose names were corrupted by CRLF "
            "manifest rows, and strip carriage returns from JSON path strings."
        )
    )
    parser.add_argument("--sweep-dir", required=True, help="Sweep directory containing run folders.")
    parser.add_argument("--apply", action="store_true", help="Actually rename/update files. Default is dry-run.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def strip_cr(value):
    if isinstance(value, str):
        return value.replace("\r", "")
    if isinstance(value, dict):
        return {key: strip_cr(val) for key, val in value.items()}
    if isinstance(value, list):
        return [strip_cr(item) for item in value]
    return value


def write_json(path, data, apply):
    cleaned = strip_cr(data)
    changed = cleaned != data
    if changed and apply:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return changed


def find_latent_candidate(run_dir, run_id):
    expected = run_dir / f"{run_id}.h5ad"
    if expected.exists():
        return None, expected

    candidates = []
    for child in run_dir.iterdir():
        if not child.is_file():
            continue
        if child.name in KNOWN_ROOT_FILES:
            continue
        candidates.append(child)

    if len(candidates) == 1:
        return candidates[0], expected

    h5ad_candidates = [child for child in candidates if child.name.endswith(".h5ad")]
    if len(h5ad_candidates) == 1:
        return h5ad_candidates[0], expected

    return None, expected


def rename_path(src, dst, apply):
    if src == dst:
        return "unchanged"
    if dst.exists():
        return f"blocked_target_exists:{dst}"
    if apply:
        src.rename(dst)
    return f"rename:{src.name}->{dst.name}"


def repair_run_dir(run_dir, apply):
    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        return {"path": str(run_dir), "status": "skipped_missing_config"}

    config = load_json(config_path)
    run_id = config.get("run_id")
    if not run_id:
        return {"path": str(run_dir), "status": "skipped_missing_run_id"}

    summary_path = run_dir / SUMMARY_NAME
    summary = load_json(summary_path) if summary_path.exists() else None

    latent_src, latent_dst = find_latent_candidate(run_dir, run_id)
    latent_action = "none"
    if latent_src is not None:
        latent_action = rename_path(latent_src, latent_dst, apply)
        if latent_action.startswith("blocked"):
            return {
                "path": str(run_dir),
                "run_id": run_id,
                "status": "blocked",
                "reason": latent_action,
            }

    config_changed = write_json(config_path, config, apply)
    summary_changed = write_json(summary_path, summary, apply) if summary is not None else False

    target_dir = run_dir.parent / run_id
    dir_action = "none"
    final_dir = run_dir
    if run_dir.name != run_id:
        dir_action = rename_path(run_dir, target_dir, apply)
        if dir_action.startswith("blocked"):
            return {
                "path": str(run_dir),
                "run_id": run_id,
                "status": "blocked",
                "reason": dir_action,
            }
        final_dir = target_dir

    return {
        "path": str(run_dir),
        "final_path": str(final_dir),
        "run_id": run_id,
        "latent_action": latent_action,
        "config_changed": config_changed,
        "summary_changed": summary_changed,
        "dir_action": dir_action,
        "status": "repaired" if apply else "dry_run",
    }


def clean_manifest(sweep_dir, apply):
    # Sweep directories live at <scvi_dir>/results/sweeps/<sweep_name>.
    # The manifest lives at <scvi_dir>/sweeps/<sweep_name>_manifest.csv.
    scvi_dir = sweep_dir.parents[2]
    manifest = scvi_dir / "sweeps" / f"{sweep_dir.name}_manifest.csv"
    if not manifest.exists():
        return {"manifest": str(manifest), "exists": False, "changed": False}

    raw = manifest.read_bytes()
    cleaned = raw.replace(b"\r\n", b"\n").replace(b"\r", b"")
    changed = cleaned != raw
    if changed and apply:
        manifest.write_bytes(cleaned)
    return {"manifest": str(manifest), "exists": True, "changed": changed}


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise FileNotFoundError(sweep_dir)

    run_dirs = [path for path in sweep_dir.iterdir() if path.is_dir() and path.name != "evaluation"]
    results = [repair_run_dir(path, args.apply) for path in run_dirs]
    manifest_result = clean_manifest(sweep_dir, args.apply)

    status_counts = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    print("mode:", "APPLY" if args.apply else "DRY-RUN")
    print("sweep_dir:", sweep_dir)
    print("run_dirs:", len(run_dirs))
    print("status_counts:", status_counts)
    print("manifest:", manifest_result)
    print()
    for result in sorted(results, key=lambda row: row.get("run_id", row["path"])):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
