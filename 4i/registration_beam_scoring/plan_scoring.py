#!/usr/bin/env python3
"""Choose a balanced random sample and assign wells to a bounded worker pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from progress import progress
from score_seams import HERE, atomic_csv, atomic_json, host_path, sha256, wells


def balanced_counts(capacities: dict[str, int], count: int, rng: random.Random) -> dict[str, int]:
    if count > sum(capacities.values()):
        raise ValueError(f"Requested {count}; only {sum(capacities.values())} are available")
    allocation = dict.fromkeys(capacities, 0)
    order = sorted(capacities)
    rng.shuffle(order)
    remaining = count
    while remaining:
        for key in order:
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                if not remaining:
                    break
    return allocation


def selection_hash(entry: dict, acquisition_ids: list[str]) -> str:
    value = {"patch_ids": entry["patch_ids"], "image_keys": entry["image_keys"],
             "acquisition_ids": acquisition_ids}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_plan(cache_path: Path, sample_count: int, sample_unit: str,
                scope: str, seed: int, max_jobs: int) -> dict:
    cache = json.loads(cache_path.read_text())
    run = host_path(cache["patch_run"])
    source_wells = wells(run)
    if len(source_wells) != int(cache["completed_wells"]):
        raise ValueError("Patch cache/well count mismatch")
    inventory = {}
    available_ids = set()
    for index, well in enumerate(progress(source_wells, desc="Indexing wells", unit="well")):
        with (well / "patches.csv").open(newline="", encoding="utf-8-sig") as stream:
            first_acquisition = None
            patch_ids = set()
            plate = None
            for row in csv.DictReader(stream):
                if first_acquisition is None:
                    first_acquisition, plate = row["acquisition_id"], row["plate_name"]
                if row["acquisition_id"] == first_acquisition:
                    patch_ids.add(row["patch_id"])
                elif index != 0:
                    break
                if index == 0:
                    available_ids.add(row["acquisition_id"])
        if not patch_ids:
            raise ValueError(f"No patches in {well}")
        inventory[well.name] = {"plate_name": plate, "patch_ids": sorted(patch_ids)}
    acquisition_ids = sorted(available_ids if scope == "all" else
                             {f"measurement_{value:02d}_dapi" for value in range(1, 13)})
    if set(acquisition_ids) - available_ids:
        raise ValueError("Requested acquisitions are absent from the cache")
    rng = random.Random(seed)
    capacities = {well: len(entry["patch_ids"]) * (len(acquisition_ids) if sample_unit == "image" else 1)
                  for well, entry in inventory.items()}
    requested = sample_count or sum(capacities.values())
    if requested > sum(capacities.values()):
        raise ValueError(f"Requested {requested} {sample_unit}s; only {sum(capacities.values())} available")
    by_plate = {}
    for well, entry in inventory.items():
        by_plate.setdefault(entry["plate_name"], []).append(well)
    if sample_count:
        plate_counts = balanced_counts({plate: sum(capacities[well] for well in members)
                                       for plate, members in by_plate.items()}, requested, rng)
        well_counts = {}
        for plate, members in sorted(by_plate.items()):
            well_counts.update(balanced_counts({well: capacities[well] for well in members}, plate_counts[plate], rng))
    else:
        well_counts = capacities
    selected = {}
    for well, entry in sorted(inventory.items()):
        count = well_counts[well]
        if not count:
            continue
        patches = entry["patch_ids"]
        image_keys = None
        if sample_count and sample_unit == "image":
            indices = rng.sample(range(len(patches) * len(acquisition_ids)), count)
            image_keys = sorted([[patches[i // len(acquisition_ids)], acquisition_ids[i % len(acquisition_ids)]]
                                 for i in indices])
            chosen_patches = sorted({key[0] for key in image_keys})
            image_count = len(image_keys)
        else:
            chosen_patches = sorted(rng.sample(patches, count)) if sample_count else patches
            image_count = len(chosen_patches) * len(acquisition_ids)
        record = {"well_slug": well, "plate_name": entry["plate_name"],
                  "patch_ids": chosen_patches, "image_keys": image_keys,
                  "patch_count": len(chosen_patches), "image_count": image_count,
                  "available_patch_count": len(patches)}
        record["selection_sha256"] = selection_hash(record, acquisition_ids)
        selected[well] = record
    worker_count = min(max_jobs - 1, len(selected))
    assignments = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for well in sorted(selected, key=lambda key: (-selected[key]["image_count"], key)):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(well)
        loads[worker] += selected[well]["image_count"]
    return {
        "schema_version": "scoring_plan_v1", "cache_manifest_sha256": sha256(cache_path),
        "patch_run": cache["patch_run"], "scope": scope, "acquisition_ids": acquisition_ids,
        "sampling": {"enabled": bool(sample_count), "unit": sample_unit,
                     "requested_count": sample_count, "seed": seed,
                     "balance": "equal across plates, then wells within each plate"},
        "max_child_jobs": max_jobs, "scoring_workers": worker_count, "gallery_jobs": 1,
        "worker_wells": assignments, "worker_image_counts": loads,
        "selected_patch_count": sum(entry["patch_count"] for entry in selected.values()),
        "selected_image_count": sum(entry["image_count"] for entry in selected.values()),
        "selected_well_count": len(selected), "wells": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, default=HERE / "results/cache_manifest.json")
    parser.add_argument("--sample-count", type=int, default=500, help="0 scores the full dataset")
    parser.add_argument("--sample-unit", choices=("patch", "image"), default="patch")
    parser.add_argument("--scope", choices=("all", "dapi"), default="dapi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-jobs", type=int, default=12, help="Total child jobs, including the gallery")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count < 0 or args.max_jobs < 2:
        parser.error("--sample-count must be nonnegative and --max-jobs must be at least 2")
    plan = create_plan(args.cache_manifest, args.sample_count, args.sample_unit, args.scope, args.seed, args.max_jobs)
    atomic_json(args.output, plan)
    distribution = [{key: entry[key] for key in ("well_slug", "plate_name", "available_patch_count", "patch_count", "image_count")}
                    for entry in plan["wells"].values()]
    atomic_csv(args.output.with_suffix(".distribution.csv"), distribution)
    print(f"Selected {plan['selected_patch_count']} spatial patches / {plan['selected_image_count']} TIFF images "
          f"from {plan['selected_well_count']} wells", flush=True)
    print(f"Child jobs: {plan['scoring_workers']} scoring workers + 1 gallery (cap {args.max_jobs})", flush=True)
    print(f"Scoring plan: {args.output}", flush=True)


if __name__ == "__main__":
    main()
