"""Portable reference-object QC exchange format; standard-library only.

Image name + original YOLO line + file checksum identify an object without
depending on the Nemo/Mac mount, quantification dataset ID or segmentation ID.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

IDENTITY = ["reference_image_name", "reference_source_line_number", "reference_id",
            "reference_contour_sha256", "reference_contour_path", "reference_round", "patch_key"]
FLAGS = ["is_edge", "is_persistent", "qc_pass", "qc_status", "qc_exclusion_reason",
         "qc_filter_mode", "qc_score_metric", "qc_score_aggregation", "qc_threshold", "qc_threshold_rule",
         "qc_score", "qc_mean_change_score", "qc_max_change_score", "qc_valid_area_fraction",
         "qc_n_comparison_rounds", "qc_n_rounds_passing", "qc_required_other_rounds"]
PATCH_COLUMNS = ["patch_key", "reference_image_name", "reference_contour_sha256", "reference_contour_path",
                 "reference_round", "reference_count", "height", "width", "qc_patch_status", "detail"]
ROUND_COLUMNS = IDENTITY + ["round", "is_reference", "is_filter_round", "score", "passes_round", "matched_contour_index", "is_edge_round"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(image_path, contour_path):
    raw = Path(contour_path).read_bytes()
    # Both readers retain every nonblank row, in order. Actual file lines can
    # differ from one-based reference IDs when the input contains blank lines.
    lines = [i for i, line in enumerate(raw.decode("utf-8").splitlines(), 1) if line.strip()]
    return dict(reference_image_name=Path(image_path).stem, reference_contour_path=str(contour_path),
                reference_contour_sha256=hashlib.sha256(raw).hexdigest(), source_line_numbers=lines)


def _at(record, name, i):
    return record.get(name, [None]*(i+1))[i]


def object_rows(record, source):
    """Yield every original reference object, including edge and failed objects."""
    n, ref = int(record["reference_count"]), int(record["reference_round"])
    if len(source["source_line_numbers"]) != n:
        raise ValueError("Reference contour count differs from the captured source identity")
    mode = record.get("filter_mode", "hungarian")
    rounds = {int(k): v for k, v in record["rounds"].items()}
    if ref not in rounds:
        raise ValueError("QC assignment is missing the reference round")
    other = sorted(set(rounds)-{ref})
    if mode == "registration_based":
        other = list(record.get("comparison_rounds", other))
    threshold = record["change_threshold"] if mode == "registration_based" else record["iou_threshold"]
    for i, line in enumerate(source["source_line_numbers"]):
        identity = {key: source[key] for key in ("reference_image_name", "reference_contour_sha256", "reference_contour_path")}
        identity.update(reference_source_line_number=line, reference_id=i+1, reference_round=ref, patch_key=record["patch_key"])
        evidence, edge = [], None
        if mode == "hungarian":
            edge = False
            edge_known = True
            for r, values in sorted(rounds.items()):
                matches = [j for j, rid in enumerate(values["reference_ids"]) if int(rid) == i+1]
                if len(matches) > 1:
                    raise ValueError("QC one-to-one assignment contains duplicate reference IDs")
                j = matches[0] if matches else None
                edge_round = values.get("is_edge", [None]*len(values["reference_ids"]))[j] if j is not None else None
                if j is not None and edge_round is None:
                    edge_known = False
                edge |= bool(edge_round)
                evidence.append(dict(identity, round=r, is_reference=int(r == ref),
                                     is_filter_round=int(r in other),
                                     score=values["ious"][j] if j is not None else None,
                                     passes_round=int(j is not None), matched_contour_index=j,
                                     is_edge_round=None if edge_round is None else int(edge_round)))
            if not edge_known:
                edge = None
            passing = sum(v["passes_round"] for v in evidence if v["round"] in other)
            status = "not_evaluable" if edge is not False else ("passed" if passing == len(other) else "failed")
            reason = "is_edge" if edge else ("edge_status_unavailable" if edge is None else ("" if status == "passed" else "missing_passing_match"))
        else:
            edge = _at(record, "is_edge", i)
            for r, values in sorted(rounds.items()):
                name = "mean_change_score" if mode == "registration_based" else "reference_coverage"
                score = values[name][i]
                passing_round = None if score is None else int(score <= threshold if mode == "registration_based" else score > threshold)
                evidence.append(dict(identity, round=r, is_reference=int(r == ref), score=score,
                                     is_filter_round=int(r in other),
                                     passes_round=passing_round, matched_contour_index=None,
                                     is_edge_round=int(edge) if r == ref and edge is not None else None))
            passing = sum(v["passes_round"] or 0 for v in evidence if v["round"] in other)
            raw_status = record["statuses"][i]
            empty = (_at(record, "reference_area_pixels", i) or 0) == 0
            if edge is None or edge or empty or raw_status == "not_evaluable":
                status = "not_evaluable"
                reason = "is_edge" if edge else ("edge_status_unavailable" if edge is None else ("empty_reference" if empty else "not_evaluable"))
            else:
                status = "passed" if i+1 in record["kept_reference_ids"] else "failed"
                reason = "" if status == "passed" else ("high_change" if mode == "registration_based" else "insufficient_round_coverage")
        row = dict(identity, is_edge=None if edge is None else int(edge),
                   is_persistent=int(status == "passed") if status in {"passed", "failed"} else None,
                   qc_pass=int(status == "passed"), qc_status=status, qc_exclusion_reason=reason,
                   qc_filter_mode=mode, qc_score_metric=record.get("score_metric", "iou"),
                   qc_score_aggregation=record.get("score_aggregation", ""), qc_threshold=threshold,
                   qc_threshold_rule=record.get("threshold_rule", ">"),
                   qc_score=_at(record, "change_score", i), qc_mean_change_score=_at(record, "mean_change_score", i),
                   qc_max_change_score=_at(record, "max_change_score", i), qc_valid_area_fraction=_at(record, "valid_area_fraction", i),
                   qc_n_comparison_rounds=len(other), qc_n_rounds_passing=passing,
                   qc_required_other_rounds=record.get("min_other_rounds", len(other)))
        yield row, evidence


def export_myonis_4i(output_dir, summary):
    """Publish only the active/main method. Manifest is written last with checksums.

    Old assignments without a captured source checksum are never silently joined
    to today's contours. They are listed as legacy_identity_missing and require a
    new QC run to establish identity. New --summary-only runs need no source files.
    """
    output_dir = Path(output_dir)
    folder = output_dir/"myonis_4i"
    folder.mkdir(exist_ok=True)
    completed = {entry["patch_key"]: entry for entry in summary.get("completed", [])}
    failed = {entry["patch_key"]: entry for entry in summary.get("failed", [])}
    incomplete = {entry["patch_key"]: entry for entry in summary.get("discovery", {}).get("incomplete_patches", [])}
    sources = summary.get("reference_sources", {})
    keys = sorted(set(summary.get("selected_patch_keys", [])) | set(completed) | set(failed) | set(incomplete))
    columns = {"nucleus_qc.csv": IDENTITY+FLAGS, "round_qc.csv": ROUND_COLUMNS, "patch_qc.csv": PATCH_COLUMNS}
    handles, writers, counts = {}, {}, dict(objects=0, rounds=0, patches=0)
    temporary = {name: folder/(name+".tmp") for name in columns}
    try:
        for name, fields in columns.items():
            handles[name] = temporary[name].open("w", newline="", encoding="utf-8")
            writers[name] = csv.DictWriter(handles[name], fieldnames=fields)
            writers[name].writeheader()
        for key in keys:
            record = None
            if key in completed:
                record = json.loads((output_dir/completed[key]["assignments"]).read_text())
                if record["patch_key"] != key or int(record["reference_round"]) != int(summary["settings"]["reference_round"]):
                    raise ValueError("QC assignment identity differs from run summary")
            source = (record or {}).get("reference_source") or sources.get(key)
            status = "complete" if record else ("failed" if key in failed else ("skipped_incomplete" if key in incomplete else "not_completed"))
            if record and not record.get("reference_source"):
                source, status = None, "legacy_identity_missing"
            patch = dict(patch_key=key, reference_round=summary["settings"]["reference_round"], qc_patch_status=status,
                         detail=failed.get(key, {}).get("error", ""))
            if source:
                patch.update({k: source[k] for k in ["reference_image_name", "reference_contour_sha256", "reference_contour_path"]})
                patch["reference_count"] = len(source["source_line_numbers"])
            if record:
                patch.update(height=record["shape"][0], width=record["shape"][1])
            writers["patch_qc.csv"].writerow(patch)
            counts["patches"] += 1
            if source and record:
                for row, evidence in object_rows(record, source):
                    writers["nucleus_qc.csv"].writerow(row)
                    writers["round_qc.csv"].writerows(evidence)
                    counts["objects"] += 1
                    counts["rounds"] += len(evidence)
            elif source:
                for i, line in enumerate(source["source_line_numbers"]):
                    row = {k: source[k] for k in ["reference_image_name", "reference_contour_sha256", "reference_contour_path"]}
                    row.update(patch_key=key, reference_round=patch["reference_round"], reference_source_line_number=line,
                               reference_id=i+1, qc_pass=0, qc_status="not_assessed", qc_exclusion_reason=status)
                    writers["nucleus_qc.csv"].writerow(row)
                    counts["objects"] += 1
        for handle in handles.values():
            handle.close()
        for name, path in temporary.items():
            path.replace(folder/name)
        manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                        source_run_status=summary.get("status", "unknown"), settings=summary["settings"], counts=counts,
                        files={name: sha256(folder/name) for name in columns},
                        definition="is_persistent means passes the configured active QC method; it is not a universal presence call. "
                                   "Blank is_persistent/is_edge means unknown; qc_pass requires evaluated persistence and no edge. "
                                   "Comparison rounds exclude the reference; registration filtering uses the selected filter rounds, including earlier rounds. "
                                   "is_filter_round marks scored comparisons; excluded registration rounds retain transforms but blank scores. "
                                   "No objects are removed or renumbered. Input contour checksums are captured during QC.")
        path = folder/"manifest.json.tmp"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        path.replace(folder/"manifest.json")
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return folder
