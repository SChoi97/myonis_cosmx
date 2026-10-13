#!/usr/bin/env python3
"""Aggregate partial or complete worker outputs into HTML, PDF, JSON, and CSV."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import html
import importlib.util
import json
import math
from pathlib import Path
import statistics

from utils.config import load_config, config_as_dict
from utils.manifests import (
    approved_selections,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from utils.provenance import (
    LOCAL_TIMEZONE,
    command_output,
    start_task,
    task_environment,
    timestamp_values,
)
from utils.registry import ACQUISITIONS, PLATES, configured_well_tasks


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #20262e; }
h1, h2, h3 { color: #17324d; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 0.88rem; }
th, td { border: 1px solid #ccd4dc; padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }
th { background: #eaf0f5; position: sticky; top: 0; }
.pass { color: #176b3a; font-weight: 600; } .fail { color: #a52727; font-weight: 600; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; }
.card { border: 1px solid #ccd4dc; border-radius: 8px; padding: 1rem; background: #f8fafc; }
.metric { font-size: 1.55rem; font-weight: 700; color: #17324d; }
img { max-width: 100%; height: auto; border: 1px solid #ccd4dc; }
code, pre { background: #f3f5f7; padding: 0.15rem 0.3rem; } pre { overflow-x: auto; padding: 1rem; }
.small { color: #596775; font-size: 0.85rem; }
"""


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def html_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "<p class='small'>No records were available.</p>"
    columns = columns or list(rows[0])
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            css_class = "pass" if str(value) in {"pass", "complete"} else (
                "fail" if str(value) in {"fail", "quarantined"} else ""
            )
            cells.append(
                f"<td class='{css_class}'>{html.escape(str(value if value is not None else ''))}</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def read_available_json(paths) -> list[dict]:
    rows = []
    for path in sorted(paths):
        try:
            rows.append(read_json(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            rows.append({"status": "unreadable", "path": str(path), "error": str(error)})
    return rows


def duration_rows(groups: dict[str, list[dict]]) -> list[dict]:
    result = []
    for stage, rows in groups.items():
        durations = []
        for row in rows:
            try:
                value = float(row.get("environment", {}).get("duration_seconds"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                durations.append(value)
        result.append(
            {
                "stage": stage,
                "task_manifest_count": len(rows),
                "timed_task_count": len(durations),
                "sum_task_seconds": sum(durations),
                "mean_task_seconds": statistics.fmean(durations) if durations else "",
                "max_task_seconds": max(durations) if durations else "",
            }
        )
    return result


def summarize_shifts(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("plate_name", "")), str(row.get("measurement", ""))), []
        ).append(row)
    summaries = []
    for (plate_name, measurement), values in sorted(grouped.items()):
        magnitudes = []
        correlations = []
        fallback_count = 0
        failed_count = 0
        for row in values:
            try:
                magnitude = float(row.get("shift_magnitude", "nan"))
                correlation = float(row.get("post_registration_correlation", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(magnitude):
                magnitudes.append(magnitude)
            if math.isfinite(correlation):
                correlations.append(correlation)
            fallback_count += row.get("shift_source") == "robust_field_consensus"
            failed_count += row.get("global_qc_status") != "pass"
        summaries.append(
            {
                "plate_name": plate_name,
                "measurement": measurement,
                "registration_unit_count": len(values),
                "well_count": len(
                    {
                        (row.get("plate_name"), row.get("row"), row.get("column"))
                        for row in values
                    }
                ),
                "field_count": len(
                    {row.get("field_id") for row in values if row.get("field_id")}
                ),
                "shift_magnitude_mean": statistics.fmean(magnitudes) if magnitudes else "",
                "shift_magnitude_std": statistics.pstdev(magnitudes) if len(magnitudes) > 1 else 0,
                "shift_magnitude_max": max(magnitudes) if magnitudes else "",
                "post_registration_correlation_mean": (
                    statistics.fmean(correlations) if correlations else ""
                ),
                "consensus_fallback_count": fallback_count,
                "failed_global_qc_count": failed_count,
            }
        )
    return summaries


def collect_slurm_accounting(run_manifest: dict) -> list[dict]:
    graph = run_manifest.get("slurm_job_graph", {})
    job_ids = [str(value) for value in graph.values() if value]
    if not job_ids:
        return []
    output = command_output(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,ReqMem",
        ]
    )
    rows = []
    for line in output.splitlines():
        values = line.split("|")
        if len(values) < 8:
            continue
        rows.append(
            dict(
                zip(
                    (
                        "job_id",
                        "job_name",
                        "state",
                        "exit_code",
                        "elapsed",
                        "allocated_cpus",
                        "max_rss",
                        "requested_memory",
                    ),
                    values[:8],
                )
            )
        )
    return rows


def well_key(status: dict) -> str:
    try:
        row = int(status.get("row", 0))
        column = int(status.get("column", 0))
    except (TypeError, ValueError):
        row, column = 0, 0
    return f"{status.get('plate_name', 'unknown')}__r{row:02d}_c{column:02d}"


def create_per_well_report(run_dir: Path, status: dict) -> Path:
    key = well_key(status)
    source_dir = run_dir / "manifests" / "wells" / key
    output_dir = run_dir / "reports" / "wells" / key
    output_dir.mkdir(parents=True, exist_ok=True)
    shifts = read_csv(source_dir / "registration_shifts.csv")
    fields = read_csv(source_dir / "field_qc.csv")
    quarantine = read_csv(source_dir / "quarantine.csv")
    failed_fields = [row for row in fields if row.get("status") == "fail"]
    figure_dir = source_dir / "figures"
    images = []
    for figure in sorted(figure_dir.glob("*.png")):
        relative = Path("../../../manifests/wells") / key / "figures" / figure.name
        images.append(
            f"<h3>{html.escape(figure.stem.replace('_', ' ').title())}</h3>"
            f"<a href='{relative.as_posix()}'><img src='{relative.as_posix()}'></a>"
        )
    body = f"""
    <p><a href='../../../run_report.html'>&larr; Run report</a></p>
    <h1>{html.escape(key)}</h1>
    <div class='card-grid'>
      <div class='card'><div class='metric'>{html.escape(str(status.get('status')))}</div>Status</div>
      <div class='card'><div class='metric'>{html.escape(str(status.get('common_overlap_fraction', '')))}</div>Common overlap</div>
      <div class='card'><div class='metric'>{html.escape(str(status.get('patch_count_retained', 0)))}</div>Retained patches</div>
      <div class='card'><div class='metric'>{len(failed_fields)}</div>Failed field observations</div>
    </div>
    <h2>Registration shifts</h2>
    {html_table(shifts, ['cohort_id','cohort_reference_measurement','measurement','field_id','shift_y','shift_x','shift_magnitude','shift_source','overlap_fraction','post_registration_correlation','global_qc_status','global_qc_reasons','initial_shift_y','initial_shift_x','initial_global_qc_status','initial_global_qc_reasons'])}
    <h2>Failed field observations</h2>
    {html_table(failed_fields, ['measurement','field_id','grid_row','grid_column','correlation','residual_shift_y','residual_shift_x','residual_magnitude','residual_consensus_disagreement','failure_reasons'])}
    <h2>Quarantine</h2>{html_table(quarantine)}
    <h2>Figures</h2>{''.join(images)}
    <h2>All field QC</h2>
    <p><a href='../../../manifests/wells/{html.escape(key)}/field_qc.csv'>Download field_qc.csv</a></p>
    """
    output = output_dir / "index.html"
    output.write_text(page(key, body), encoding="utf-8")
    return output


def create_pdf(
    output: Path,
    run_manifest: dict,
    config: dict,
    selections: list[dict],
    well_statuses: list[dict],
    shifts: list[dict],
    field_qc: list[dict],
    quarantine: list[dict],
    run_dir: Path,
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise ImportError("reportlab is required to create run_report.pdf.") from error

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredTitle", parent=styles["Title"], alignment=TA_CENTER))
    document = SimpleDocTemplate(
        str(output), pagesize=landscape(A4), rightMargin=12 * mm, leftMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title=f"4i preprocessing {run_manifest.get('run_id', '')}",
    )
    story = [
        Paragraph("4i preprocessing run report", styles["CenteredTitle"]),
        Paragraph(str(run_manifest.get("run_id", "unknown run")), styles["Heading2"]),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Started: {run_manifest.get('started', {}).get('local', '')}<br/>"
            f"Completed/report generated: {datetime.now(LOCAL_TIMEZONE).isoformat()}<br/>"
            f"Run status: {run_manifest.get('status', '')}<br/>"
            f"Host: {run_manifest.get('hostname', '')}<br/>"
            f"Git commit: {run_manifest.get('git', {}).get('commit', '')}",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Execution summary", styles["Heading1"]),
        Preformatted(
            json.dumps(
                {
                    "counts": run_manifest.get("counts", {}),
                    "stage_durations": run_manifest.get("stage_durations", []),
                    "warnings": run_manifest.get("warnings", []),
                    "failed_slurm_jobs": run_manifest.get("failed_slurm_jobs", []),
                    "manual_overrides": run_manifest.get("manual_overrides", []),
                    "image_geometry": run_manifest.get("image_geometry", {}),
                    "snapshots": run_manifest.get("snapshots", {}),
                },
                indent=2,
            ),
            styles["Code"],
        ),
        Paragraph("Configuration", styles["Heading1"]),
        Preformatted(json.dumps(config, indent=2), styles["Code"]),
    ]

    def table(data, widths=None, font_size=6, vertical_padding=None):
        item = Table(data, repeatRows=1, colWidths=widths)
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f0")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        if vertical_padding is not None:
            commands.extend(
                [
                    ("TOPPADDING", (0, 0), (-1, -1), vertical_padding),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), vertical_padding),
                ]
            )
        item.setStyle(TableStyle(commands))
        return item

    code_hash_path = run_dir / "provenance" / "code_sha256.json"
    code_hashes = read_json(code_hash_path) if code_hash_path.is_file() else {}
    code_hash_data = [["Production file", "SHA256"]]
    code_hash_data.extend([path, digest] for path, digest in code_hashes.items())
    story.extend(
        [
            PageBreak(),
            Paragraph("Production script hashes", styles["Heading1"]),
            table(code_hash_data, [110 * mm, 145 * mm], 5),
            PageBreak(),
            Paragraph("Illumination method selections", styles["Heading1"]),
        ]
    )

    selection_data = [["Acquisition", "Measurement", "Marker", "Method", "Model SHA256"]]
    selection_data.extend(
        [
            row.get("acquisition_id", ""),
            row.get("measurement", ""),
            row.get("marker", ""),
            row.get("chosen_method", ""),
            (
                row.get("basic_model_sha256", "")
                if row.get("chosen_method") == "BaSiC"
                else row.get("harmony_model_sha256", row.get("harmony_profile_sha256", ""))
            )[:18],
        ]
        for row in selections
    )
    story.extend([table(selection_data, [62 * mm, 20 * mm, 35 * mm, 28 * mm, 42 * mm], 7), PageBreak()])
    story.append(Paragraph("Well outcomes", styles["Heading1"]))
    well_data = [["Plate/well", "Status", "Overlap", "Failed fields", "Retained patches", "Crop bounds"]]
    for status in well_statuses:
        well_data.append(
            [
                well_key(status),
                status.get("status", ""),
                status.get("common_overlap_fraction", ""),
                ",".join(map(str, status.get("failed_field_ids", []))),
                status.get("patch_count_retained", 0),
                str(status.get("crop_bounds", "")),
            ]
        )
    story.extend([table(well_data, [48 * mm, 25 * mm, 22 * mm, 55 * mm, 25 * mm, 62 * mm], 6), PageBreak()])
    patch_level_report = any(
        row.get("registration_level") == "patch_level" for row in shifts
    )
    story.append(
        Paragraph(
            (
                "Patch-level registration summary (aggregated)"
                if patch_level_report
                else "Complete registration shift table"
            ),
            styles["Heading1"],
        )
    )
    if patch_level_report:
        story.extend(
            [
                Paragraph(
                    "Patch-level registration metrics were aggregated from "
                    "independent field-level registrations for this concise "
                    "run summary. Each original 1080 x 1080 Phenix field has "
                    "its own exact (dy, dx) shift; no mean shift was applied "
                    "to the images. Rows below group fields across wells by "
                    "plate and measurement. Mean magnitude, SD, maximum, mean "
                    "NCC, and failed counts therefore describe the distribution "
                    "of field-specific results. Exact shifts are retained in "
                    "each well's registration_shifts.csv and HTML report.",
                    styles["BodyText"],
                ),
                Spacer(1, 3 * mm),
            ]
        )
        shift_data = [["Plate", "M", "Fields (n)", "Mean magnitude", "SD", "Maximum", "Mean NCC", "Failed"]]
        shift_data.extend(
            [
                row.get("plate_name", ""),
                row.get("measurement", ""),
                row.get("registration_unit_count", ""),
                row.get("shift_magnitude_mean", ""),
                row.get("shift_magnitude_std", ""),
                row.get("shift_magnitude_max", ""),
                row.get("post_registration_correlation_mean", ""),
                row.get("failed_global_qc_count", ""),
            ]
            for row in summarize_shifts(shifts)
        )
    else:
        shift_data = [["Plate", "Row", "Col", "M", "dy", "dx", "Magnitude", "Source", "Overlap", "NCC", "QC"]]
        shift_data.extend(
            [
                row.get("plate_name", ""), row.get("row", ""), row.get("column", ""),
                row.get("measurement", ""), row.get("shift_y", ""), row.get("shift_x", ""),
                row.get("shift_magnitude", ""), row.get("shift_source", ""),
                row.get("overlap_fraction", ""), row.get("post_registration_correlation", ""),
                row.get("global_qc_status", ""),
            ]
            for row in shifts
        )
    story.extend(
        [
            table(
                shift_data,
                font_size=4.5 if patch_level_report else 5,
                vertical_padding=1.5 if patch_level_report else None,
            ),
            PageBreak(),
        ]
    )
    failed_fields = [row for row in field_qc if row.get("status") == "fail"]
    story.append(Paragraph("Failed field observations", styles["Heading1"]))
    failed_data = [["Plate", "Well", "M", "Field", "NCC", "Residual", "Reasons"]]
    failed_data.extend(
        [
            row.get("plate_name", ""), f"r{row.get('row','')}c{row.get('column','')}",
            row.get("measurement", ""), row.get("field_id", ""), row.get("correlation", ""),
            row.get("residual_magnitude", ""), row.get("failure_reasons", ""),
        ]
        for row in failed_fields
    )
    story.extend([table(failed_data, font_size=6), PageBreak()])
    story.append(Paragraph("Quarantine manifest", styles["Heading1"]))
    quarantine_data = [["Plate", "Well", "Scope", "Patch", "Failed fields", "Reason"]]
    quarantine_data.extend(
        [
            row.get("plate_name", ""), f"r{row.get('row','')}c{row.get('column','')}",
            row.get("scope", ""), row.get("patch_id", ""), row.get("failed_field_ids", ""),
            row.get("reason", ""),
        ]
        for row in quarantine
    )
    story.append(table(quarantine_data, font_size=6))
    flagged_wells = [status for status in well_statuses if status.get("status") != "complete" or status.get("failed_field_ids")]
    for status in flagged_wells:
        key = well_key(status)
        heatmap = run_dir / "manifests" / "wells" / key / "figures" / "field_qc_heatmaps.png"
        story.extend([PageBreak(), Paragraph(f"Flagged well: {key}", styles["Heading1"])])
        story.append(Preformatted(json.dumps(status, indent=2), styles["Code"]))
        if heatmap.is_file():
            story.append(Image(str(heatmap), width=240 * mm, height=180 * mm, kind="proportional"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(12 * mm, 7 * mm, str(run_manifest.get("run_id", "")))
        canvas.drawRightString(285 * mm, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_started, report_monotonic_start = start_task()
    config_object = load_config(run_dir / "provenance" / "pipeline_config.json")
    config = config_as_dict(config_object)
    selections_dict = approved_selections(
        run_dir / "provenance" / "illumination_method_selection.csv"
    )
    selections = [selections_dict[item.acquisition_id] for item in ACQUISITIONS]
    run_manifest = read_json(run_dir / "run_manifest.json")
    smoke_statuses = read_available_json(
        (run_dir / "manifests" / "smoke").glob("*/smoke_status.json")
    )
    illumination = read_available_json((run_dir / "manifests" / "illumination").glob("*.json"))
    well_statuses = read_available_json((run_dir / "manifests" / "wells").glob("*/well_status.json"))
    patch_statuses = read_available_json((run_dir / "manifests" / "wells").glob("*/patch_status.json"))
    clipping_statuses = read_available_json((run_dir / "manifests" / "clipping").glob("*.json"))
    cleanup_statuses = read_available_json(
        (run_dir / "manifests" / "cleanup").glob("*.json")
    )
    # Add plate/well identity to per-well CSV rows before root aggregation.
    shifts = []
    field_qc = []
    quarantine = []
    patches = []
    patch_level_regions = []
    for status in well_statuses:
        key = well_key(status)
        source = run_dir / "manifests" / "wells" / key
        identity = {
            "plate_name": status.get("plate_name"),
            "row": status.get("row"),
            "column": status.get("column"),
            "registration_level": status.get("registration_level", "field_level"),
        }
        shifts.extend({**identity, **row} for row in read_csv(source / "registration_shifts.csv"))
        field_qc.extend({**identity, **row} for row in read_csv(source / "field_qc.csv"))
        quarantine.extend({**identity, **row} for row in read_csv(source / "quarantine.csv"))
        patches.extend({**identity, **row} for row in read_csv(source / "patches.csv"))
        patch_level_regions.extend(
            {**identity, **row}
            for row in read_csv(source / "patch_level_regions.csv")
        )
    clipping = []
    for path in sorted((run_dir / "manifests" / "clipping").glob("*.csv")):
        clipping.extend(read_csv(path))
    replaced_outputs = []
    for path in sorted((run_dir / "manifests" / "replaced_outputs").glob("*.csv")):
        replaced_outputs.extend(read_csv(path))
    corrections = []
    for task in illumination:
        acquisition = task.get("acquisition", {})
        for plate_result in task.get("plates", []):
            corrections.append(
                {
                    "acquisition_id": acquisition.get("acquisition_id"),
                    "measurement": acquisition.get("measurement"),
                    "source_channel": acquisition.get("source_channel"),
                    "marker": acquisition.get("marker"),
                    "method": task.get("method"),
                    "plate_name": plate_result.get("plate_name"),
                    **plate_result,
                }
            )
    crops = [
        {
            "plate_name": status.get("plate_name"),
            "row": status.get("row"),
            "column": status.get("column"),
            "registration_level": status.get("registration_level", "field_level"),
            "status": status.get("status"),
            "common_bounds": status.get("common_bounds"),
            "common_overlap_fraction": status.get("common_overlap_fraction"),
            "crop_bounds": status.get("crop_bounds"),
            "crop_area_fraction": status.get("crop_area_fraction"),
            "patch_count_total": status.get("patch_count_total"),
            "patch_count_retained": status.get("patch_count_retained"),
            "whole_well_quarantine": status.get("whole_well_quarantine"),
        }
        for status in well_statuses
    ]
    shift_summaries = summarize_shifts(shifts)
    stage_durations = duration_rows(
        {
            "real_data_smoke_test": smoke_statuses,
            "illumination_correction": illumination,
            "registration_and_crop": well_statuses,
            "patch_generation": patch_statuses,
            "clip_and_export_png": clipping_statuses,
            "intermediate_cleanup": cleanup_statuses,
        }
    )
    clipping_limits = [
        {
            "acquisition_id": item.get("acquisition", {}).get("acquisition_id"),
            "measurement": item.get("acquisition", {}).get("measurement"),
            "marker": item.get("acquisition", {}).get("marker"),
            **item.get("clip_limits", {}),
            "input_patch_count": item.get("input_patch_count"),
            "written_png_count": item.get("written_png_count"),
        }
        for item in clipping_statuses
    ]
    override_rows = read_csv(run_dir / "provenance" / "registration_overrides.csv")
    slurm_accounting = collect_slurm_accounting(run_manifest)
    failed_slurm_jobs = [
        row
        for row in slurm_accounting
        if row.get("state")
        and not row["state"].startswith(("COMPLETED", "RUNNING", "PENDING"))
    ]
    aggregate_dir = run_dir / "manifests" / "aggregate"
    write_csv(aggregate_dir / "corrections.csv", corrections)
    write_csv(aggregate_dir / "real_data_smoke_tests.csv", smoke_statuses)
    write_csv(aggregate_dir / "registration_shifts.csv", shifts)
    write_csv(aggregate_dir / "registration_shift_summary.csv", shift_summaries)
    write_csv(aggregate_dir / "field_qc.csv", field_qc)
    write_csv(aggregate_dir / "crops.csv", crops)
    write_csv(aggregate_dir / "quarantine_manifest.csv", quarantine)
    write_csv(aggregate_dir / "patch_manifest.csv", patches)
    write_csv(aggregate_dir / "patch_level_regions.csv", patch_level_regions)
    write_csv(aggregate_dir / "clipping_manifest.csv", clipping)
    write_csv(aggregate_dir / "clipping_limits.csv", clipping_limits)
    write_csv(aggregate_dir / "intermediate_cleanup.csv", cleanup_statuses)
    write_csv(aggregate_dir / "replaced_outputs.csv", replaced_outputs)
    write_csv(aggregate_dir / "stage_durations.csv", stage_durations)
    write_csv(aggregate_dir / "slurm_accounting.csv", slurm_accounting)
    for status in well_statuses:
        create_per_well_report(run_dir, status)
    well_counts = Counter(str(item.get("status")) for item in well_statuses)
    field_counts = Counter(str(item.get("status")) for item in field_qc)
    passed_wells = sum(
        item.get("status") == "complete" and not item.get("failed_field_ids")
        for item in well_statuses
    )
    field_quarantined_wells = sum(
        item.get("status") == "complete" and bool(item.get("failed_field_ids"))
        for item in well_statuses
    )
    unique_quarantined_patches = {
        (row.get("plate_name"), row.get("row"), row.get("column"), row.get("patch_id"))
        for row in quarantine
        if row.get("scope") == "patch"
    }
    expected_patch_files = sum(
        int(
            item.get("expected_quantitative_patch_files")
            if item.get("expected_quantitative_patch_files") is not None
            else int(item.get("patch_count_retained") or 0) * len(ACQUISITIONS)
        )
        for item in well_statuses
    )
    expected_wells = len(configured_well_tasks(config_object))
    cleanup_graph_keys = {"cleanup_corrected", "cleanup_registered_crops"}
    cleanup_expected = cleanup_graph_keys.issubset(
        set(run_manifest.get("slurm_job_graph", {}))
    )
    cleanup_complete = (
        not cleanup_expected
        or (
            len(cleanup_statuses) == 2
            and all(
                item.get("status") in {"complete", "skipped_disabled"}
                for item in cleanup_statuses
            )
        )
    )
    pipeline_complete = (
        len(smoke_statuses) == len(PLATES)
        and all(item.get("status") == "pass" for item in smoke_statuses)
        and len(illumination) == len(ACQUISITIONS)
        and len(well_statuses) == expected_wells
        and len(patch_statuses) == expected_wells
        and len(clipping_statuses) == len(ACQUISITIONS)
        and all(item.get("status") == "complete" for item in illumination)
        and all(item.get("status") in {"complete", "quarantined"} for item in well_statuses)
        and all(
            item.get("status") in {"complete", "skipped_quarantined_well"}
            for item in patch_statuses
        )
        and all(item.get("status") == "complete" for item in clipping_statuses)
        and cleanup_complete
        and len(patches) == expected_patch_files
        and len(clipping) == expected_patch_files
        and not failed_slurm_jobs
    )
    warnings = []
    for label, observed, expected in (
        ("real-data smoke-test manifests", len(smoke_statuses), len(PLATES)),
        ("illumination task manifests", len(illumination), len(ACQUISITIONS)),
        ("registration task manifests", len(well_statuses), expected_wells),
        ("patch task manifests", len(patch_statuses), expected_wells),
        ("clipping task manifests", len(clipping_statuses), len(ACQUISITIONS)),
    ):
        if observed != expected:
            warnings.append(f"Expected {expected} {label}; found {observed}.")
    if cleanup_expected and len(cleanup_statuses) != 2:
        warnings.append(
            f"Expected 2 intermediate-cleanup manifests; found {len(cleanup_statuses)}."
        )
    if len(patch_statuses) == expected_wells and len(patches) != expected_patch_files:
        warnings.append(
            f"Expected {expected_patch_files} retained quantitative patch files from "
            f"manifests; found {len(patches)}."
        )
    failed_smoke = [item for item in smoke_statuses if item.get("status") != "pass"]
    if failed_smoke:
        warnings.append(
            f"{len(failed_smoke)} real-data plate smoke test(s) did not pass."
        )
    if failed_slurm_jobs:
        warnings.append(f"Slurm reported {len(failed_slurm_jobs)} failed/cancelled job records.")
    pdf_dependency_available = importlib.util.find_spec("reportlab") is not None
    if not pdf_dependency_available:
        warnings.append(
            "PDF report was skipped because reportlab is not installed; "
            "run_report.html and all machine-readable manifests remain available."
        )
    report_time = timestamp_values()
    code_hash_path = run_dir / "provenance" / "code_sha256.json"
    code_hashes = read_json(code_hash_path) if code_hash_path.is_file() else {}
    run_manifest.update(
        {
            "status": "complete" if pipeline_complete else "partial",
            "ended": report_time,
            "report_generated": report_time,
            "stage_durations": stage_durations,
            "warnings": warnings,
            "failed_slurm_jobs": failed_slurm_jobs,
            "manual_overrides": override_rows,
            "counts": {
                "real_data_smoke_tests": len(smoke_statuses),
                "passed_real_data_smoke_tests": sum(
                    item.get("status") == "pass" for item in smoke_statuses
                ),
                "illumination_tasks": len(illumination),
                "well_statuses": len(well_statuses),
                "passed_wells": passed_wells,
                "field_quarantined_wells": field_quarantined_wells,
                "whole_well_quarantined": well_counts.get("quarantined", 0),
                "unique_patch_quarantines": len(unique_quarantined_patches),
                "patch_tasks": len(patch_statuses),
                "clipping_tasks": len(clipping_statuses),
                "intermediate_cleanup_tasks": len(cleanup_statuses),
                "intermediate_directories_removed": sum(
                    bool(item.get("removed")) for item in cleanup_statuses
                ),
                "expected_quantitative_patch_files": expected_patch_files,
                "quantitative_patch_files": len(patches),
                "png_files": len(clipping),
                "replaced_stale_output_files": len(replaced_outputs),
                "failed_field_observations": field_counts.get("fail", 0),
                "not_evaluable_field_observations": field_counts.get("not_evaluable", 0),
            },
        }
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    summary_cards = [
        (
            "Run mode",
            f"{config_object.registration.level} / "
            f"{'TEST' if config_object.test_mode else 'PRODUCTION'}",
        ),
        ("Real-data smoke tests", f"{sum(item.get('status') == 'pass' for item in smoke_statuses)}/2"),
        ("Illumination tasks", f"{len(illumination)}/40"),
        ("Well registrations", f"{len(well_statuses)}/{expected_wells}"),
        ("Complete wells", str(well_counts.get("complete", 0))),
        ("Quarantined wells", str(well_counts.get("quarantined", 0))),
        ("Field-quarantined wells", str(field_quarantined_wells)),
        ("Patch quarantines", str(len(unique_quarantined_patches))),
        ("Failed field observations", str(field_counts.get("fail", 0))),
        ("Quantitative patches", str(len(patches))),
        ("PNG derivatives", str(len(clipping))),
        (
            "Intermediates removed",
            str(sum(bool(item.get("removed")) for item in cleanup_statuses)),
        ),
    ]
    cards = "".join(
        f"<div class='card'><div class='metric'>{html.escape(value)}</div>{html.escape(label)}</div>"
        for label, value in summary_cards
    )
    well_rows = []
    for status in sorted(well_statuses, key=well_key):
        key = well_key(status)
        well_rows.append(
            {
                "well": f"<a href='reports/wells/{key}/index.html'>{key}</a>",
                "status": status.get("status"),
                "common_overlap_fraction": status.get("common_overlap_fraction"),
                "crop_area_fraction": status.get("crop_area_fraction"),
                "failed_field_ids": ",".join(map(str, status.get("failed_field_ids", []))),
                "patch_count_retained": status.get("patch_count_retained"),
            }
        )
    # Preserve links in this one table without allowing arbitrary HTML elsewhere.
    well_table = "<table><thead><tr><th>Well</th><th>Status</th><th>Common overlap</th><th>Final crop area</th><th>Failed fields</th><th>Retained patches</th></tr></thead><tbody>"
    for row in well_rows:
        well_table += (
            f"<tr><td>{row['well']}</td><td>{html.escape(str(row['status']))}</td>"
            f"<td>{html.escape(str(row['common_overlap_fraction']))}</td>"
            f"<td>{html.escape(str(row['crop_area_fraction']))}</td>"
            f"<td>{html.escape(str(row['failed_field_ids']))}</td>"
            f"<td>{html.escape(str(row['patch_count_retained']))}</td></tr>"
        )
    well_table += "</tbody></table>"
    shift_summary_explanation = (
        "<p><strong>Patch-level registration summary:</strong> Each original "
        "1080 x 1080 Phenix field was registered independently and retains its "
        "own exact (dy, dx) shift. The table below aggregates those field-specific "
        "metrics across wells by plate and measurement; the mean and SD were not "
        "used as image transformations. Exact shifts are available in "
        "<code>registration_shifts.csv</code> and the per-well HTML reports.</p>"
        if config_object.registration.level == "patch_level"
        else ""
    )
    report_body = f"""
    <h1>4i preprocessing run report</h1>
    <p><code>{html.escape(str(run_manifest.get('run_id', 'unknown')))}</code></p>
    <p>Started: {html.escape(str(run_manifest.get('started', {}).get('local', '')))}<br>
    Report generated: {html.escape(report_time['local'])}<br>
    Status: {html.escape(str(run_manifest.get('status')))}</p>
    <div class='card-grid'>{cards}</div>
    <h2>Warnings and failed jobs</h2>
    {html_table([{'warning': warning} for warning in warnings])}
    {html_table(failed_slurm_jobs)}
    <h2>Stage durations</h2>{html_table(stage_durations)}
    <h2>Real-data smoke tests</h2>{html_table(smoke_statuses)}
    <h2>Pipeline configuration</h2><pre>{html.escape(json.dumps(config, indent=2))}</pre>
    <h2>Image geometry</h2><pre>{html.escape(json.dumps(run_manifest.get('image_geometry', {}), indent=2))}</pre>
    <h2>Production script hashes</h2>
    {html_table([{'production_file': path, 'sha256': digest} for path, digest in code_hashes.items()])}
    <h2>Illumination selections</h2>
    {html_table(selections, ['acquisition_id','measurement','source_channel','marker','chosen_method','review_status','training_field_count','basic_model_sha256','harmony_model_sha256'])}
    <h2>Registration shift distributions</h2>
    {shift_summary_explanation}
    {html_table(shift_summaries)}
    <h2>Clipping limits and output counts</h2>{html_table(clipping_limits)}
    <h2>Intermediate cleanup</h2>{html_table(cleanup_statuses)}
    <h2>Manual registration overrides</h2>{html_table(override_rows)}
    <h2>Wells</h2>{well_table}
    <h2>Machine-readable manifests</h2>
    <ul>
      <li><a href='run_manifest.json'>run_manifest.json</a></li>
      <li><a href='manifests/aggregate/registration_shifts.csv'>registration_shifts.csv</a></li>
      <li><a href='manifests/aggregate/real_data_smoke_tests.csv'>real_data_smoke_tests.csv</a></li>
      <li><a href='manifests/aggregate/registration_shift_summary.csv'>registration_shift_summary.csv</a></li>
      <li><a href='manifests/aggregate/field_qc.csv'>field_qc.csv</a></li>
      <li><a href='manifests/aggregate/corrections.csv'>corrections.csv</a></li>
      <li><a href='manifests/aggregate/crops.csv'>crops.csv</a></li>
      <li><a href='manifests/aggregate/quarantine_manifest.csv'>quarantine_manifest.csv</a></li>
      <li><a href='manifests/aggregate/patch_manifest.csv'>patch_manifest.csv</a></li>
      <li><a href='manifests/aggregate/patch_level_regions.csv'>patch_level_regions.csv</a></li>
      <li><a href='manifests/aggregate/clipping_manifest.csv'>clipping_manifest.csv</a></li>
      <li><a href='manifests/aggregate/clipping_limits.csv'>clipping_limits.csv</a></li>
      <li><a href='manifests/aggregate/intermediate_cleanup.csv'>intermediate_cleanup.csv</a></li>
      <li><a href='manifests/aggregate/replaced_outputs.csv'>replaced_outputs.csv</a></li>
      <li><a href='manifests/aggregate/stage_durations.csv'>stage_durations.csv</a></li>
      <li><a href='manifests/aggregate/slurm_accounting.csv'>slurm_accounting.csv</a></li>
      <li><a href='provenance/illumination_training/'>illumination training coordinate snapshots</a></li>
      <li><a href='provenance/pipeline_config.json'>pipeline configuration snapshot</a></li>
      <li><a href='provenance/illumination_method_selection.csv'>approved method-selection snapshot</a></li>
      <li><a href='provenance/illumination_method_choices.xlsx'>human review workbook snapshot</a></li>
      <li><a href='provenance/code_sha256.json'>production script hashes</a></li>
      <li><a href='provenance/git_diff.patch'>dirty-worktree diff</a></li>
      <li><a href='provenance/pip_freeze.txt'>Python package versions</a></li>
      <li><a href='provenance/conda_environment.yml'>Conda environment export</a></li>
    </ul>
    """
    (run_dir / "run_report.html").write_text(
        page("4i preprocessing run report", report_body), encoding="utf-8"
    )
    pdf_path = run_dir / "run_report.pdf"
    if pdf_dependency_available:
        try:
            create_pdf(
                pdf_path,
                run_manifest,
                config,
                selections,
                well_statuses,
                shifts,
                field_qc,
                quarantine,
                run_dir,
            )
            run_manifest["pdf_report"] = {
                "status": "complete",
                "path": str(pdf_path),
            }
        except ImportError as error:
            # A partially installed reportlab should not hide the more useful
            # HTML report or turn an afterany reporting task into a failure.
            message = f"PDF report was skipped: {error}"
            if message not in run_manifest["warnings"]:
                run_manifest["warnings"].append(message)
            run_manifest["pdf_report"] = {
                "status": "skipped_missing_dependency",
                "error": str(error),
            }
    else:
        run_manifest["pdf_report"] = {
            "status": "skipped_missing_dependency",
            "error": "reportlab is not installed",
        }
    run_manifest["report_task"] = task_environment(
        report_started, report_monotonic_start
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    print(run_dir / "run_report.html")
    if pdf_path.is_file():
        print(pdf_path)
    else:
        print("PDF report skipped; see run_report.html.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
