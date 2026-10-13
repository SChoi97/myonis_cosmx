"""Three-page measurement summary, with a complete field-exclusion appendix."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import anndata as ad
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

INK = colors.HexColor("#183448")
TEAL = colors.HexColor("#087E83")
LIGHT = colors.HexColor("#EDF4F5")
GRAY = colors.HexColor("#536570")
WIDTH, HEIGHT = A4
MARGIN = 42


class Page:
    def __init__(self, pdf, number, subtitle, dataset, total_pages=3):
        self.pdf = pdf
        self.y = HEIGHT-44
        pdf.setFillColor(TEAL); pdf.rect(MARGIN, self.y, 32, 4, fill=1, stroke=0)
        self.y -= 30
        pdf.setFillColor(INK); pdf.setFont("Helvetica-Bold", 23)
        pdf.drawString(MARGIN, self.y, "MYONIS 4i")
        self.y -= 23
        self.text(subtitle, size=13, bold=True)
        self.text(dataset, size=8, color=GRAY)
        pdf.setFont("Helvetica", 8); pdf.setFillColor(GRAY)
        pdf.drawString(MARGIN, 24, "myonis_4i_data.h5ad  |  Measurement archive and provenance")
        pdf.drawRightString(WIDTH-MARGIN, 24, f"{number} / {total_pages}")

    def text(self, text, size=9, bold=False, color=INK, gap=7):
        style = ParagraphStyle("p", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                               leading=size*1.35, textColor=color, spaceAfter=gap)
        p = Paragraph(text, style)
        _, height = p.wrap(WIDTH-2*MARGIN, 1000)
        self.y -= height
        if self.y < 43:
            raise ValueError("Report page overflow: reduce summary content")
        p.drawOn(self.pdf, MARGIN, self.y)
        self.y -= gap

    def table(self, rows, widths, font_size=8, row_padding=4):
        style = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, leading=font_size+2, textColor=INK)
        cells = [[Paragraph(escape(str(v)), style) for v in row] for row in rows]
        table = Table(cells, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .6, TEAL),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#DDE6E8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), row_padding), ("BOTTOMPADDING", (0, 0), (-1, -1), row_padding),
        ]))
        _, height = table.wrap(WIDTH-2*MARGIN, 1000)
        self.y -= height
        if self.y < 43:
            raise ValueError("Report table overflow")
        table.drawOn(self.pdf, MARGIN, self.y)
        self.y -= 12


def exclusion_pages(exclusions):
    """Paginate every reason; never silently truncate the excluded-field list."""
    style = ParagraphStyle("exclusion", fontName="Helvetica", fontSize=9, leading=12, textColor=INK)
    pages, page, used = [], [], 0
    for row in exclusions.to_dict("records"):
        label = (f"{row['plate_name']} | repeat {row['repeat']} | {row['cell_line']} | "
                 f"{row['well_id']} | field {row['field_id']} | {row['cohort_id']}")
        text = ("<b>"+escape(label)+"</b><br/>"+escape(str(row['reason']))+
                f"<br/>Excluded: the whole field ({row['excluded_patches']} spatial patches).")
        paragraph = Paragraph(text, style)
        blocks = [paragraph]
        while blocks:
            block = blocks.pop(0)
            if block.wrap(WIDTH-2*MARGIN, 100000)[1] > 550:
                pieces = block.split(WIDTH-2*MARGIN, 550)
                if len(pieces) < 2:
                    raise ValueError("Unable to paginate a field-exclusion reason")
                blocks[0:0] = pieces
                continue
            height = block.wrap(WIDTH-2*MARGIN, 10000)[1]+14
            if page and used+height > 570:
                pages.append(page); page, used = [], 0
            page.append(block); used += height
    if page:
        pages.append(page)
    return pages


def build_report(h5ad_path, output_path, run=None):
    # Finalisation already holds this exact archive in memory. Standalone report
    # regeneration still reads the saved H5AD, preserving the public entry point.
    if run is None:
        run = ad.read_h5ad(h5ad_path)
    config = json.loads(run.uns["provenance"]["config_json"])
    manifest = json.loads(run.uns["provenance"]["run_manifest_json"])
    markers = json.loads(run.uns["provenance"]["markers_json"])
    inputs, qc = run.uns["inputs"], run.uns["qc"]
    myotubes = run.uns["myotubes"]
    exclusions = qc.get("excluded_fields")
    appendix = exclusion_pages(exclusions) if exclusions is not None and len(exclusions) else []
    total_pages = 3+len(appendix)
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.pdf")
    pdf = canvas.Canvas(str(temporary), pagesize=A4)
    pdf.setTitle("MYONIS 4i run summary - "+config["dataset_id"])
    page = Page(pdf, 1, "Run summary and dataset organisation", escape(config["dataset_id"]), total_pages)
    page.text("Status: "+escape(manifest["status"].upper()), 12, True, TEAL)
    images = inputs["images"]
    assigned = int((run.obs.get("parent_myotube_uid", np.array([])) != "").sum())
    n_structures = sum(len(t) for t in run.uns["structures"].values())
    pixel = config["geometry"].get("pixel_size_um")
    rows = [["Dataset", "Result"],
            ["Plates / physical wells / field-cohorts", f"{images.plate_name.nunique()} / {images[['plate_name','well_id']].drop_duplicates().shape[0]} / {images.field_key.nunique()}"],
            ["Completed / failed field-cohorts", f"{manifest['completed_tasks']} / {manifest['failed_tasks']}"],
            ["Nuclei / myotubes / substructures", f"{run.n_obs:,} / {len(myotubes):,} / {n_structures:,}"],
            ["Assigned nuclei", f"{assigned:,} / {run.n_obs:,}"],
            ["Pixel-overlap policy", config.get("measurement", {}).get("overlap_policy", "include")+" (within each object type/source; original morphology and assignments retained)"],
            ["Cohorts / selected acquisitions", ", ".join(sorted(images.cohort_id.unique()))+f" / {len(markers)}"],
            ["Physical pixel size", f"{pixel} um/px" if pixel else "Not supplied; physical-unit measurements omitted"],
            ["Workers / summed task time", f"{manifest['worker_count']} / {qc['field_summary'].get('duration_seconds', np.array([])).sum():.1f} s"],
            ["Reference segmentations", config["segmentations"]["nuclei"]["acquisition_id"]+"; "+config["segmentations"]["myotubes"]["acquisition_id"]]]
    registration = config.get("registration_qc", {})
    if registration.get("enabled"):
        rows.append(["DAPI registration / nucleus QC",
                     f"Per patch/round; {registration['metric']} change <= {registration['threshold']}; "
                     f"{registration['aggregation']} pooling; fast mode {registration['fast_mode']}. Transforms: uns['registration']."])
    page.table(rows, [190, WIDTH-2*MARGIN-190])
    page.text("Cell line and repeat coverage", 11, True)
    sample = run.uns["sample_metadata"]
    repeats = sorted(sample["repeat"].unique())
    coverage = [["Cell line"] + [f"Repeat {r}\nNuclei / myotubes" for r in repeats]]
    for cell in config["plate_layout_data"]["rows"].values():
        line = cell["cell_line"]
        cells = [line]
        for repeat in repeats:
            group = sample[(sample.cell_line == line) & (sample["repeat"] == repeat)]
            cells.append(f"{int(group.n_nuclei.sum()):,} / {int(group.n_myotubes.sum()):,}" if len(group) else "Not selected")
        coverage.append(cells)
    page.table(coverage, [100]+[(WIDTH-2*MARGIN-100)/max(1, len(repeats))]*len(repeats), 8, 3)
    problems = qc["input_validation"]
    counts = problems[problems.severity != "info"].status.value_counts() if len(problems) else {}
    issue_text = "; ".join(f"{key}: {value}" for key, value in dict(counts).items()) or "No input omissions detected."
    if appendix:
        issue_text = (f"{len(exclusions)} fields / {int(exclusions.excluded_patches.sum())} patches excluded. "
                      "Identities and reasons: appendix (page 4 onward), excluded_fields.csv, "
                      "and .uns['qc']['excluded_fields']. " + issue_text)
    page.text("Input/QC summary: "+escape(issue_text), 8)
    if manifest.get("visualisation_status", "disabled") not in {"disabled", "not_requested"}:
        page.text(f"Visual QC: {manifest.get('visualisation_views', 0)} views; {escape(manifest['visualisation_status'])}. "
                  f"{manifest.get('visualisation_selection', {}).get('selected_fields', 'Selected')} sampled fields. "
                  "Open qc_visualisations/index.html; companion audits/masks are optional.", 8)
    page.text("Cell line is derived from the physical well row. Repeat and plate come from the filename. Edge myotubes remain supplied fragments; cohorts are processed independently.", 8, color=GRAY)
    pdf.showPage()
    page = Page(pdf, 2, "Measurements and marker coverage", escape(config["dataset_id"]), total_pages)
    page.text("I = intensity; T = masked texture; S = individual structures and parent summaries. Existing morphology is measured once for each nucleus/myotube and each supplied structure.", 8)
    rows = [["Marker", "Round", "Compartments", "Features", "Image / structure coverage"]]
    status = qc["measurement_status"]
    for marker in markers:
        key = marker["key"]
        subset = status[status.marker == key] if "marker" in status else status
        measured = int((subset.status == "measured").sum()) if "status" in subset else 0
        missing = int((subset.status == "missing_image").sum()) if "status" in subset else 0
        structures_ok = int((subset.status == "structures_available").sum()) if "status" in subset else 0
        structures_missing = int((subset.status == "structures_unavailable").sum()) if "status" in subset else 0
        compartments = "N, M"
        if "cytoplasm" in marker["myotube_compartments"]: compartments += ", C"
        if "assigned_nuclei" in marker["myotube_compartments"]: compartments += ", U"
        if "nuclear_rim" in marker["nuclear_compartments"]: compartments += ", R/I"
        if "local_perinuclear_cytoplasm" in marker["nuclear_compartments"]: compartments += ", P"
        features = "I"+(" T" if marker.get("texture") else "")+(" S" if marker.get("structure_source") else "")
        coverage_text = f"{measured}/{measured+missing}"
        if marker.get("structure_source"): coverage_text += f"; S {structures_ok}/{structures_ok+structures_missing}"
        rows.append([key, marker["acquisition_id"].split("_")[1], compartments, features, coverage_text])
    page.table(rows, [100, 35, 140, 55, WIDTH-2*MARGIN-330], 7.5 if len(rows) > 29 else 8, 2 if len(rows) > 29 else 3)
    page.text("N: whole nucleus; M: whole myotube; C: myotube cytoplasm; U: union of resident nuclei inside the myotube; R/I: nuclear rim/interior; P: local perinuclear cytoplasm. Coverage counts available patches in completed field-cohorts.", 8)
    aberr = config["nuclear_aberration"]
    page.text("Nuclear aberration: "+("enabled; existing mean ensemble, sigmoid sums/means and nuclear-area-normalized myotube summaries." if aberr["enabled"] else "not enabled in this run."), 8)
    page.text("Intensity uses quantitative TIFFs. Texture uses acquisition-wide limits recorded in .var and the resolved marker configuration; it does not change the intensity measurements. Missing or too-small regions remain NaN.", 8, color=GRAY)
    pdf.showPage()
    page = Page(pdf, 3, "Reading the archive and interpreting measurements", escape(config["dataset_id"]), total_pages)
    page.text("All measurements are in myonis_4i_data.h5ad", 12, True)
    page.table([
        ["Location", "Contents"], [".obs / .X / .var", "Nuclear identity, metadata and existing aberration features / mean nuclear intensity / acquisition metadata"],
        ['.obsm["morphology_features"]', "Existing nuclear morphology; column names in .uns"],
        ['.obsm["intensity_features"]', "Named nuclear and compartment intensity measurements"],
        ['.obsm["texture_features"]', "Named masked texture measurements"],
        ['.obsm["structure_features"]', "Structure counts, sizes, intensity and occupancy per nucleus"],
        ['.uns["myotubes"]', "Complete myotube metadata, measurements and resident-nucleus summaries"],
        ['.uns["structures"]', "Individual structure tables, including unassigned objects"],
        ['.uns["relationships"]', "Parent IDs, overlap evidence and competing candidates"],
        ['.uns["feature_dictionary"]', "Definitions, units and storage locations for all measurements"],
        ['.uns["inputs"], ["qc"], ["provenance"]', "Source paths, coverage, failures and exact configuration"]], [220, WIDTH-2*MARGIN-220], 8, 3)
    code = ['import anndata as ad', 'run = ad.read_h5ad("myonis_4i_data.h5ad")',
            'nuclei = run.obs', 'myotubes = run.uns["myotubes"]',
            'sc35 = run.uns["structures"].get("sc35")',
            'counts = run.obsm["structure_features"].filter(like="__count")']
    for line in code:
        page.text('<font name="Courier">'+escape(line)+'</font>', 8, gap=3)
    page.y -= 10
    page.text("Structure -> nucleus -> myotube", 10, True)
    page.text("Every structure keeps its own ID and measurements. Nuclear summaries distinguish equal-object mean intensity from the mean across the union of structure pixels. Myotube ancestry is inherited through the assigned nucleus; direct cytoplasmic links are labelled separately.", 8)
    page.text("Zero means a valid measurement with no detected objects. NaN means missing or undefined. Existing aberration missing-value rules are preserved. Resident-nucleus aggregates include n_valid denominators. Sigmoid sums are score mass, not counts of abnormal nuclei.", 8)
    page.text("External nucleus QC is metadata only: .obs.qc_pass selects evaluated, persistent, non-edge nuclei. Myotube measurements still describe all reference nuclei. AnnData nuclear slicing does not filter linked .uns tables or recalculate their summaries; use explicit reaggregation after nuclear filtering.", 8, color=GRAY)
    for number, blocks in enumerate(appendix, start=4):
        pdf.showPage()
        page = Page(pdf, number, "Excluded fields and failure reasons", escape(config["dataset_id"]), total_pages)
        page.text("These fields contribute no measurements to this archive. Successful fields retain all "
                  "their computed measurements. Exclusions differ from nucleus-level persistence/edge flags.", 9)
        for block in blocks:
            height = block.wrap(WIDTH-2*MARGIN, 10000)[1]
            page.y -= height
            if page.y < 43:
                raise ValueError("Report exclusion appendix overflow")
            block.drawOn(pdf, MARGIN, page.y)
            page.y -= 14
    pdf.save()
    temporary.replace(output_path)
    return output_path
