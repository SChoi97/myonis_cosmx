"""MYONIS-VisTools QC built from the actual masks and saved measurements.

The renderer is deliberately separate from quantification. Pixel fills use the
original boolean masks; MYONIS-VisTools supplies unsmoothed contour overlays.
"""
from __future__ import annotations

import hashlib
import html
from importlib import metadata
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import pandas as pd


def renderer_identity():
    try:
        import myonis_vis as mvt
    except ImportError as exc:
        raise ImportError("Visualisation or integrated registration QC requires MYONIS-VisTools. Install it in this environment with "
                          "python -m pip install -e /absolute/path/MYONIS-VisTools") from exc
    root = Path(mvt.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(package="myonis-vistools", version=mvt.__version__, source_sha256=digest.hexdigest(),
                dependencies={name: metadata.version(name) for name in ["matplotlib", "Pillow", "numpy", "scikit-image", "scipy"]})


def region(row, mask, level, compartment, marker, *, mean_key=None, sum_key=None,
           area_key=None, empty_sum_zero=False):
    prefix = f"{marker}__{compartment}__"
    label = {"nucleus": "N", "myotube": "M", "structure": "S"}[level]+str(row["local_id"])
    return dict(object_uid=row["object_uid"], label=label, object_level=level,
                parent_nucleus_uid=row.get("parent_nucleus_uid", ""),
                parent_myotube_uid=row.get("parent_myotube_uid", ""),
                compartment=compartment, mask=mask,
                mean_feature=mean_key or prefix+"mean", sum_feature=sum_key or prefix+"integrated",
                saved_mean=row.get(mean_key or prefix+"mean", np.nan),
                saved_integrated=row.get(sum_key or prefix+"integrated", np.nan),
                integrated_stored=(sum_key or prefix+"integrated") in row,
                saved_n_pixels=row.get(area_key or prefix+"n_pixels", np.nan),
                empty_sum_zero=empty_sum_zero)


def audit_regions(regions, image, shape, pack_masks=True):
    """Independent arithmetic check; never call the feature-extraction function."""
    rows, masks = [], {}
    for i, item in enumerate(regions):
        mask = np.asarray(item["mask"], dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError("QC mask and measurement image have different grids")
        pixels = np.asarray(image[mask], dtype=np.float64) if image is not None else np.full(int(mask.sum()), np.nan)
        valid = np.isfinite(pixels)
        total = float(pixels.sum()) if pixels.size and valid.all() else np.nan
        mean = total/pixels.size if pixels.size and valid.all() else np.nan
        if pixels.size == 0 and item["empty_sum_zero"] and image is not None:
            total = 0.0
        # Legacy sigmoid results are stored in float32; allow only roundoff.
        mean_ok = bool(np.isclose(mean, item["saved_mean"], rtol=2e-6, atol=1e-7, equal_nan=True))
        sum_ok = bool(not item["integrated_stored"] or np.isclose(total, item["saved_integrated"], rtol=2e-6, atol=1e-7, equal_nan=True))
        area_ok = bool(not np.isfinite(item["saved_n_pixels"]) or int(mask.sum()) == item["saved_n_pixels"])
        key = f"mask_{i:05d}"
        if pack_masks:
            masks[key] = np.packbits(mask.ravel(), bitorder="little")
        rows.append(dict({k: v for k, v in item.items() if k not in {"mask", "empty_sum_zero"}},
                         mask_key=key, n_pixels=int(mask.sum()), n_valid_pixels=int(valid.sum()),
                         recomputed_mean=mean, recomputed_integrated=total,
                         mean_matches=mean_ok, integrated_matches=sum_ok, area_matches=area_ok,
                         status="pass" if mean_ok and sum_ok and area_ok else "mismatch"))
    columns = ["object_uid", "label", "object_level", "parent_nucleus_uid", "parent_myotube_uid", "compartment",
               "mean_feature", "sum_feature", "saved_mean", "saved_integrated", "integrated_stored", "saved_n_pixels", "mask_key",
               "n_pixels", "n_valid_pixels", "recomputed_mean", "recomputed_integrated", "mean_matches",
               "integrated_matches", "area_matches", "status"]
    return pd.DataFrame(rows, columns=columns), masks


def read_qc_mask(archive, key):
    """Decode one exported ROI (no pickle loading)."""
    with np.load(archive, allow_pickle=False) as data:
        shape = tuple(data["shape"].astype(int))
        return np.unpackbits(data[key], bitorder="little", count=int(np.prod(shape))).reshape(shape).astype(bool)


def _safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]", lambda m: f"~{ord(m[0]):x}~", value)


def _style():
    return dict(smoothing="none", smooth_iterations=0, render_scale=1,
                line_width_mode="absolute", line_width=0.7, fill_alpha=0)


def _scene(image, nuclei, myotubes, structures, limits):
    import myonis_vis as mvt
    style = _style()
    return mvt.render_contours(image, [o["local_polygon"] for o in nuclei], color="#00e5ff",
                              cell_contours=[o["local_polygon"] for o in myotubes],
                              cell_options=dict(style, color="#ffe04b"),
                              markers={"structures": dict(style, contours=[o["local_polygon"] for o in structures], color="#ff63cf")},
                              intensity_range=limits, **style)


def _annotate(axis, objects, rows, prefix, limit):
    for obj, row in list(zip(objects, rows))[:limit]:
        label = prefix+str(row["local_id"])
        if prefix == "N": label += f">M{row['myotube_id']}" if row.get("myotube_id", -1) >= 0 else ">-"
        axis.text(obj["centroid_local_x"], obj["centroid_local_y"], label,
                  color="white", fontsize=5, bbox=dict(facecolor="black", alpha=.45, pad=.3, edgecolor="none"))


def _measurement_panel(axis, base, entries, limits, max_labels):
    import myonis_vis as mvt
    from matplotlib.colors import Normalize
    norm = Normalize(*limits, clip=True)
    cmap = mvt.make_colormap("viridis")
    rgb = base.astype(float)*.18
    occupancy = np.zeros(base.shape[:2], dtype=np.int32)
    occupied = []
    for item in entries:
        mask = item["mask"]
        if not mask.any(): continue
        occupied.append(item)
        val = item["saved_mean"]
        color = np.asarray(cmap(norm(val))[:3])*255 if np.isfinite(val) else np.array([255, 70, 130])
        rgb[mask] = color
        occupancy[mask] += 1
    rgb[occupancy > 1] = 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # Fill above is exactly pixel-wise. Outlines are display only, with no smoothing.
    if occupied:
        rgb = mvt.render_contours(rgb, np.stack([e["mask"] for e in occupied]), input_format="masks",
                                  color="#d0d0d0", **_style())
    axis.imshow(rgb, interpolation="nearest")
    for item in occupied[:max_labels]:
        yy, xx = np.nonzero(item["mask"])
        value = item["saved_mean"]
        label = f"{item['label']} {value:.3g}" if np.isfinite(value) else f"{item['label']} NA"
        axis.text(float(xx.mean()), float(yy.mean()), label, color="white", fontsize=5,
                  bbox=dict(facecolor="black", alpha=.55, pad=.3, edgecolor="none"))
    return cmap, norm


class FieldVisualiser:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.config = config["visualisation"]
        self.overlap_policy = config.get("measurement", {}).get("overlap_policy", "include")
        self.records = []

    def wants(self, marker, patch_position):
        limit = self.config["max_patches_per_field"]
        return ((limit is None or patch_position < limit) and
                (self.config["markers"] == "all" or marker in self.config["markers"]))

    def _path(self, path):
        # Relative to the run root, so the run folder remains portable.
        return str(Path(path).relative_to(self.folder.parent.parent))

    def failed(self, meta, marker, exc):
        record = dict(field_key=meta["field_key"], spatial_key=meta["spatial_key"], patch_index=meta["patch_index"],
                      marker=marker, kind="measurement", status="failed", detail=str(exc), png="", audit="", masks="", metadata="",
                      audit_rows=0, mismatches=0)
        self.records.append(record)

    def save_marker(self, marker, image, meta, nuclei, myotubes, structures, nrows, mrows, srows, regions,
                    structure_status="not applicable", note="", limits=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        import myonis_vis as mvt

        limits = tuple(marker["texture_range"] if limits is None else limits)
        key = marker["key"]
        patch_label = (f"patch_{int(meta['patch_index']):04d}__pr{int(meta['patch_row']):02d}-pc{int(meta['patch_column']):02d}"
                       f"__y{int(meta['field_reference_y'])}-x{int(meta['field_reference_x'])}")
        folder = self.folder/patch_label
        folder.mkdir(exist_ok=True)
        stem = folder/(self.folder.name+"__"+patch_label+"__marker-"+_safe_name(key))
        shape = image.shape if image is not None else (meta["patch_size"], meta["patch_size"])
        audit, packed = audit_regions(regions, image, shape, pack_masks=self.config["export_masks"])
        audit["marker"], audit["spatial_key"] = key, meta["spatial_key"]
        audit["overlap_policy"] = self.overlap_policy
        audit_path = stem.with_suffix(".tsv") if self.config["export_audits"] else None
        if audit_path:
            audit.to_csv(audit_path, sep="\t", index=False)
        masks_path = stem.with_suffix(".npz") if self.config["export_masks"] else None
        if masks_path is not None:
            if (key == "nuclear_aberration" or meta.get("registration")) and image is not None:
                packed["source_values"] = image
            np.savez_compressed(masks_path, shape=np.asarray(shape), **packed)
        mismatch_count = int((audit.status == "mismatch").sum())
        background = image if image is not None else np.zeros(shape, dtype=np.float32)
        base = mvt.prepare_image(background, intensity_range=limits)
        groups = {}
        for entry in regions:
            groups.setdefault((entry["object_level"], entry["compartment"]), []).append(entry)
        panels = 2+len(groups)
        if marker.get("structure_source") and not any(r["object_level"] == "structure" for r in regions):
            panels += 1
        nrows_fig = math.ceil(panels/3)
        height = 4.45*nrows_fig+1.9
        fig, axes = plt.subplots(nrows_fig, 3, figsize=(13.5, height), squeeze=False)
        axes = axes.ravel()
        try:
            axes[0].imshow(base, interpolation="nearest")
            axes[0].set_title(("Registered intensity" if meta.get("registration") else "Source intensity") if image is not None else "IMAGE UNAVAILABLE", fontsize=10)
            axes[1].imshow(_scene(background, nuclei, myotubes, structures, limits), interpolation="nearest")
            axes[1].set_title("Segmentations and IDs (N > resident M)", fontsize=10)
            label_limit = self.config["max_labels_per_panel"]
            _annotate(axes[1], nuclei, nrows, "N", label_limit)
            _annotate(axes[1], myotubes, mrows, "M", label_limit)
            _annotate(axes[1], structures, srows, "S", label_limit)
            nuclear_by_uid = {o["object_uid"]: o for o in nuclei}
            for obj, row in list(zip(structures, srows))[:label_limit]:
                parent = nuclear_by_uid.get(row.get("parent_nucleus_uid"))
                if parent:
                    axes[1].plot([obj["centroid_local_x"], parent["centroid_local_x"]],
                                 [obj["centroid_local_y"], parent["centroid_local_y"]], color="#ff63cf", lw=.4, ls=":")
            cmap = norm = None
            for axis, ((level, compartment), entries) in zip(axes[2:], groups.items()):
                cmap, norm = _measurement_panel(axis, base, entries, limits, label_limit)
                axis.set_title(f"{level}: {compartment}\nROI colour / label = saved mean", fontsize=9)
            if panels > 2+len(groups):
                axis = axes[2+len(groups)]
                axis.imshow(base, interpolation="nearest")
                axis.text(.5, .5, "Structures: "+structure_status, transform=axis.transAxes,
                          ha="center", va="center", color="white", bbox=dict(facecolor="black", alpha=.8))
            for axis in axes:
                axis.set_axis_off()
            title = (f"{meta['plate_name']} | {meta['cell_line']} | repeat {meta['repeat']} | {meta['well_id']} | "
                     f"{meta['cohort_id']} | field {meta['field_id']} | patch {meta['patch_index']}\n"
                     f"{key} | {marker['acquisition_id']} | overlap: {self.overlap_policy} | intensity checks: "
                     f"{len(audit)-mismatch_count}/{len(audit)} passed")
            fig.suptitle(title, fontsize=11, y=1-.16/height, va="top")
            fig.subplots_adjust(left=.02, right=.92, top=1-1.15/height, bottom=.75/height, wspace=.06, hspace=.20)
            if cmap is not None:
                bar_axis = fig.add_axes([.94, .20, .013, .56])
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis, label="Saved mean (native units)")
            footer = (f"Cyan = nucleus; yellow = myotube; magenta = structure; dotted lines = structure > nucleus. "
                      f"Structure input: {structure_status}.\nExact ROI fills; no smoothing. White = overlapping ROIs; pink = undefined mean. "
                      f"Display/colour limits {limits[0]:.4g} to {limits[1]:.4g}; labels capped at {label_limit}/panel.\n"
                      + ("IDs and values: TSV; packed masks decode with read_qc_mask(). " if audit_path else
                       "Full measurements and object IDs are stored in myonis_4i_data.h5ad. ")+note)
            fig.text(.02, .015, footer, fontsize=7, va="bottom")
            png = stem.with_suffix(".png")
            temporary = stem.with_suffix(".tmp.png")
            fig.savefig(temporary, dpi=self.config["dpi"], bbox_inches="tight", pad_inches=.12)
            temporary.replace(png)
        finally:
            plt.close(fig)
        metadata_path = stem.with_suffix(".json") if self.config["export_audits"] else None
        if metadata_path:
            metadata_path.write_text(json.dumps(dict(marker=key, acquisition_id=marker["acquisition_id"], spatial_key=meta["spatial_key"],
            source_path=meta.get("resolved_path", ""), shape=list(shape), display_range=limits, structure_status=structure_status,
            overlap_policy=self.overlap_policy, registration=meta.get("registration", {}),
            mask_encoding="numpy.packbits, bitorder=little, row-major; shape stored in NPZ", note=note), indent=2)+"\n")
        record = dict(field_key=meta["field_key"], spatial_key=meta["spatial_key"], patch_index=meta["patch_index"], marker=key,
                      kind="measurement", status="mismatch" if mismatch_count else ("complete" if image is not None else "image_unavailable"),
                      detail=structure_status, png=self._path(png), audit=self._path(audit_path) if audit_path else "",
                      masks=self._path(masks_path) if masks_path else "", metadata=self._path(metadata_path) if metadata_path else "",
                      source_path=meta.get("resolved_path", ""), field_label=self.folder.name,
                      audit_rows=len(audit), mismatches=mismatch_count)
        self.records.append(record)


def build_gallery(run_dir, records, selection):
    """A portable, offline index of completed/failed QC views; no embedded data."""
    root = Path(run_dir)
    folder = root/"qc_visualisations"
    folder.mkdir(exist_ok=True)
    rows = []
    for entry in records.to_dict("records"):
        links = []
        for column, label in [("png", "Figure"), ("audit", "Intensity audit"), ("masks", "Exact masks"), ("metadata", "Source")]:
            if entry.get(column):
                href = quote("../"+entry[column], safe="/")
                links.append(f'<a href="{href}">{label}</a>')
        cells = [entry.get("field_key", ""), entry.get("patch_index", ""), entry.get("marker", ""), entry.get("status", "")]
        rows.append("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in cells)+"<td>"+" | ".join(links)+"</td></tr>")
    text = ('<!doctype html><html><head><meta charset="utf-8"><title>MYONIS 4i QC</title>'
            '<style>body{font:15px system-ui;margin:36px;color:#183448}table{border-collapse:collapse;width:100%}'
            'td,th{padding:9px;border-bottom:1px solid #d8e2e7;text-align:left}a{color:#087e83}'
            'input{padding:10px;width:55%;margin:15px 0}tr.hide{display:none}</style></head><body>'
            '<h1>MYONIS 4i visual checks</h1><p>Figures use MYONIS-VisTools and exact measurement masks. '
            'Intensity audits compare saved values against pixels in the exported ROIs. Missing structures remain unavailable.</p>'
            f'<p>{html.escape(json.dumps(selection, sort_keys=True))}</p>'
            '<input id="search" placeholder="Filter by field, marker or status" aria-label="Filter visualisations">'
            '<table><thead><tr><th>Field / cohort</th><th>Patch</th><th>Marker</th><th>Status</th><th>Files</th></tr></thead><tbody>'
            +"".join(rows)+'</tbody></table><script>document.getElementById("search").oninput=function(){'
            'for(const r of document.querySelectorAll("tbody tr")){r.classList.toggle("hide",!r.textContent.toLowerCase().includes(this.value.toLowerCase()));}'
            '};</script></body></html>')
    (folder/"index.html").write_text(text)
    records.to_csv(folder/"index.tsv", sep="\t", index=False)
    return folder/"index.html"
