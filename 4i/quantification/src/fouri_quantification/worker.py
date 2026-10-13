"""One process owns one field/cohort. No concurrent shared HDF5 writers."""
from __future__ import annotations

import time
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial.distance import pdist, squareform

from .compatibility._aberration import add_nuclear_aberration_features, _get_polygon_rasterizer
from .compatibility._filtering import build_myotube_intensity_drop_map, filter_myotubes_by_intensity
from .features.intensity import intensity
from .features.morphology import legacy_morphology, additional_morphology
from .features.structures import structure_summary
from .features.texture import texture
from .objects.geometry import load_objects, metadata_row, object_mask, packed_geometry
from .objects.compartments import build_compartments, union_masks
from .objects.overlap import measurement_masks, restrict_compartments, iter_object_measurements
from .objects.relationships import assign
from .inputs.nucleus_qc import attach_nucleus_qc
from .registration import transform_lookup, source_transform, translate_image, translate_objects, registered_predictions
from .diagnostics import StageTimer


def _prefix(row, marker, compartment, values):
    row.update({f"{marker}__{compartment}__{k}": v for k, v in values.items()})


def _frame(rows, index="object_uid"):
    frame = pd.DataFrame(rows)
    if index not in frame:
        frame[index] = pd.Series(dtype=str)
    return frame.set_index(index, drop=False)


def process_field(config, images, assets, markers, overlay_path=None, visualisation_dir=None, nucleus_qc_rows=None,
                  registration_rows=None, timer=None):
    start = time.perf_counter()
    timer = timer or StageTimer()
    overlap_policy = config.get("measurement", {}).get("overlap_policy", "include")
    images = sorted(images, key=lambda r: (r["patch_index"], r["acquisition_id"]))
    by_image = {(r["spatial_key"], r["acquisition_id"]): r for r in images}
    by_asset = {(a["spatial_key"], a["source_key"]): a for a in assets}
    patches = {r["spatial_key"]: r for r in images}
    nuclei, myotubes, structures, registry, qc = [], [], {}, [], []
    field_key = images[0]["field_key"]
    visualiser = None
    if visualisation_dir is not None:
        from .reporting.visualisation import FieldVisualiser, region
        visualiser = FieldVisualiser(visualisation_dir, config)
    predictions_for_qc = {}
    transforms = transform_lookup(registration_rows or [])
    for typ in ["nuclei", "myotubes"] + ["structures."+s for s in config["segmentations"].get("structures", {})]:
        objects = []
        spec = config["segmentations"].get(typ) if typ in ["nuclei", "myotubes"] else config["segmentations"]["structures"][typ.split(".", 1)[1]]
        for spatial, representative in patches.items():
            image = by_image.get((spatial, spec.get("acquisition_id")))
            if image is None:
                continue
            asset = by_asset.get((spatial, typ))
            if typ in {"nuclei", "myotubes"} and (asset is None or asset["status"] != "present"):
                raise ValueError(f"Missing required {typ} contours for {spatial}")
            objects.extend(timer.call("load_contours", load_objects, asset, image, typ, config["geometry"]["edge_threshold"]))
        if typ == "nuclei":
            nuclei = objects
        elif typ == "myotubes":
            myotubes = objects
        else:
            structures[typ.split(".", 1)[1]] = objects
        registry.extend(objects)
    source_objects = {obj["object_uid"]: dict(obj) for obj in registry}
    source_geometry = packed_geometry(registry)
    source_morphology = {obj["object_uid"]: values for obj, values in zip(registry, timer.call("polygon_morphology", legacy_morphology, registry))}
    translate_objects(registry, transforms, config["geometry"]["edge_threshold"])
    if config["myhc_filter"]["enabled"]:
        filt = config["myhc_filter"]
        drop, _ = build_myotube_intensity_drop_map(filt["metadata_csv"], filt["intensity_threshold"])
        original = myotubes
        myotubes, _ = filter_myotubes_by_intensity(original, drop)
        kept = {o["object_uid"] for o in myotubes}
        for o in original:
            if o["object_uid"] not in kept:
                o.update(analysis_included=False, exclusion_reason="myhc_intensity_filter")
    attach_nucleus_qc(nuclei, nucleus_qc_rows, bool(config.get("nucleus_qc", {}).get("directory") or transforms))
    # Retain the existing post-filter local indexing as a compatibility column.
    for group, prefix in [(nuclei, "nuc"), (myotubes, "myo")]:
        local_counts = {}
        for i, obj in enumerate(group):
            p = obj["patch_idx"]
            obj["local_id"] = local_counts.get(p, 0)
            obj["object_id"] = f"{field_key}_{prefix}_{i}"
            local_counts[p] = obj["local_id"]+1
    assignments, n_links, candidates = timer.call("parent_assignment", assign, nuclei, myotubes, config["nucleus_myotube_assignment"]["overlap_threshold"])
    nrows = [dict(metadata_row(o), **source_morphology[o["object_uid"]]) for o in nuclei]
    mrows = [dict(metadata_row(o), **source_morphology[o["object_uid"]]) for o in myotubes]
    for i, row in enumerate(nrows):
        parent = assignments.get(i)
        row.update(parent_myotube_uid=myotubes[parent]["object_uid"] if parent is not None else "",
                   myotube_id=myotubes[parent]["local_id"] if parent is not None else -1,
                   myotube_patch_idx=myotubes[parent]["patch_idx"] if parent is not None else -1,
                   assignment_status=n_links[i]["status"], assignment_overlap=n_links[i]["overlap_fraction"],
                   assignment_competing_parent=n_links[i]["competing_parent"])
        row["Cell Line"] = row["cell_line"]
    for j, row in enumerate(mrows):
        residents = [i for i, p in assignments.items() if p == j]
        row["n_nuclei"] = len(residents)
        for status in ["passed", "failed", "not_evaluable", "not_assessed"]:
            row["n_nuclei_qc_"+status] = sum(nrows[i]["qc_status"] == status for i in residents)
        row["assigned_nuclei_area_px2"] = float(sum(nrows[i]["area_px2"] for i in residents))
        row["area_per_nucleus_px2"] = row["area_px2"]/len(residents) if residents else np.nan
        row["nuclei_density_per_1000px2"] = len(residents)/row["area_px2"]*1000 if row["area_px2"] else np.nan
        row["nuclear_nearest_neighbour_mean_px"] = np.nan
        if len(residents) > 1:
            coords = [[nrows[i]["centroid_field_x"], nrows[i]["centroid_field_y"]] for i in residents]
            distances = squareform(pdist(coords)); np.fill_diagonal(distances, np.inf)
            row["nuclear_nearest_neighbour_mean_px"] = float(distances.min(axis=1).mean())
        row["Cell Line"] = row["cell_line"]
    srows, s_links, sm_links, s_to_n, s_to_m = {}, [], [], {}, {}
    active_structure_markers = {m.get("structure_source"): m for m in markers if m.get("structure_source")}
    for source, marker in active_structure_markers.items():
        objects = structures.get(source, [])
        structure_rows = [dict(metadata_row(o), **source_morphology[o["object_uid"]]) for o in objects]
        policy = marker["structure_parent"]
        nmap, links, evidence = timer.call("parent_assignment", assign, objects, nuclei, marker["structure_overlap_threshold"], "structure_to_nucleus") if policy != "myotube" else ({}, [], [])
        candidates.extend(evidence)
        direct = {}
        if policy in {"nucleus_then_myotube", "myotube"}:
            remaining = [i for i in range(len(objects)) if i not in nmap]
            dmap, dlinks, evidence = timer.call("parent_assignment", assign, [objects[i] for i in remaining], myotubes, marker["structure_overlap_threshold"], "structure_to_myotube")
            candidates.extend(evidence)
            direct = {remaining[i]: j for i, j in dmap.items()}
        m_map = {}
        for i, row in enumerate(structure_rows):
            n = nmap.get(i)
            m = assignments.get(n) if n is not None else direct.get(i)
            if m is not None:
                m_map[i] = m
            row.update(marker=marker["key"], acquisition_id=marker["acquisition_id"],
                       parent_nucleus_uid=nuclei[n]["object_uid"] if n is not None else "",
                       parent_myotube_uid=myotubes[m]["object_uid"] if m is not None else "",
                       myotube_link_method="via_nucleus" if n is not None and m is not None else ("direct_cytoplasmic" if m is not None else "unassigned"))
            row.update(parent_nucleus_qc_status=nrows[n]["qc_status"] if n is not None else "no_parent_nucleus",
                       parent_nucleus_qc_pass=bool(n is not None and nrows[n]["qc_pass"]))
            if policy == "myotube":
                links.append(dict(child_uid=row["object_uid"], parent_uid="", relationship="structure_to_nucleus", status="not_applicable"))
            sm_links.append(dict(child_uid=row["object_uid"], parent_uid=row["parent_myotube_uid"],
                                 via_nucleus_uid=row["parent_nucleus_uid"], method=row["myotube_link_method"]))
        srows[source], s_to_n[source], s_to_m[source] = structure_rows, nmap, m_map
        s_links.extend(links)
    with timer.stage("aberration_features"):
        aberration_provenance = {}
        if config["nuclear_aberration"]["enabled"]:
            na = SimpleNamespace(obs=pd.DataFrame(index=range(len(nuclei))), uns={})
            ma = SimpleNamespace(obs=pd.DataFrame(index=range(len(myotubes))), uns={})
            sizes = {r["patch_size"] for r in images}
            if len(sizes) != 1:
                raise ValueError("Legacy aberration calculations require a consistent patch size per field/cohort")
            # An explicit asset map may point to renamed files. Give the unchanged
            # legacy loader canonical patch aliases, retaining actual paths in provenance.
            with tempfile.TemporaryDirectory(prefix="myonis-4i-predictions-") as scratch:
                paths_by_model = []
                for i in range(len(config["nuclear_aberration"]["prediction_directories"])):
                    folder = Path(scratch)/f"model_{i}"; folder.mkdir()
                    paths = []
                    for asset in assets:
                        if asset["source_key"] == f"aberration.{i}" and asset["status"] == "present":
                            alias = folder/f"patch_{asset['patch_index']:04d}.png"
                            alias.symlink_to(asset["resolved_path"])
                            paths.append(alias)
                    paths_by_model.append(paths)
                patch_size = sizes.pop()
                if transforms:
                    from .compatibility._aberration import _load_nuclear_aberration_predictions_by_patch
                    from .features.aberration import add_exclusive_aberration_features
                    predictions_for_qc, pred_provenance = _load_nuclear_aberration_predictions_by_patch(paths_by_model, patch_size, field_key)
                    registered_predictions(predictions_for_qc, images, config["nuclear_aberration"]["acquisition_id"], transforms)
                    pred_provenance = dict(pred_provenance, registration="DAPI source-to-reference translation scaled to native probability grid; linear float32, missing borders")
                    add_exclusive_aberration_features(na, ma, nuclei, myotubes, assignments, paths_by_model, patch_size, field_key,
                        predictions=predictions_for_qc, provenance=pred_provenance, policy=overlap_policy)
                elif overlap_policy == "exclude":
                    from .features.aberration import add_exclusive_aberration_features
                    add_exclusive_aberration_features(na, ma, nuclei, myotubes, assignments, paths_by_model, patch_size, field_key)
                else:
                    add_nuclear_aberration_features(na, ma, nuclei, myotubes, assignments, paths_by_model, patch_size, field_key)
                if visualiser and not transforms:
                    from .compatibility._aberration import _load_nuclear_aberration_predictions_by_patch
                    predictions_for_qc, _ = _load_nuclear_aberration_predictions_by_patch(paths_by_model, patch_size, field_key)
            for row, values in zip(nrows, na.obs.to_dict("records")): row.update(values)
            for row, values in zip(mrows, ma.obs.to_dict("records")): row.update(values)
            aberration_provenance = na.uns["nuclear_aberration_ensemble"]
    overlay_written = False
    for patch_position, (spatial, representative) in enumerate(patches.items()):
        size = representative["patch_size"]; shape = (size, size)
        ni = [i for i, o in enumerate(nuclei) if o["spatial_key"] == spatial]
        mi = [i for i, o in enumerate(myotubes) if o["spatial_key"] == spatial]
        nmasks = [object_mask(nuclei[i], shape) for i in ni]
        mmasks = [object_mask(myotubes[i], shape) for i in mi]
        nlocal, mlocal = {i: k for k, i in enumerate(ni)}, {j: k for k, j in enumerate(mi)}
        local_assignments = {nlocal[i]: mlocal[j] for i, j in assignments.items() if i in nlocal and j in mlocal}
        ncomp, mcomp = timer.call("compartments", build_compartments, nmasks, mmasks, local_assignments, config["geometry"], shape)
        for indices, masks, rows, is_myo in [(ni, nmasks, nrows, False), (mi, mmasks, mrows, True)]:
            for i, mask in zip(indices, masks):
                original_mask = object_mask(source_objects[rows[i]["object_uid"]], shape)
                _prefix(rows[i], "geometry", "additional", timer.call("raster_morphology", additional_morphology, original_mask, config["geometry"]["pixel_size_um"], is_myo))
        for j, mask, comps in zip(mi, mmasks, mcomp):
            mrows[j]["nuclear_area_fraction"] = comps["assigned_nuclei"].sum()/mask.sum() if mask.any() else np.nan
        nmeasure, ndiagnostics = measurement_masks(nmasks, shape, overlap_policy)
        mmeasure, mdiagnostics = measurement_masks(mmasks, shape, overlap_policy)
        for indices, rows, diagnostics in [(ni, nrows, ndiagnostics), (mi, mrows, mdiagnostics)]:
            for i, values in zip(indices, diagnostics):
                rows[i].update(values)
        if overlap_policy == "exclude":
            restrict_compartments(ncomp, mcomp, nmeasure, mmeasure, local_assignments, shape)
        # A compartment is identical across marker channels. Find its bounds once,
        # then measure only that crop; exported masks retain the exact full grid.
        roi_crops = {}
        with timer.stage("compartment_bounds"):
            for compartments in ncomp + mcomp:
                for mask in compartments.values():
                    yy, xx = np.nonzero(mask)
                    bounds = (slice(int(yy.min()), int(yy.max())+1), slice(int(xx.min()), int(xx.max())+1)) if yy.size else (slice(0,0), slice(0,0))
                    roi_crops[id(mask)] = (bounds, mask[bounds])
        for marker in markers:
            key, acq = marker["key"], marker["acquisition_id"]
            draw_qc = visualiser is not None and visualiser.wants(key, patch_position)
            qc_regions, qc_objects, qc_structure_rows = [], [], []
            structure_status = "not applicable"
            entry = by_image.get((spatial, acq))
            image, error = None, ""
            if entry is not None and entry["available"]:
                try:
                    image = np.asarray(timer.call("read_marker_images", tifffile.imread, entry["resolved_path"]), dtype=np.float32)
                    if image.shape != shape:
                        raise ValueError(f"Expected {shape}, got {image.shape}")
                    transform = source_transform(transforms, spatial, acq)
                    image = timer.call("shift_marker_images", translate_image, image, transform)
                    if transform is not None:
                        entry = dict(entry, registration=transform)
                except Exception as exc:
                    image, error = None, str(exc)
                    if not config["execution"]["allow_partial"]:
                        raise
            qc.append(dict(field_key=field_key, spatial_key=spatial, marker=key, acquisition_id=acq,
                           status="measured" if image is not None else "missing_image", detail=error,
                           n_nuclei=len(ni), n_myotubes=len(mi)))
            measurement_image = image if image is not None else np.full(shape, np.nan)
            for indices, comps, rows, compartments in [(ni, ncomp, nrows, marker["nuclear_compartments"]),
                                                       (mi, mcomp, mrows, marker["myotube_compartments"])]:
                for i, masks in zip(indices, comps):
                    for compartment in compartments:
                        mask = masks[compartment]
                        bounds, cropped_mask = roi_crops[id(mask)]
                        values = timer.call("intensity_features", intensity, measurement_image[bounds], cropped_mask)
                        _prefix(rows[i], key, compartment, values)
                        if draw_qc:
                            qc_regions.append(region(rows[i], mask, "nucleus" if rows is nrows else "myotube", compartment, key))
                        if marker.get("texture", False):
                            values = timer.call("texture_features", texture, measurement_image[bounds], cropped_mask, config["texture"], marker["texture_range"])
                            _prefix(rows[i], key, compartment, values)
            if "local_perinuclear_cytoplasm" in marker["nuclear_compartments"]:
                for i in ni:
                    a = nrows[i].get(f"{key}__whole_nucleus__mean", np.nan)
                    b = nrows[i].get(f"{key}__local_perinuclear_cytoplasm__mean", np.nan)
                    nrows[i][f"{key}__nuclear_cytoplasmic__ratio"] = a/b if np.isfinite(a) and np.isfinite(b) and b > marker.get("ratio_min_denominator", 1e-6) else np.nan
            source = marker.get("structure_source")
            if source:
                objects = structures.get(source, [])
                asset = by_asset.get((spatial, "structures."+source))
                available = bool(asset and asset["status"] == "present")
                structure_status = "available (empty)" if available else (asset["status"] if asset else "unavailable")
                qc.append(dict(field_key=field_key, spatial_key=spatial, marker=key, acquisition_id=acq,
                               status="structures_available" if available else "structures_unavailable", detail="", n_nuclei=len(ni), n_myotubes=len(mi)))
                summaries = {i: [] for i in ni}
                direct_items = {j: [] for j in mi}
                si_indices = [si for si, obj in enumerate(objects) if obj["spatial_key"] == spatial]
                regions = iter_object_measurements([objects[si] for si in si_indices], shape, overlap_policy)
                for si, (original_mask, mask, diagnostics) in zip(si_indices, regions):
                    obj = objects[si]
                    row = srows[source][si]
                    row.update(diagnostics)
                    _prefix(row, key, "object", timer.call("structure_intensity", intensity, measurement_image, mask))
                    if draw_qc:
                        qc_objects.append(obj); qc_structure_rows.append(row)
                        qc_regions.append(region(row, mask, "structure", "object", key))
                        structure_status = "available"
                    source_mask = object_mask(source_objects[obj["object_uid"]], shape)
                    _prefix(row, "geometry", "additional", timer.call("raster_morphology", additional_morphology, source_mask, config["geometry"]["pixel_size_um"]))
                    if marker.get("texture", False):
                        _prefix(row, key, "object", timer.call("structure_texture", texture, measurement_image, mask, config["texture"], marker["texture_range"]))
                    n, m = s_to_n[source].get(si), s_to_m[source].get(si)
                    if n in nlocal:
                        clipped = mask & nmeasure[nlocal[n]]
                        values = timer.call("structure_intensity", intensity, measurement_image, clipped)
                        row["intensity_mean_in_parent"] = values["mean"]
                        row["area_in_parent_px2"] = int(clipped.sum())
                        if draw_qc:
                            qc_regions.append(region(row, clipped, "structure", "inside_parent_nucleus", key,
                                mean_key="intensity_mean_in_parent", area_key="area_in_parent_px2"))
                        summaries[n].append(dict(pixels=np.flatnonzero(clipped), area_px2=row["area_px2"], mean_in_parent=values["mean"]))
                    elif m in mlocal:
                        clipped = mask & mcomp[mlocal[m]]["cytoplasm"]
                        values = timer.call("structure_intensity", intensity, measurement_image, clipped)
                        row["intensity_mean_in_parent"] = values["mean"]
                        row["area_in_parent_px2"] = int(clipped.sum())
                        if draw_qc:
                            qc_regions.append(region(row, clipped, "structure", "inside_parent_cytoplasm", key,
                                mean_key="intensity_mean_in_parent", area_key="area_in_parent_px2"))
                        direct_items[m].append(dict(pixels=np.flatnonzero(clipped), area_px2=row["area_px2"], mean_in_parent=values["mean"]))
                for i in ni:
                    _prefix(nrows[i], key, "structures", timer.call("structure_summaries", structure_summary, summaries[i], image, nmeasure[nlocal[i]], available))
                    if draw_qc and available:
                        mask = np.zeros(shape, dtype=bool)
                        for item in summaries[i]: mask.ravel()[item["pixels"]] = True
                        mask &= nmeasure[nlocal[i]]
                        qc_regions.append(region(nrows[i], mask, "nucleus", "structures", key,
                            mean_key=f"{key}__structures__mean_structure_pixel_intensity",
                            sum_key=f"{key}__structures__integrated_structure_intensity",
                            area_key=f"{key}__structures__occupied_area_px2", empty_sum_zero=True))
                for j in mi:
                    resident_indices = [i for i in ni if assignments.get(i) == j]
                    resident_mask = union_masks([nmeasure[nlocal[i]] for i in resident_indices], shape)
                    items = [item for i in resident_indices for item in summaries[i]]
                    if overlap_policy == "exclude":
                        # Retain the existing resident-union footprint, removing only
                        # the parent's shared pixels (not its outlying nuclear pixels).
                        resident_mask &= ~(mmasks[mlocal[j]] & ~mmeasure[mlocal[j]])
                        items = [dict(item, pixels=item["pixels"][resident_mask.ravel()[item["pixels"]]]) for item in items]
                        for item in items:
                            item["mean_in_parent"] = (float(np.asarray(image.ravel()[item["pixels"]], dtype=np.float64).mean())
                                                      if image is not None and len(item["pixels"]) else np.nan)
                    _prefix(mrows[j], key, "nuclear_structures", timer.call("structure_summaries", structure_summary, items, image, resident_mask, available))
                    _prefix(mrows[j], key, "cytoplasmic_structures", timer.call("structure_summaries", structure_summary, direct_items[j], image, mcomp[mlocal[j]]["cytoplasm"], available))
                    mrows[j][f"{key}__structures__count_total"] = len(items)+len(direct_items[j]) if available else np.nan
                    mrows[j][f"{key}__structures__count_per_nucleus"] = len(items)/len(resident_indices) if available and resident_indices else np.nan
                    mrows[j][f"{key}__structures__fraction_nuclei_positive"] = float(np.mean([bool(summaries[i]) for i in resident_indices])) if available and resident_indices else np.nan
                    mrows[j][f"{key}__structures__n_assessed_nuclei"] = len(resident_indices) if available else 0
                    if draw_qc and available:
                        for compartment, members, parent_mask in [("nuclear_structures", items, resident_mask),
                                ("cytoplasmic_structures", direct_items[j], mcomp[mlocal[j]]["cytoplasm"])]:
                            mask = np.zeros(shape, dtype=bool)
                            for item in members: mask.ravel()[item["pixels"]] = True
                            mask &= parent_mask
                            qc_regions.append(region(mrows[j], mask, "myotube", compartment, key,
                                mean_key=f"{key}__{compartment}__mean_structure_pixel_intensity",
                                sum_key=f"{key}__{compartment}__integrated_structure_intensity",
                                area_key=f"{key}__{compartment}__occupied_area_px2", empty_sum_zero=True))
            if draw_qc:
                try:
                    timer.call("measurement_visualisations", visualiser.save_marker, marker, image, entry or representative, [nuclei[i] for i in ni],
                        [myotubes[j] for j in mi], qc_objects, [nrows[i] for i in ni], [mrows[j] for j in mi],
                        qc_structure_rows, qc_regions, structure_status)
                except Exception as exc:
                    visualiser.failed(representative, key, exc)
            if overlay_path and not overlay_written and image is not None and acq == config["segmentations"]["nuclei"]["acquisition_id"]:
                from .reporting.plots import save_overlay
                save_overlay(image, [nuclei[i] for i in ni], [myotubes[j] for j in mi], [nrows[i] for i in ni], overlay_path)
                overlay_written = True
        if visualiser is not None and config["nuclear_aberration"]["enabled"] and visualiser.wants("nuclear_aberration", patch_position):
            from .compatibility._aberration import _scale_local_polygon_to_prediction
            from .features.aberration import prediction_masks
            prediction = predictions_for_qc.get(representative["patch_index"])
            grid = prediction.shape if prediction is not None else shape
            probability_regions, scaled_nuclei, scaled_myotubes = [], [], []
            for indices, objs, rows, level, mean_col, sum_col, scaled in [
                    (ni, nuclei, nrows, "nucleus", "nuclear_aberration_mean_sigmoid", "nuclear_aberration_total_sigmoid", scaled_nuclei),
                    (mi, myotubes, mrows, "myotube", "nuclear_aberration_myotube_mean_sigmoid", "nuclear_aberration_myotube_total_sigmoid", scaled_myotubes)]:
                masks = prediction_masks([objs[i] for i in indices], grid, size, overlap_policy)
                for i, mask in zip(indices, masks):
                    polygon = _scale_local_polygon_to_prediction(objs[i]["local_polygon"], grid, size)
                    probability_regions.append(region(rows[i], mask, level, "sigmoid", "nuclear_aberration",
                        mean_key=mean_col, sum_key=sum_col, area_key="nuclear_aberration_measurement_area_px2", empty_sum_zero=True))
                    scaled.append(dict(objs[i], local_polygon=polygon,
                        centroid_local_x=objs[i]["centroid_local_x"]*grid[1]/size,
                        centroid_local_y=objs[i]["centroid_local_y"]*grid[0]/size))
            try:
                probability_meta = dict(representative, resolved_path="")
                prediction_paths = [a["resolved_path"] for a in assets if a["spatial_key"] == spatial and a["source_key"].startswith("aberration.") and a["status"] == "present"]
                probability_meta["resolved_path"] = ";".join(prediction_paths)
                timer.call("prediction_visualisations", visualiser.save_marker, dict(key="nuclear_aberration", acquisition_id=config["nuclear_aberration"]["acquisition_id"], texture_range=[0, 1]),
                    prediction, probability_meta, scaled_nuclei, scaled_myotubes, [], [nrows[i] for i in ni], [mrows[j] for j in mi], [],
                    probability_regions, note="Mean sigmoid ensemble on its native prediction grid; score sums are not abnormal-nucleus counts.")
            except Exception as exc:
                visualiser.failed(representative, "nuclear_aberration", exc)
    with timer.stage("resident_nuclei_aggregation"):
        # Aggregate nucleus-level numerical features with valid-count denominators.
        feature_columns = sorted({k for row in nrows for k in row if "__" in k and not k.startswith("geometry__")})
        for j, row in enumerate(mrows):
            residents = [i for i, p in assignments.items() if p == j]
            for col in feature_columns:
                vals = np.asarray([nrows[i].get(col, np.nan) for i in residents], dtype=float)
                finite = vals[np.isfinite(vals)]
                row[f"resident_nuclei__{col}__n_valid"] = len(finite)
                row[f"resident_nuclei__{col}__mean"] = float(finite.mean()) if len(finite) else np.nan
                row[f"resident_nuclei__{col}__median"] = float(np.median(finite)) if len(finite) else np.nan
                row[f"resident_nuclei__{col}__std"] = float(finite.std()) if len(finite) else np.nan
    return dict(nuclei=_frame(nrows), myotubes=_frame(mrows), structures={k: _frame(v) for k, v in srows.items()},
                registry=_frame([metadata_row(o) for o in registry]), geometry=packed_geometry(registry), source_geometry=source_geometry,
                relationships=dict(nuclei_to_myotubes=pd.DataFrame(n_links), structures_to_nuclei=pd.DataFrame(s_links),
                                   structures_to_myotubes=pd.DataFrame(sm_links), candidates=pd.DataFrame(candidates)),
                qc=pd.DataFrame(qc), visualisations=pd.DataFrame(visualiser.records if visualiser else []),
                field_key=field_key, duration_seconds=time.perf_counter()-start,
                aberration_provenance=aberration_provenance, rasterizer=_get_polygon_rasterizer()[0])
