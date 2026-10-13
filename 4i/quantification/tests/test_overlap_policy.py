import copy
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from conftest import box, polygon_line
from fouri_quantification.access import get_nuclei
from fouri_quantification.config import load_config
from fouri_quantification.features.aberration import exclusive_stats, add_exclusive_aberration_features
from fouri_quantification.inputs.manifests import build_index
from fouri_quantification.objects.compartments import build_compartments
from fouri_quantification.objects.overlap import measurement_masks, restrict_compartments
from fouri_quantification.pipeline import run_pipeline, prepare
from fouri_quantification.reporting.visualisation import read_qc_mask
from fouri_quantification.worker import process_field


def rectangle(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, bool)
    mask[y0:y1+1, x0:x1+1] = True
    return mask


def test_exclusion_removes_shared_pixels_from_every_peer_and_handles_empty():
    shape = (8, 9)
    a, b, c = [rectangle(shape, *coords) for coords in [(1, 1, 4, 5), (3, 1, 6, 5), (4, 3, 7, 6)]]
    occupancy = a.astype(int)+b+c
    masks, diagnostics = measurement_masks([a, b, c], shape, "exclude")
    for original, measured, row in zip([a, b, c], masks, diagnostics):
        np.testing.assert_array_equal(measured, original & (occupancy == 1))
        assert row["overlap_area_px2"] == (original & (occupancy > 1)).sum()
        assert row["measurement_area_px2"] == measured.sum()
    reverse, _ = measurement_masks([c, b, a], shape, "exclude")
    for actual, expected in zip(reverse, masks[::-1]):
        np.testing.assert_array_equal(actual, expected)
    identical, qc = measurement_masks([a, a.copy()], shape, "exclude")
    assert not any(m.any() for m in identical)
    assert all(r["measurement_empty"] and r["overlap_fraction"] == 1 for r in qc)
    assert measurement_masks([], shape, "exclude") == ([], [])
    retained, _ = measurement_masks([a, b], shape, "include")
    np.testing.assert_array_equal(retained[0], a)
    np.testing.assert_array_equal(retained[1], b)


def test_compartments_trim_without_new_rims_or_reclassifying_nuclei():
    shape = (24, 24)
    nuclei = [rectangle(shape, 3, 3, 13, 15), rectangle(shape, 10, 3, 20, 15)]
    myotubes = [rectangle(shape, 1, 1, 22, 22), rectangle(shape, 18, 1, 23, 22)]
    assignments = {0: 0, 1: 0}
    nc, mc = build_compartments(nuclei, myotubes, assignments, dict(rim_width_px=2, perinuclear_width_px=3), shape)
    before = copy.deepcopy(nc)
    nm, _ = measurement_masks(nuclei, shape, "exclude")
    mm, _ = measurement_masks(myotubes, shape, "exclude")
    restrict_compartments(nc, mc, nm, mm, assignments, shape)
    for i in range(2):
        for part in ["whole_nucleus", "nuclear_rim", "nuclear_interior"]:
            np.testing.assert_array_equal(nc[i][part], before[i][part] & nm[i])
        assert not (mc[i]["cytoplasm"] & (nuclei[0] | nuclei[1])).any()
        assert not (mc[i]["whole_myotube"] & (myotubes[0] & myotubes[1])).any()
    assert not (nc[0]["local_perinuclear_cytoplasm"] & nc[1]["local_perinuclear_cytoplasm"]).any()
    assert not (mc[0]["assigned_nuclei"] & (nuclei[0] & nuclei[1])).any()


@pytest.mark.parametrize("policy", ["split", None, False, 1, []])
def test_invalid_policy_is_rejected(dataset, policy):
    raw = yaml.safe_load(dataset.read_text())
    raw["measurement"] = dict(overlap_policy=policy)
    dataset.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="measurement.overlap_policy"):
        load_config(dataset)


def field_result(config, directory=None):
    index = build_index(config)
    field = sorted(index["images"].field_key.unique())[0]
    return process_field(config, index["images"][index["images"].field_key == field].to_dict("records"),
                         index["assets"][index["assets"].field_key == field].to_dict("records"), index["markers"],
                         visualisation_dir=directory)


def overlapping_dataset(dataset):
    config = load_config(dataset)
    for source, coords in [("nuclei", [box(2, 2, 14, 14), box(8, 2, 20, 14)]),
                           ("myotubes", [box(1, 1, 30, 16), box(24, 1, 31, 16)]),
                           ("sc35", [box(3, 3, 5, 5), box(9, 3, 11, 5), box(10, 3, 12, 5), box(18, 4, 20, 6)])]:
        spec = config["segmentations"].get(source) or config["segmentations"]["structures"][source]
        for path in Path(spec["directory"]).glob("*.txt"):
            path.write_text("\n".join(polygon_line(p) for p in coords)+"\n")
    # High shared signal makes accidental reuse visible in means and texture.
    for path in (Path(config["preprocessing"]["root"])/"quantitative_patches").glob("*/*.tif"):
        image = tifffile.imread(path)
        image[2:15, 8:15] = 200
        tifffile.imwrite(path, image)
    return config


def test_worker_exclusion_changes_pixels_not_geometry_counts_or_parent_ids(dataset):
    config = overlapping_dataset(dataset)
    included = field_result(config)
    config["measurement"]["overlap_policy"] = "exclude"
    excluded = field_result(config)
    for level in ["nuclei", "myotubes"]:
        before, after = included[level], excluded[level]
        columns = [c for c in before if c.startswith("geometry__") or c in
                   ["object_uid", "parent_myotube_uid", "area_px2", "n_nuclei", "nuclear_area_fraction", "assigned_nuclei_area_px2"]]
        pd.testing.assert_frame_equal(before[columns], after[columns])
    for name in included["relationships"]:
        pd.testing.assert_frame_equal(included["relationships"][name], excluded["relationships"][name])
    # Each 13x13 nucleus loses the shared 7x13 strip; these are manual expectations.
    nuclei = excluded["nuclei"]
    assert nuclei.overlap_area_px2.eq(91).all()
    assert nuclei.measurement_area_px2.eq(78).all()
    assert nuclei.sc35__whole_nucleus__n_pixels.eq(78).all()
    assert (included["nuclei"].sc35__whole_nucleus__mean > nuclei.sc35__whole_nucleus__mean).all()
    assert (included["nuclei"].sc35__whole_nucleus__entropy != nuclei.sc35__whole_nucleus__entropy).all()
    np.testing.assert_allclose(nuclei.nuclear_aberration_total_sigmoid, 78*96/255, rtol=1e-6)
    np.testing.assert_allclose(nuclei.nuclear_aberration_mean_sigmoid, 96/255, rtol=1e-6)
    resident = excluded["myotubes"][excluded["myotubes"].n_nuclei > 0]
    assert resident.measurement_area_px2.eq(368).all()
    assert resident.sc35__whole_myotube__n_pixels.eq(368).all()
    assert excluded["myotubes"][excluded["myotubes"].local_id == 1].measurement_area_px2.eq(16).all()
    np.testing.assert_allclose(resident.nuclear_aberration_assigned_nuclei_total_sigmoid, 156*96/255, rtol=1e-6)
    np.testing.assert_allclose(resident.nuclear_aberration_assigned_nuclei_total_per_nuclear_area, 96/255, rtol=1e-6)
    # The second and third structures each retain a 1x3 strip, but both lie in
    # ambiguous nuclear pixels, so their parent-clipped measurement is empty.
    structures = excluded["structures"]["sc35"]
    ambiguous = structures[structures.local_id.isin([1, 2])]
    assert ambiguous.sc35__object__n_pixels.eq(3).all()
    assert ambiguous.area_in_parent_px2.eq(0).all()
    assert ambiguous.intensity_mean_in_parent.isna().all()
    pd.testing.assert_series_equal(included["nuclei"].sc35__structures__count, nuclei.sc35__structures__count)
    assert nuclei[nuclei.local_id == 0].sc35__structures__occupied_area_px2.eq(9).all()


def test_no_overlap_modes_agree_and_include_is_default(dataset):
    config = load_config(dataset)
    assert config["measurement"]["overlap_policy"] == "include"
    included = field_result(config)
    config["measurement"]["overlap_policy"] = "exclude"
    excluded = field_result(config)
    for level in ["nuclei", "myotubes"]:
        pd.testing.assert_frame_equal(included[level], excluded[level][included[level].columns], check_exact=False, rtol=1e-6)
    for name, frame in included["structures"].items():
        pd.testing.assert_frame_equal(frame, excluded["structures"][name])


def test_fully_overlapping_nuclei_retained_with_empty_measurements(dataset):
    config = load_config(dataset)
    for path in Path(config["segmentations"]["nuclei"]["directory"]).glob("*.txt"):
        path.write_text((polygon_line(box(2, 2, 14, 14))+"\n")*2)
    config["measurement"]["overlap_policy"] = "exclude"
    result = field_result(config)
    nuclei = result["nuclei"]
    assert len(nuclei) == 4 and nuclei.measurement_empty.all()
    assert nuclei.sc35__whole_nucleus__n_pixels.eq(0).all()
    assert nuclei.sc35__whole_nucleus__mean.isna().all()
    assert nuclei.sc35__whole_nucleus__entropy.isna().all()
    assert nuclei.nuclear_aberration_total_sigmoid.eq(0).all()
    assert nuclei.nuclear_aberration_mean_sigmoid.isna().all()
    assert result["myotubes"].n_nuclei.sum() == 4
    assert result["myotubes"].nuclear_aberration_assigned_nuclei_total_per_nuclear_area.isna().all()


def test_sigmoid_exclusion_uses_native_grid_and_keeps_missing_semantics(monkeypatch):
    from fouri_quantification.features import aberration
    def obj(patch, coords):
        return dict(patch_idx=patch, local_polygon=np.array(coords, np.float32))
    nuclei = [obj(1, box(2, 2, 14, 14)), obj(1, box(8, 2, 20, 14)), obj(2, box(2, 2, 14, 14))]
    myotubes = [obj(1, box(0, 0, 30, 30)), obj(3, box(0, 0, 30, 30))]
    predictions = {1: np.full((16, 16), .5, dtype=np.float32)}
    stats = exclusive_stats(nuclei, predictions, 32)
    np.testing.assert_array_equal(stats["area"][:2], [21, 21])
    np.testing.assert_allclose(stats["total"][:2], [10.5, 10.5])
    assert np.isnan(stats["total"][2])
    monkeypatch.setattr(aberration, "_load_nuclear_aberration_predictions_by_patch", lambda *a: (predictions, {}))
    na, ma = SimpleNamespace(obs=pd.DataFrame(index=range(3)), uns={}), SimpleNamespace(obs=pd.DataFrame(index=range(2)), uns={})
    add_exclusive_aberration_features(na, ma, nuclei, myotubes, {0: 0, 1: 0, 2: 0}, [], 32, "test")
    assert np.isnan(ma.obs.loc[0, "nuclear_aberration_assigned_nuclei_total_sigmoid"])
    assert ma.obs.loc[1, "nuclear_aberration_assigned_nuclei_total_sigmoid"] == 0
    assert np.isnan(ma.obs.loc[1, "nuclear_aberration_myotube_total_sigmoid"])


def test_exclusion_h5ad_qc_masks_and_resume_policy(dataset, tmp_path):
    pytest.importorskip("myonis_vis")
    overlapping_dataset(dataset)
    raw = yaml.safe_load(dataset.read_text())
    raw["measurement"] = dict(overlap_policy="exclude")
    raw["selection"].update(max_fields=1, field_sampling="first")
    raw["visualisation"] = dict(enabled=True, markers=["sc35", "nuclear_aberration"], max_patches_per_field=1, dpi=70, export_masks=True)
    dataset.write_text(yaml.safe_dump(raw))
    directory, manifest = run_pipeline(dataset, tmp_path/"exclusive", workers=2)
    assert manifest["status"] == "complete" and manifest["visualisation_status"] == "complete", manifest
    assert manifest["pdf_status"] == "complete"
    run = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.uns["measurement"]["overlap_policy"] == "exclude"
    assert run.obs.measurement_area_px2.eq(78).all()
    np.testing.assert_allclose(run.X[:, list(run.var.marker).index("sc35")], get_nuclei(run).sc35__whole_nucleus__mean)
    for _, view in run.uns["qc"]["visualisations"].iterrows():
        audit = pd.read_csv(directory/view.audit, sep="\t")
        assert audit.status.eq("pass").all()
        assert audit.overlap_policy.eq("exclude").all()
        assert json.loads((directory/view.metadata).read_text())["overlap_policy"] == "exclude"
        compartment = "sigmoid" if view.marker == "nuclear_aberration" else "whole_nucleus"
        nuclear = audit[(audit.object_level == "nucleus") & (audit.compartment == compartment)]
        masks = [read_qc_mask(directory/view.masks, key) for key in nuclear.mask_key]
        assert len(masks) == 2 and not (masks[0] & masks[1]).any()
        assert all(mask.sum() == 78 for mask in masks)
        if view.marker == "sc35":
            cytoplasm = audit[audit.compartment == "cytoplasm"]
            for key in cytoplasm.mask_key:
                assert not read_qc_mask(directory/view.masks, key)[2:15, 8:15].any()
    raw["measurement"]["overlap_policy"] = "include"
    dataset.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Cannot resume"):
        prepare(dataset, directory, resume=True)
