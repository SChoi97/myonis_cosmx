import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from fouri_quantification.config import load_config
from fouri_quantification.diagnostics import sample_visualisation_fields
from fouri_quantification.features.intensity import intensity
from fouri_quantification.features.texture import texture
from fouri_quantification.inputs.manifests import build_index
from fouri_quantification.pipeline import run_pipeline, execute_task
from test_registration import registered_dataset


def test_reference_dapi_only_keeps_all_registration_inputs(registered_dataset):
    path, _ = registered_dataset
    index = build_index(load_config(path))
    assert [m["key"] for m in index["markers"] if m["acquisition_id"].endswith("_dapi")] == ["dapi_r02"]
    assert len(index["images"][index["images"].marker == "dapi"]) == 32
    assert set(index["registration_inputs"]["round"]) == set(range(1, 9))


def test_random_visualisation_limit_is_independent_reproducible_and_bounded(dataset):
    config = load_config(dataset)
    config["visualisation"].update(enabled=True, max_fields=100, field_seed=19)
    fields = [f"field_{n:05d}" for n in range(20000)]
    sample = sample_visualisation_fields(fields, config)
    assert len(sample) == len(set(sample)) == 100
    assert sample == sample_visualisation_fields(reversed(fields), config)
    config["selection"]["field_seed"] = 12
    assert sample == sample_visualisation_fields(fields, config)
    config["visualisation"]["field_seed"] = 20
    assert sample != sample_visualisation_fields(fields, config)
    config["visualisation"]["max_fields"] = 0
    assert sample_visualisation_fields(fields, config) == []


@pytest.mark.parametrize("value", [-1, True, "100", 1.5])
def test_invalid_visualisation_field_limit(dataset, value):
    data = yaml.safe_load(dataset.read_text())
    data["visualisation"] = dict(enabled=True, max_fields=value)
    dataset.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="visualisation.max_fields"):
        load_config(dataset)


def test_lean_sample_readable_names_and_timing_preserve_all_fields(dataset, tmp_path):
    pytest.importorskip("myonis_vis")
    raw = yaml.safe_load(dataset.read_text())
    raw["visualisation"] = dict(enabled=True, max_fields=1, field_seed=23,
                                max_patches_per_field=1, markers=["myhc"], dpi=70)
    dataset.write_text(yaml.safe_dump(raw))
    folder, result = run_pipeline(dataset, tmp_path/"lean", workers=2, troubleshoot_timing=True)
    assert result["status"] == "complete", result
    run = ad.read_h5ad(folder/"myonis_4i_data.h5ad")
    assert run.obs.field_id.nunique() == 2
    assert run.n_obs == 8
    views = run.uns["qc"]["visualisations"]
    assert len(views) == 1 and views.status.eq("complete").all()
    png = Path(views.iloc[0].png)
    assert "plate-sc_rep5-6__rep-05__cell-K32Del__well-r04c02" in png.name
    assert "__patch_" in png.name and "__marker-myhc.png" in png.name
    assert views.iloc[0].audit == views.iloc[0].metadata == views.iloc[0].masks == ""
    assert not list((folder/"qc_visualisations").rglob("*.npz"))
    assert not list((folder/"qc_visualisations").rglob("*.json"))
    selected = json.loads((folder/"inputs/visualisation_fields.json").read_text())
    assert len(selected["field_keys"]) == 1 and selected["eligible_fields"] == 2
    times = pd.read_csv(folder/"logs/timings.csv")
    assert {"index_inputs", "intensity_features", "texture_features", "measurement_visualisations", "write_h5ad"} <= set(times.stage)
    assert (times.seconds >= 0).all()
    summary = result["timing_summary"]
    assert summary["completed_fields"] == summary["timed_fields"] == 2
    assert summary["total"]["patches"] == 4
    assert summary["total"]["images"] == 16
    assert summary["groups"]["with_diagnostics"]["fields"] == 1
    assert summary["groups"]["without_diagnostics"]["fields"] == 1
    assert summary["wall"]["seconds_per_patch"] == summary["wall"]["elapsed_seconds"]/4
    assert json.loads((folder/"run_manifest.json").read_text())["timing_summary"] == summary
    assert execute_task(folder, 0)["status"] == "resumed"
    _, resumed = run_pipeline(dataset, folder, resume=True, workers=2, troubleshoot_timing=True)
    assert resumed["timing_summary"]["wall"]["execution_counts"]["resumed"] == 2
    assert resumed["timing_summary"]["wall"]["seconds_per_patch"] is None


def test_troubleshoot_samples_all_diagnostics_without_changing_measurements(registered_dataset, tmp_path):
    pytest.importorskip("myonis_vis")
    path, _ = registered_dataset
    raw = yaml.safe_load(path.read_text())
    # One patch per field keeps the integration check small; both fields quantify.
    manifest = Path(raw["preprocessing"]["root"])/"manifest.csv"
    rows = pd.read_csv(manifest)
    rows[rows.quantitative_path.str.contains("patch_0001")].to_csv(manifest, index=False)
    raw["visualisation"] = dict(enabled=False, max_fields=1, field_seed=7, dpi=60, max_labels_per_panel=3)
    path.write_text(yaml.safe_dump(raw))
    lean_dir, lean_status = run_pipeline(path, tmp_path/"plain", workers=2)
    debug_dir, debug_status = run_pipeline(path, tmp_path/"debug", workers=2, troubleshoot=True)
    assert lean_status["status"] == debug_status["status"] == "complete"
    lean = ad.read_h5ad(lean_dir/"myonis_4i_data.h5ad")
    debug = ad.read_h5ad(debug_dir/"myonis_4i_data.h5ad")
    np.testing.assert_allclose(lean.X, debug.X, equal_nan=True)
    for family in ["intensity_features", "texture_features", "morphology_features"]:
        if isinstance(lean.obsm[family], pd.DataFrame):
            pd.testing.assert_frame_equal(lean.obsm[family], debug.obsm[family])
        else:
            np.testing.assert_allclose(lean.obsm[family], debug.obsm[family], equal_nan=True)
    pd.testing.assert_frame_equal(lean.uns["myotubes"], debug.uns["myotubes"])
    views = debug.uns["qc"]["visualisations"]
    assert views.field_key.nunique() == 1 and len(views) == 6
    assert set(views.marker) == {"dapi_r02", "emyhc", "53bp1", "myhc", "sc35", "nuclear_aberration"}
    assert views[["audit", "masks", "metadata"]].ne("").all().all()
    assert views.mismatches.sum() == 0
    assert not (lean_dir/"logs/timings.csv").exists()
    assert (debug_dir/"logs/timings.csv").is_file()
    summaries = list((debug_dir/"registration_qc").glob("*/summary.json"))
    assert sorted(json.loads(p.read_text())["settings"]["fast_mode"] for p in summaries) == [False, True]
    assert len(list((debug_dir/"registration_qc").glob("*/registration_overlays/*.png"))) == 1


def test_cropped_compartments_preserve_full_mask_statistics():
    rng = np.random.default_rng(7)
    image = rng.normal(20, 3, (127, 191)).astype(np.float32)
    mask = np.zeros(image.shape, dtype=bool)
    mask[30:43, 50:90] = True
    mask[32:40, 60:71] = False
    image[32, 55] = np.nan
    settings = dict(levels=16, distances_px=[1, 3], min_pairs=16, min_pixels=16)
    crop = np.s_[30:43, 50:90]
    for function, args in [(intensity, ()), (texture, (settings, [0, 40]))]:
        full = function(image, mask, *args)
        cropped = function(image[crop], mask[crop], *args)
        assert full.keys() == cropped.keys()
        np.testing.assert_allclose(list(full.values()), list(cropped.values()), equal_nan=True)
