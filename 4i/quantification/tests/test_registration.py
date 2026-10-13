"""Known shifts exercise registration, every channel, contours and saved audits."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy.ndimage import gaussian_filter, shift
import tifffile
import yaml

from conftest import canonical, box, polygon_line
from fouri_quantification.config import load_config
from fouri_quantification.inputs.manifests import build_index
from fouri_quantification.pipeline import prepare, execute_task, finalize, run_pipeline
from fouri_quantification.registration import translate_image


@pytest.fixture
def registered_dataset(tmp_path):
    root = tmp_path/"input"
    size = 96
    root.mkdir()
    for name in ["nuclei", "myotubes", "sc35", "pred"]:
        (root/name).mkdir()
    yy, xx = np.indices((size, size))
    fixed = gaussian_filter(np.random.default_rng(31).normal(0, 1, (size, size)), 1)*12 + 15
    for y, x in [(30, 30), (60, 62), (43, 82), (76, 17)]:
        fixed += 180*np.exp(-((xx-x)**2+(yy-y)**2)/50)
    fixed = np.clip(fixed, 1, 255).astype(np.uint8)
    nuclei = [box(21, 21, 39, 39), box(53, 51, 71, 69), box(0, 40, 5, 49)]
    tubes = [box(12, 12, 80, 80)]
    # The second punctum fails the 80% parent overlap unless its source-round
    # contour is actually translated back along with its marker image.
    puncta = [box(27, 27, 30, 30), box(36, 36, 39, 39), box(59, 58, 62, 61)]
    acquisitions = [(r, 1, "dapi", "dapi") for r in range(1, 9)] + [
        (1, 2, "488", "emyhc"), (1, 4, "647", "53bp1"),
        (3, 3, "555", "myhc"), (5, 3, "555", "sc35")]
    rows, expected = [], {}
    for field in [1, 2]:
        for patch in [1, 2]:
            for r, ch, fluor, marker in acquisitions:
                dy, dx = (0, 0) if r == 2 else ((r % 3)-1, (r % 4)-2)
                dy *= field
                dx *= (1 if patch == 1 else -1)
                expected[(field, patch, r)] = (-dy, -dx)
                stem = canonical(r, ch, fluor, marker, field, patch)
                stem = stem.replace(f"fieldrefx-{(patch-1)*32:04d}", f"fieldrefx-{(patch-1)*size:04d}")
                stem = stem.replace(f"refx-{(patch-1)*32:06d}", f"refx-{(patch-1)*size:06d}")
                acq = f"measurement_{r:02d}_"+("dapi" if marker == "dapi" else f"{fluor}_{marker}")
                folder = root/"quantitative_patches"/acq
                folder.mkdir(parents=True, exist_ok=True)
                if marker == "dapi":
                    image = shift(fixed, (dy, dx), order=0, mode="constant", cval=0)
                    png = root/"png_8bit"/acq
                    png.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(png/(stem+".png"))
                    image = image.astype(np.float32)*100
                    if r == 2:
                        (root/"nuclei"/(stem+".txt")).write_text("\n".join(polygon_line(p, size) for p in nuclei)+"\n")
                    if r == 1:
                        probability = np.zeros((size//2, size//2), dtype=np.uint8)
                        probability[10:20, 10:20] = 180
                        probability = shift(probability, (dy/2, dx/2), order=0, mode="constant")
                        Image.fromarray(probability).save(root/"pred"/(stem+".png"))
                else:
                    base = np.full((size, size), 1000, dtype=np.float32)
                    base[21:40, 21:40] = 50000
                    base[51:70, 53:72] = 25000
                    image = shift(base, (dy, dx), order=0, mode="constant", cval=0)
                    kind, polys = ("myotubes", tubes) if marker == "myhc" else ("sc35", puncta)
                    if marker in {"myhc", "sc35"}:
                        (root/kind/(stem+".txt")).write_text("\n".join(
                            polygon_line(np.asarray(p)+[dx, dy], size) for p in polys)+"\n")
                path = folder/(stem+".ome.tif")
                tifffile.imwrite(path, image)
                rows.append(dict(quantitative_path=str(path), acquisition_id=acq, patch_size=size))
    pd.DataFrame(rows).to_csv(root/"manifest.csv", index=False)
    panel = dict(include_dapi=True, markers={
        "emyhc": dict(acquisition_id="measurement_01_488_emyhc"),
        "53bp1": dict(acquisition_id="measurement_01_647_53bp1"),
        "myhc": dict(acquisition_id="measurement_03_555_myhc"),
        "sc35": dict(acquisition_id="measurement_05_555_sc35", structure_source="sc35")})
    (tmp_path/"markers.yaml").write_text(yaml.safe_dump(panel))
    settings = dict(registration=dict(downsample=1, upsample_factor=1, min_global_correlation=.25,
                                      patch_level_min_overlap_fraction=.8, patch_level_max_shift_px=10))
    (root/"registration.json").write_text(json.dumps(settings))
    raw = dict(schema_version=1, dataset_id="registration_fixture", preprocessing=dict(root=str(root), image_manifest="manifest.csv", display_manifest=None),
        marker_config="markers.yaml", output_root="results", selection=dict(cohorts=["m01_08"]),
        registration_qc=dict(enabled=True, registration_config="registration.json"),
        segmentations=dict(nuclei=dict(directory="nuclei", acquisition_id="measurement_02_dapi", segmentation_id="n1"),
                           myotubes=dict(directory="myotubes", acquisition_id="measurement_03_555_myhc", segmentation_id="m1"),
                           structures=dict(sc35=dict(directory="sc35", acquisition_id="measurement_05_555_sc35", segmentation_id="s1"))),
        nuclear_aberration=dict(enabled=True, acquisition_id="measurement_01_dapi", prediction_directories=["pred"]),
        execution=dict(workers=1), texture=dict(enabled=True, distances_px=[1], fallback_range=[0,65535]),
        outputs=dict(pdf=False, qc_overlays=0))
    path = tmp_path/"dataset.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, expected


def test_registered_channels_parent_ids_qc_and_resume(registered_dataset, tmp_path):
    path, expected = registered_dataset
    config = load_config(path)
    assert config["registration_qc"]["fast_mode"] is True
    directory, status = run_pipeline(path, tmp_path/"run")
    assert status["status"] == "complete", status
    run = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.n_obs == 12
    assert len(run.uns["myotubes"]) == 4
    assert run.obs.qc_status.value_counts().to_dict() == {"passed": 8, "not_evaluable": 4}
    assert run.obs.is_persistent.isna().sum() == 4
    assert run.uns["myotubes"].n_nuclei.sum() == 8
    transforms = run.uns["registration"]["transforms"]
    assert len(transforms) == 32
    for tr in transforms.to_dict("records"):
        # Resolve the full identity through its image row, not a local field label.
        source = run.uns["inputs"]["images"].loc[run.uns["inputs"]["images"].spatial_key == tr["spatial_key"]].iloc[0]
        assert (tr["shift_y"], tr["shift_x"]) == expected[(int(source.field_id), int(source.patch_index), int(tr["round"]))]
    means = run.obsm["intensity_features"]
    np.testing.assert_allclose(means["emyhc__whole_nucleus__mean"].to_numpy().reshape(4,3)[:,:2], [[50000,25000]]*4)
    np.testing.assert_allclose(means["53bp1__whole_nucleus__mean"], means["emyhc__whole_nucleus__mean"], equal_nan=True)
    structures = run.uns["structures"]["sc35"]
    assert structures.parent_nucleus_uid.ne("").all()
    assert structures.parent_myotube_uid.ne("").all()
    assert len(structures) == 12
    assert run.obsm["structure_features"]["sc35__structures__count"].tolist() == [2,1,0]*4
    np.testing.assert_allclose(structures.area_px2, 9)
    np.testing.assert_allclose(structures.centroid_local_x,
                               structures.source_centroid_local_x + structures.registration_shift_x)
    np.testing.assert_allclose(run.uns["myotubes"].area_px2, 68*68)
    assert len(run.uns["qc"]["nucleus_qc"]["rounds"]) == 12*8
    assert run.obs.nuclear_aberration_mean_sigmoid.notna().any()
    assert (directory/"registration_qc/myonis_4i/nucleus_qc.csv").is_file()
    assert not list((directory/"registration_qc").rglob("*.png"))
    assert not list((directory/"registration_qc").rglob("*.npz"))
    assert execute_task(directory, 0)["status"] == "resumed"
    # A missing quantification checkpoint reuses the already checked QC transforms.
    task = json.loads((directory/"tasks.json").read_text())[0]
    qc_summary = directory/"registration_qc"/task["task_id"]/"summary.json"
    before = qc_summary.stat().st_mtime_ns
    (directory/"checkpoints"/(task["task_id"]+".h5ad")).unlink()
    assert execute_task(directory, 0)["status"] == "complete"
    assert qc_summary.stat().st_mtime_ns == before
    selected = json.loads((directory/"tasks"/(task["task_id"]+".json")).read_text())
    png = Path(selected["registration_inputs"][0]["path"])
    png.write_bytes(png.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="Input changed after prepare"):
        execute_task(directory, 0)


def test_registration_missing_dapi_and_conflicting_qc(registered_dataset, tmp_path):
    path, _ = registered_dataset
    raw = yaml.safe_load(path.read_text())
    raw["nucleus_qc"] = dict(directory="external_qc")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="either"):
        load_config(path)
    raw.pop("nucleus_qc")
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    next((Path(config["preprocessing"]["root"])/"png_8bit/measurement_07_dapi").glob("*.png")).unlink()
    assert "missing_dapi_png" in set(build_index(config)["issues"].status)
    with pytest.raises(ValueError, match="Preflight"):
        prepare(path, tmp_path/"missing")


def test_registration_failure_never_quantifies_unregistered(registered_dataset, tmp_path):
    path, _ = registered_dataset
    raw = yaml.safe_load(path.read_text())
    raw["selection"]["max_fields"] = 1
    path.write_text(yaml.safe_dump(raw))
    for p in (Path(raw["preprocessing"]["root"])/"png_8bit/measurement_07_dapi").glob("*.png"):
        Image.fromarray(np.zeros((96,96), dtype=np.uint8)).save(p)
    run = prepare(path, tmp_path/"failed")
    with pytest.raises(ValueError, match="Registration QC failed"):
        execute_task(run, 0)
    assert not list((run/"checkpoints").glob("*.h5ad"))
    status = finalize(run)
    assert status["status"] == "failed"
    assert list((run/"registration_qc").rglob("nucleus_qc.csv"))


def test_translation_subpixel_sign_and_missing_borders():
    image = np.arange(80, dtype=np.float32).reshape(8,10)
    tr = dict(shift_y=1.5, shift_x=-.5, valid_y0=2, valid_y1=8, valid_x0=0, valid_x1=9)
    registered = translate_image(image, tr)
    assert registered[4, 3] == pytest.approx((image[2,3]+image[2,4]+image[3,3]+image[3,4])/4)
    assert np.isnan(registered[:2]).all() and np.isnan(registered[:,-1]).all()


def test_registered_visualisations_audit_actual_float_pixels(registered_dataset, tmp_path):
    path, _ = registered_dataset
    raw = yaml.safe_load(path.read_text())
    raw["selection"]["max_fields"] = 1
    raw["visualisation"] = dict(enabled=True, markers=["emyhc", "nuclear_aberration"], max_patches_per_field=1, export_masks=True)
    path.write_text(yaml.safe_dump(raw))
    directory, result = run_pipeline(path, tmp_path/"vis")
    assert result["status"] == "complete", result
    assert result["visualisation_failures"] == 0
    run = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    views = run.uns["qc"]["visualisations"]
    view = views.loc[views.marker == "emyhc"].iloc[0]
    with np.load(directory/view["masks"]) as archive:
        assert "source_values" in archive
    metadata = json.loads((directory/view["metadata"]).read_text())
    assert metadata["registration"]["round"] == 1


def test_fast_and_full_qc_have_identical_scores(registered_dataset, tmp_path):
    from fouri_quantification.registration import run_field
    path, _ = registered_dataset
    config = load_config(path)
    index = build_index(config)
    spatial = index["images"].iloc[0].spatial_key
    task = dict(fingerprint="same-inputs", field_key=index["images"].iloc[0].field_key,
                images=index["images"].loc[index["images"].spatial_key == spatial].to_dict("records"),
                assets=index["assets"].loc[index["assets"].spatial_key == spatial].to_dict("records"),
                registration_inputs=index["registration_inputs"].loc[index["registration_inputs"].spatial_key == spatial].to_dict("records"))
    fast_transforms, fast_qc = run_field(config, task, tmp_path/"fast")
    config["registration_qc"]["fast_mode"] = False
    full_transforms, full_qc = run_field(config, task, tmp_path/"full")
    pd.testing.assert_frame_equal(fast_transforms, full_transforms)
    pd.testing.assert_frame_equal(fast_qc["objects"], full_qc["objects"])
    pd.testing.assert_frame_equal(fast_qc["rounds"], full_qc["rounds"])
    assert not (tmp_path/"fast/plots").exists()
    assert list((tmp_path/"full/registration_maps").glob("*.npz"))
    assert list((tmp_path/"full/plots").glob("*.png"))


def test_late_cohort_and_parallel_field_workers(registered_dataset, tmp_path):
    path, _ = registered_dataset
    raw = yaml.safe_load(path.read_text())
    root = Path(raw["preprocessing"]["root"])
    mapping = {1:9, 2:10, 3:11, 5:12}

    def renamed(value):
        value = str(value).replace("cohort-m01-08", "cohort-m09-12").replace("__m01_08", "__m09_12")
        for old, new in mapping.items():
            value = value.replace(f"measurement_{old:02d}_", f"measurement_{new:02d}_")
            value = value.replace(f"measurement-{old:02d}__", f"measurement-{new:02d}__")
        return value

    rows = pd.read_csv(root/"manifest.csv")
    rows = rows[rows.acquisition_id.map(lambda v: int(v.split("_")[1]) in mapping)]
    rows["quantitative_path"] = rows.quantitative_path.map(renamed)
    rows["acquisition_id"] = rows.acquisition_id.map(renamed)
    for file in list(root.rglob("*.png"))+list(root.rglob("*.txt"))+list(root.rglob("*.tif")):
        destination = Path(renamed(file))
        if destination != file:
            destination.parent.mkdir(parents=True, exist_ok=True)
            file.rename(destination)
    rows.to_csv(root/"manifest.csv", index=False)
    panel_path = path.parent/"markers.yaml"
    panel = yaml.safe_load(panel_path.read_text())
    for marker in panel["markers"].values():
        marker["acquisition_id"] = renamed(marker["acquisition_id"])
    panel_path.write_text(yaml.safe_dump(panel))
    for source in [raw["segmentations"]["nuclei"], raw["segmentations"]["myotubes"], raw["segmentations"]["structures"]["sc35"]]:
        source["acquisition_id"] = renamed(source["acquisition_id"])
    raw["nuclear_aberration"]["acquisition_id"] = renamed(raw["nuclear_aberration"]["acquisition_id"])
    raw["selection"]["cohorts"] = ["m09_12"]
    raw["registration_qc"]["compare_metrics"] = True
    path.write_text(yaml.safe_dump(raw))
    folder, status = run_pipeline(path, tmp_path/"late", workers=2)
    assert status["status"] == "complete", status
    run = ad.read_h5ad(folder/"myonis_4i_data.h5ad")
    assert set(run.uns["registration"]["transforms"]["round"]) == {9,10,11,12}
    assert run.uns["registration"]["transforms"].reference_round.eq(10).all()
    assert set(run.uns["qc"]["nucleus_qc"]["rounds"]["round"]) == {9,10,11,12}
    assert run.n_obs == 12
    from fouri_quantification.qc import run_4i_nucleus_segmentation_qc as qc
    summary = json.loads((folder/"registration_qc/summary.json").read_text())
    assert summary["settings"]["rounds"] == [9,10,11,12]
    qc.write_statistics(folder/"registration_qc", summary)
    comparison = pd.read_csv(folder/"registration_qc/registration_metrics/field_comparison.csv")
    assert set(comparison.metric) == {"ssim", "relative_calibrated", "zncc_signal", "gradient"}
