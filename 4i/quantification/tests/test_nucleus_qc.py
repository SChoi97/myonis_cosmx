import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from conftest import canonical, polygon_line, box
from fouri_quantification.config import load_config
from fouri_quantification.inputs.manifests import build_index, file_hash
from fouri_quantification.pipeline import run_pipeline, prepare, execute_task

SCRIPT_DIR = Path(__file__).resolve().parents[3]/"downstream_analysis/4i_analysis"
spec = importlib.util.spec_from_file_location("qc_exchange_tests", SCRIPT_DIR/"myonis_4i_qc_export.py")
exchange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exchange)


def registration_record(path, statuses=("kept", "filtered_change"), edges=(False, False)):
    source = exchange.source_snapshot(path.with_suffix(".png"), path)
    n = len(source["source_line_numbers"])
    scores = [None if s == "not_evaluable" else (.1 if s == "kept" else .8) for s in statuses]
    return dict(patch_key=path.stem, filter_mode="registration_based", score_metric="ssim", score_aggregation="max",
                reference_round=5, reference_count=n, shape=[32, 32], reference_source=source,
                change_threshold=.35, threshold_rule="<=", reference_area_pixels=[100]*n,
                is_edge=list(edges), statuses=list(statuses), change_score=scores, mean_change_score=scores,
                max_change_score=scores, valid_area_fraction=[1]*n,
                kept_reference_ids=[i+1 for i, status in enumerate(statuses) if status == "kept"],
                rounds={str(r): dict(mean_change_score=scores if r != 5 else [0]*n) for r in range(1, 9)})


def write_exchange(config, folder, only_first=False):
    folder.mkdir()
    (folder/"assignments").mkdir()
    paths = sorted(Path(config["segmentations"]["nuclei"]["directory"]).glob("*.txt"))
    if only_first:
        paths = paths[:1]
    completed = []
    for i, path in enumerate(paths):
        record = registration_record(path, statuses=("kept", "not_evaluable" if i == 0 else "filtered_change"), edges=(False, i == 0))
        assignment = f"assignments/{i}.json"
        (folder/assignment).write_text(json.dumps(record))
        completed.append(dict(patch_key=record["patch_key"], assignments=assignment))
    summary = dict(status="complete", settings=dict(reference_round=5), completed=completed)
    (folder/"summary.json").write_text(json.dumps(summary))
    return exchange.export_myonis_4i(folder, summary)


def configure(dataset, folder, require_all=False):
    raw = yaml.safe_load(dataset.read_text())
    raw["nucleus_qc"] = dict(directory=str(folder), require_all=require_all)
    dataset.write_text(yaml.safe_dump(raw))


def update_table(folder, name, function):
    frame = pd.read_csv(folder/name, dtype=str, keep_default_na=False)
    frame = function(frame)
    frame.to_csv(folder/name, index=False)
    path = folder/"manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = file_hash(folder/name)
    path.write_text(json.dumps(manifest))


def test_export_preserves_file_lines_and_nullable_flags(dataset, tmp_path):
    path = sorted(Path(load_config(dataset)["segmentations"]["nuclei"]["directory"]).glob("*.txt"))[0]
    path.write_text("\n"+path.read_text().replace("\n", "\n\n"))
    record = registration_record(path, ("kept", "not_evaluable"), (False, True))
    rows = list(exchange.object_rows(record, record["reference_source"]))
    assert [row[0]["reference_source_line_number"] for row in rows] == [2, 4]
    assert rows[0][0]["is_persistent"] == 1 and rows[0][0]["qc_pass"] == 1
    assert rows[1][0]["is_persistent"] is None and rows[1][0]["is_edge"] == 1
    assert rows[1][0]["qc_status"] == "not_evaluable"
    assert len(rows[0][1]) == 8


def test_binary_and_hungarian_export_keep_distinct_semantics(dataset):
    path = sorted(Path(load_config(dataset)["segmentations"]["nuclei"]["directory"]).glob("*.txt"))[0]
    source = exchange.source_snapshot(path.with_suffix(".png"), path)
    binary = dict(patch_key="patch", reference_round=5, reference_count=2, filter_mode="binary_mask_reference",
                  score_metric="reference_coverage", min_other_rounds=2, iou_threshold=.8, is_edge=[False, True],
                  reference_area_pixels=[10, 10], statuses=["kept", "not_evaluable"], kept_reference_ids=[1],
                  rounds={str(r): dict(reference_coverage=[1 if r in {1, 2, 5} else 0, 1]) for r in range(1, 9)})
    rows = list(exchange.object_rows(binary, source))
    assert rows[0][0]["is_persistent"] == 1 and rows[0][0]["qc_n_rounds_passing"] == 2
    assert rows[0][0]["qc_required_other_rounds"] == 2
    hungarian = dict(patch_key="patch", reference_round=5, reference_count=2, iou_threshold=.8,
                     rounds={str(r): dict(reference_ids=[1, 2 if r != 8 else 0], ious=[1, 1],
                                          is_edge=[r == 3, False]) for r in range(1, 9)})
    rows = list(exchange.object_rows(hungarian, source))
    assert rows[0][0]["is_edge"] == 1 and rows[0][0]["is_persistent"] is None
    assert rows[1][0]["is_persistent"] == 0 and rows[1][0]["qc_status"] == "failed"


def test_failed_and_legacy_exports_never_fabricate_negative_persistence(dataset, tmp_path):
    config = load_config(dataset)
    folder = write_exchange(config, tmp_path/"qc", only_first=True)
    summary_path = folder.parent/"summary.json"
    summary = json.loads(summary_path.read_text())
    entry = summary["completed"][0]
    record = json.loads((folder.parent/entry["assignments"]).read_text())
    summary.update(completed=[], failed=[dict(patch_key=entry["patch_key"], error="registration failed")],
                   reference_sources={entry["patch_key"]: record["reference_source"]})
    exchange.export_myonis_4i(folder.parent, summary)
    flags = pd.read_csv(folder/"nucleus_qc.csv")
    assert flags.qc_status.eq("not_assessed").all() and flags.is_persistent.isna().all()
    summary["completed"], summary["failed"] = [entry], []
    del record["reference_source"]
    (folder.parent/entry["assignments"]).write_text(json.dumps(record))
    exchange.export_myonis_4i(folder.parent, summary)
    assert pd.read_csv(folder/"nucleus_qc.csv").empty
    assert pd.read_csv(folder/"patch_qc.csv").qc_patch_status.eq("legacy_identity_missing").all()


def test_qc_h5ad_preserves_all_measurements_ids_and_ancestry(dataset, tmp_path):
    _, before_status = run_pipeline(dataset, tmp_path/"baseline")
    assert before_status["status"] == "complete"
    folder = write_exchange(load_config(dataset), tmp_path/"qc")
    configure(dataset, folder, require_all=True)
    run_dir, result = run_pipeline(dataset, tmp_path/"with_qc", workers=2)
    assert result["status"] == "complete" and result["pdf_status"] == "complete", result
    run, before = [ad.read_h5ad(tmp_path/name/"myonis_4i_data.h5ad") for name in ["with_qc", "baseline"]]
    assert run.n_obs == 8 and run.obs.qc_pass.sum() == 4
    assert run.obs.qc_status.value_counts().to_dict() == dict(passed=4, failed=3, not_evaluable=1)
    assert run.obs.is_persistent.isna().sum() == 1
    assert run.obs.is_edge.sum() == 1 and run.obs.is_edge_geometry.sum() == 0
    np.testing.assert_array_equal(run.obs_names, before.obs_names)
    np.testing.assert_allclose(run.X, before.X, equal_nan=True)
    for key in run.obsm:
        np.testing.assert_allclose(run.obsm[key], before.obsm[key], equal_nan=True)
    pd.testing.assert_series_equal(run.obs.parent_myotube_uid, before.obs.parent_myotube_uid)
    myotubes = run.uns["myotubes"]
    assert myotubes.n_nuclei.sum() == 8 and myotubes.n_nuclei_qc_passed.sum() == 4
    columns = [col for col in myotubes if not col.startswith("n_nuclei_qc_")]
    pd.testing.assert_frame_equal(myotubes[columns], before.uns["myotubes"][columns])
    structures = run.uns["structures"]["sc35"]
    assigned = structures[structures.parent_nucleus_uid != ""]
    assert assigned.parent_nucleus_qc_pass.equals(assigned.parent_nucleus_uid.map(run.obs.qc_pass).rename("parent_nucleus_qc_pass"))
    assert len(run.uns["qc"]["nucleus_qc"]["rounds"]) == 64
    assert run.uns["object_registry"].loc[run.obs_names, "qc_status"].tolist() == run.obs.qc_status.tolist()
    # All QC files are pinned into tasks, even when their flags are read only once.
    with (folder/"nucleus_qc.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Input changed"):
        execute_task(run_dir, 0)


def test_partial_qc_and_required_coverage(dataset, tmp_path):
    folder = write_exchange(load_config(dataset), tmp_path/"qc", only_first=True)
    configure(dataset, folder)
    index = build_index(load_config(dataset))
    assert len(index["nucleus_qc"]["objects"]) == 2
    assert "qc_not_assessed" in set(index["issues"].status)
    configure(dataset, folder, require_all=True)
    with pytest.raises(ValueError, match="require_all"):
        build_index(load_config(dataset))


@pytest.mark.parametrize("fault", ["reference", "contours", "duplicate", "missing_line", "bad_flag", "rounds", "checksum", "shape"])
def test_identity_and_exchange_validation(dataset, tmp_path, fault):
    folder = write_exchange(load_config(dataset), tmp_path/"qc")
    configure(dataset, folder)
    if fault == "reference":
        raw = yaml.safe_load(dataset.read_text())
        raw["segmentations"]["nuclei"]["acquisition_id"] = "measurement_02_dapi"
        # Direct loader invocation reaches its reference check independently of acquisition preflight.
        from fouri_quantification.inputs.nucleus_qc import load_nucleus_qc
        with pytest.raises(ValueError, match="reference round"):
            load_nucleus_qc({**load_config(dataset), "segmentations": raw["segmentations"]}, None, None, file_hash)
        return
    if fault == "contours":
        path = sorted(Path(load_config(dataset)["segmentations"]["nuclei"]["directory"]).glob("*.txt"))[0]
        path.write_text("\n".join(reversed(path.read_text().splitlines()))+"\n")
    elif fault == "duplicate":
        update_table(folder, "nucleus_qc.csv", lambda df: pd.concat([df, df.iloc[:1]]))
    elif fault == "missing_line":
        update_table(folder, "nucleus_qc.csv", lambda df: df.iloc[1:])
    elif fault == "rounds":
        update_table(folder, "round_qc.csv", lambda df: df.iloc[1:])
    elif fault == "checksum":
        with (folder/"nucleus_qc.csv").open("a") as stream:
            stream.write("\n")
    else:
        def change(df):
            df.loc[0, "is_persistent" if fault == "bad_flag" else "height"] = "0" if fault == "bad_flag" else "16"
            return df
        update_table(folder, "nucleus_qc.csv" if fault == "bad_flag" else "patch_qc.csv", change)
    with pytest.raises(ValueError):
        build_index(load_config(dataset))


def test_real_qc_cli_binary_export_and_stdlib_summary_rebuild(dataset, tmp_path):
    pytest.importorskip("myonis_vis")
    config = load_config(dataset)
    root = Path(config["preprocessing"]["root"])
    for r in range(1, 9):
        stem = canonical(r, 1, "dapi", "dapi")
        image_dir = root/f"png_8bit/measurement_{r:02d}_dapi"
        contour_dir = root/f"segmentation/multiround_nuclei/dapi_r{r}/4i_phenix_myonis_segmentation_nuclei_r{r}/02_nuclei_deduplication/contours"
        image_dir.mkdir(parents=True)
        contour_dir.mkdir(parents=True)
        Image.fromarray(np.full((32, 32), 100, np.uint8)).save(image_dir/(stem+".png"))
        contours = [box(2, 2, 14, 14)] + ([] if r == 8 else [box(18, 2, 28, 14)])
        (contour_dir/(stem+".txt")).write_text("\n".join(polygon_line(p) for p in contours)+"\n")
    output = tmp_path/"qc_cli"
    command = [sys.executable, str(SCRIPT_DIR/"run_4i_nucleus_segmentation_qc.py"), "--datapath", str(root),
               "--reference-round", "5", "--reference-contour-dir", config["segmentations"]["nuclei"]["directory"],
               "--use-binary-mask-reference", "--min-other-rounds", "7", "--iou-threshold", ".8", "--workers", "1",
               "--dpi", "50", "--render-scale", "1", "--output-dir", str(output)]
    finished = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert finished.returncode == 0, finished.stdout+finished.stderr
    flags = pd.read_csv(output/"myonis_4i/nucleus_qc.csv")
    assert flags.is_persistent.tolist() == [1, 0]
    configure(dataset, output/"myonis_4i")
    index = build_index(load_config(dataset))
    assert index["nucleus_qc"]["objects"].qc_pass.tolist() == [True, False]
    hashes = {name: file_hash(output/"myonis_4i"/name) for name in ["nucleus_qc.csv", "round_qc.csv", "patch_qc.csv"]}
    # -S proves summary-only export does not require NumPy, pandas, images or source contours.
    (root/"nuclei").rename(root/"nuclei_unavailable")
    finished = subprocess.run([sys.executable, "-S", str(SCRIPT_DIR/"run_4i_nucleus_segmentation_qc.py"),
                               "--summary-only", "--output-dir", str(output)], capture_output=True, text=True, timeout=30)
    assert finished.returncode == 0, finished.stdout+finished.stderr
    assert all(file_hash(output/"myonis_4i"/name) == digest for name, digest in hashes.items())
