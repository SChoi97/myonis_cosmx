import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pypdf import PdfReader
import pytest
import yaml

from fouri_quantification.pipeline import run_pipeline, prepare, execute_task, finalize
from fouri_quantification.access import get_nuclei, get_myotubes, get_structures, subset_by_myotubes
from fouri_quantification.config import load_config


def test_end_to_end_hierarchy_h5ad_pdf_and_resume(dataset,tmp_path):
    directory,manifest=run_pipeline(dataset,tmp_path/"serial")
    assert manifest["status"]=="complete", list((directory/"logs").glob("*"))
    assert manifest["pdf_status"]=="complete", manifest.get("pdf_error")
    assert manifest["measurements_h5ad_status"]=="complete", manifest.get("measurements_h5ad_error")
    assert (directory/"myonis_4i_data_measurements.h5ad").is_file()
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    slim=ad.read_h5ad(directory/"myonis_4i_data_measurements.h5ad")
    assert slim.n_obs==run.n_obs and len(slim.uns["myotubes"])==len(run.uns["myotubes"])
    assert "structures" not in slim.uns and "geometry" not in slim.uns
    nuclei=get_nuclei(run);myotubes=get_myotubes(run);sc35=get_structures(run,"sc35")
    assert len(nuclei)==8 and len(myotubes)==8 and len(sc35)==16
    assert set(nuclei.cell_line)=={"K32Del"}
    assert set(nuclei["repeat"])=={5}
    first=nuclei[nuclei.local_id==0]
    np.testing.assert_allclose(first.sc35__structures__count,2)
    np.testing.assert_allclose(first.sc35__structures__mean_object_intensity,15)
    np.testing.assert_allclose(first.sc35__structures__mean_structure_pixel_intensity,16.4)
    np.testing.assert_allclose(nuclei.nuclear_aberration_mean_sigmoid,96/255,rtol=1e-6)
    assert myotubes.n_nuclei.sum()==8
    assert myotubes.sc35__structures__count_total.sum()==12
    assert (sc35.parent_nucleus_uid=="").sum()==4
    assert len(PdfReader(directory/"run_summary.pdf").pages)==3
    assert "K32Del" in PdfReader(directory/"run_summary.pdf").pages[0].extract_text()
    selected=myotubes[myotubes.n_nuclei>0].index[:1]
    sub=subset_by_myotubes(run,selected)
    assert sub.n_obs==2 and len(sub.uns["structures"]["sc35"])==3
    assert len(sub.uns["myotubes"])==1
    myo_ad=get_myotubes(run,as_anndata=True)
    assert myo_ad.n_obs==8 and myo_ad.n_vars==run.n_vars
    checkpoints={p:p.stat().st_mtime_ns for p in (directory/"checkpoints").glob("*.h5ad")}
    run_pipeline(dataset,directory,resume=True)
    assert checkpoints=={p:p.stat().st_mtime_ns for p in checkpoints}


def test_parallel_matches_serial(dataset,tmp_path):
    one,_=run_pipeline(dataset,tmp_path/"one",workers=1)
    two,result=run_pipeline(dataset,tmp_path/"two",workers=2)
    assert result["status"]=="complete"
    a,b=ad.read_h5ad(one/"myonis_4i_data.h5ad"),ad.read_h5ad(two/"myonis_4i_data.h5ad")
    pd.testing.assert_frame_equal(get_nuclei(a),get_nuclei(b))
    pd.testing.assert_frame_equal(a.uns["myotubes"],b.uns["myotubes"])
    pd.testing.assert_frame_equal(a.uns["structures"]["sc35"],b.uns["structures"]["sc35"])


def test_missing_and_empty_structure_counts(dataset,tmp_path):
    config=load_config(dataset)
    paths=sorted(Path(config["segmentations"]["structures"]["sc35"]["directory"]).glob("*.txt"))
    paths[0].write_text("");paths[1].unlink()
    directory,result=run_pipeline(dataset,tmp_path/"missing")
    assert result["status"]=="complete_with_omissions"
    nuclei=get_nuclei(ad.read_h5ad(directory/"myonis_4i_data.h5ad"))
    f1=nuclei[nuclei.field_id==1]
    assert (f1[f1.patch_index==1].sc35__structures__count==0).all()
    assert f1[f1.patch_index==2].sc35__structures__count.isna().all()


def test_resume_rejects_changed_input(dataset,tmp_path):
    directory=prepare(dataset,tmp_path/"run")
    config=load_config(dataset)
    path=next(Path(config["segmentations"]["structures"]["sc35"]["directory"]).glob("*.txt"))
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="Cannot resume"):
        prepare(dataset,directory,resume=True)


def test_finalize_partial_slurm_run(dataset,tmp_path):
    directory=prepare(dataset,tmp_path/"partial")
    execute_task(directory,0)
    result=finalize(directory)
    assert result["status"]=="partial" and result["failed_tasks"]==1
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.n_obs==4 and len(run.uns["qc"]["failures"])==1


def test_zero_object_run_is_valid(dataset,tmp_path):
    config=load_config(dataset)
    for name in ["nuclei","myotubes"]:
        for p in Path(config["segmentations"][name]["directory"]).glob("*.txt"):p.write_text("")
    for source in config["segmentations"]["structures"].values():
        for p in Path(source["directory"]).glob("*.txt"):p.write_text("")
    directory,result=run_pipeline(dataset,tmp_path/"zero")
    assert result["status"]=="complete" and result["pdf_status"]=="complete",result
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.n_obs==0 and run.obsm["morphology_features"].shape==(0,4)
    assert len(run.uns["myotubes"])==0


def test_myhc_filter_keeps_source_identity(dataset,tmp_path):
    config=load_config(dataset)
    root=Path(config["preprocessing"]["root"])
    rows=[]
    for p in Path(config["segmentations"]["myotubes"]["directory"]).glob("*.txt"):
        rows.extend([dict(image_name=p.stem,contour_idx=0,average_intensity=10),dict(image_name=p.stem,contour_idx=1,average_intensity=20)])
    pd.DataFrame(rows).to_csv(root/"filter.csv",index=False)
    raw=yaml.safe_load(dataset.read_text());raw["myhc_filter"]=dict(enabled=True,metadata_csv="filter.csv",intensity_threshold=15)
    dataset.write_text(yaml.safe_dump(raw))
    directory,result=run_pipeline(dataset,tmp_path/"filtered")
    assert result["status"]=="complete"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    myotubes=get_myotubes(run)
    assert set(myotubes.source_object_index)=={1}
    assert set(myotubes.local_id)=={0}
    assert all(uid.endswith("row2") for uid in myotubes.index)
    assert (run.uns["object_registry"].exclusion_reason=="myhc_intensity_filter").sum()==4
    assert (run.obs.parent_myotube_uid=="").all()


def test_missing_prediction_model_uses_available_ensemble(dataset,tmp_path):
    config=load_config(dataset)
    p=sorted(Path(config["nuclear_aberration"]["prediction_directories"][1]).glob("*.png"))[0]
    p.unlink()
    directory,result=run_pipeline(dataset,tmp_path/"ensemble")
    assert result["status"]=="complete_with_omissions"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    subset=run.obs[(run.obs.field_id==1)&(run.obs.patch_index==1)]
    np.testing.assert_allclose(subset.nuclear_aberration_mean_sigmoid,64/255,rtol=1e-6)


def test_missing_all_predictions_preserves_nan_aggregation(dataset,tmp_path):
    config=load_config(dataset)
    for directory in config["nuclear_aberration"]["prediction_directories"]:
        for p in Path(directory).glob("*.png"):p.unlink()
    directory,result=run_pipeline(dataset,tmp_path/"no_predictions")
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.obs.nuclear_aberration_mean_sigmoid.isna().all()
    myotubes=get_myotubes(run)
    assert myotubes[myotubes.n_nuclei>0].nuclear_aberration_assigned_nuclei_total_sigmoid.isna().all()
    assert (myotubes[myotubes.n_nuclei==0].nuclear_aberration_assigned_nuclei_total_sigmoid==0).all()


def test_direct_cytoplasmic_structures_are_not_double_counted(dataset,tmp_path):
    panel_path=dataset.parent/"markers.yaml"
    panel=yaml.safe_load(panel_path.read_text())
    panel["markers"]["sc35"]["structure_parent"]="nucleus_then_myotube"
    panel_path.write_text(yaml.safe_dump(panel))
    config=load_config(dataset)
    from conftest import box,polygon_line
    for path in Path(config["segmentations"]["structures"]["sc35"]["directory"]).glob("*.txt"):
        lines=path.read_text().splitlines();lines[-1]=polygon_line(box(20,24,22,26));path.write_text("\n".join(lines)+"\n")
    directory,result=run_pipeline(dataset,tmp_path/"direct")
    assert result["status"]=="complete"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    objects=get_structures(run,"sc35");myotubes=get_myotubes(run)
    assert (objects.myotube_link_method=="direct_cytoplasmic").sum()==4
    direct=objects[objects.myotube_link_method=="direct_cytoplasmic"]
    assert (direct.area_in_parent_px2>0).all()
    assert direct.intensity_mean_in_parent.notna().all()
    assert myotubes.sc35__structures__count_total.sum()==16
    assert myotubes.sc35__nuclear_structures__count.sum()==12
    assert myotubes.sc35__cytoplasmic_structures__count.sum()==4


def test_missing_marker_acquisition_is_not_outside_selection(dataset,tmp_path):
    config=load_config(dataset)
    manifest=Path(config["preprocessing"]["image_manifest"])
    rows=pd.read_csv(manifest);rows=rows[rows.acquisition_id!="measurement_05_555_sc35"];rows.to_csv(manifest,index=False)
    with pytest.raises(ValueError,match="Preflight"):
        prepare(dataset,tmp_path/"strict_missing_channel")


def test_wide_h5ad_tables_round_trip(tmp_path):
    from fouri_quantification.outputs.h5ad import write_partition,read_partition
    columns=[f"resident_nuclei__marker{i}__whole_nucleus__intensity_mean__mean" for i in range(5000)]
    frame=pd.DataFrame(np.arange(10000,dtype=float).reshape(2,5000),columns=columns)
    path=tmp_path/"wide.h5ad"
    write_partition(dict(myotubes=frame),path)
    actual=read_partition(path)["myotubes"]
    frame.index=frame.index.astype(str);frame.index.name="row_id"
    pd.testing.assert_frame_equal(frame,actual)


def test_worker_rejects_modified_prepared_input(dataset,tmp_path):
    directory=prepare(dataset,tmp_path/"mutated")
    config=load_config(dataset)
    path=sorted(Path(config["segmentations"]["nuclei"]["directory"]).glob("*.txt"))[0]
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="Input changed"):
        execute_task(directory,0)


def test_renamed_assets_keep_canonical_identity_and_sigmoid_values(dataset,tmp_path):
    config=load_config(dataset)
    rows=[]
    for source,folder in [("nuclei",config["segmentations"]["nuclei"]["directory"]),
                          ("aberration.0",config["nuclear_aberration"]["prediction_directories"][0])]:
        for i,path in enumerate(sorted(Path(folder).glob("*"))):
            renamed=path.with_name(f"renamed_{i}{path.suffix}")
            rows.append(dict(source_key=source,path=str(renamed),image_name=path.name))
            path.rename(renamed)
    mapping=tmp_path/"assets.csv";pd.DataFrame(rows).to_csv(mapping,index=False)
    raw=yaml.safe_load(dataset.read_text());raw["asset_map"]=str(mapping);dataset.write_text(yaml.safe_dump(raw))
    directory,result=run_pipeline(dataset,tmp_path/"renamed")
    assert result["status"]=="complete"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.obs.image_name.str.startswith("plate-sc_").all()
    assert run.obs.source_path.str.contains("renamed_").all()
    np.testing.assert_allclose(run.obs.nuclear_aberration_mean_sigmoid,96/255,rtol=1e-6)


def test_worker_rejects_changed_software_snapshot(dataset,tmp_path):
    directory=prepare(dataset,tmp_path/"code_changed")
    path=directory/"run_manifest.json";snapshot=json.loads(path.read_text())
    snapshot["code_hashes"]={};path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError,match="code changed"):
        execute_task(directory,0)


def test_feature_dictionary_preserves_aggregate_units():
    from fouri_quantification.features.registry import describe
    for feature,unit in [("sc35__structures__count","count"),("sc35__structures__area_mean_px2","px^2"),
                         ("myhc__whole_nucleus__entropy","bits"),("myhc__whole_nucleus__mean","image intensity units")]:
        assert describe(f"resident_nuclei__{feature}__mean","myotube","uns")["units"]==unit
        assert describe(f"resident_nuclei__{feature}__n_valid","myotube","uns")["units"]=="count"
    assert describe("nuclei_density_per_1000px2","myotube","uns")["units"]=="nuclei / 1000 px^2"
