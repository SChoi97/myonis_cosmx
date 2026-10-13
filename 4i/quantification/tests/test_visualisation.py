import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from fouri_quantification.config import load_config
from fouri_quantification.inputs.manifests import build_index
from fouri_quantification.pipeline import run_pipeline, execute_task, finalize
from fouri_quantification.reporting.visualisation import audit_regions, read_qc_mask, region


def enable_qc(dataset, markers="all", **options):
    data=yaml.safe_load(dataset.read_text())
    data["selection"].update(max_fields=1,field_sampling="first")
    options.setdefault("export_masks", True)
    options.setdefault("max_patches_per_field", None)
    data["visualisation"]=dict(enabled=True,markers=markers,dpi=70,max_labels_per_panel=5,**options)
    dataset.write_text(yaml.safe_dump(data))


@pytest.fixture
def renderer():
    return pytest.importorskip("myonis_vis", reason="Install the local MYONIS-VisTools package to test detailed QC")


def test_select_50_complete_fields_is_global_and_order_independent(dataset):
    config=load_config(dataset)
    manifest=Path(config["preprocessing"]["image_manifest"])
    original=pd.read_csv(manifest)
    template=original[original.field_id==1]
    frames=[]
    for field in range(1,61):
        frame=template.copy()
        frame["field_id"]=field
        frame["quantitative_path"]=frame.quantitative_path.str.replace("__field-f01__",f"__field-f{field:02d}__",regex=False).str.replace("__field_01__",f"__field_{field:02d}__",regex=False)
        frames.append(frame)
    pd.concat(frames).to_csv(manifest,index=False)
    config["selection"].update(max_fields=50,field_sampling="random",field_seed=17)
    one=build_index(config)
    assert one["field_selection"]["eligible_fields"]==60
    assert one["images"].field_key.nunique()==50
    assert one["images"].groupby("field_key").size().eq(8).all()
    pd.read_csv(manifest).sample(frac=1,random_state=123).to_csv(manifest,index=False)
    two=build_index(config)
    assert set(one["images"].field_key)==set(two["images"].field_key)
    config["selection"]["field_seed"]=18
    assert set(build_index(config)["images"].field_key)!=set(one["images"].field_key)


@pytest.mark.parametrize("value",[0,-1,True,"50",2.5])
def test_invalid_field_limit(dataset,value):
    data=yaml.safe_load(dataset.read_text());data["selection"]["max_fields"]=value
    dataset.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match="max_fields"):load_config(dataset)


def test_filter_precedes_limit_and_shortfall_is_recorded(dataset):
    config=load_config(dataset)
    config["selection"].update(fields=[2],max_fields=50)
    index=build_index(config)
    assert set(index["images"].field_id)=={2}
    assert index["field_selection"]["selected_fields"]==1
    assert "fewer_fields_available" in set(index["issues"].status)


def test_independent_audit_detects_shifted_masks():
    image=np.arange(100,dtype=np.float32).reshape(10,10)
    correct=np.zeros((10,10),bool);correct[2:5,3:7]=True
    row=dict(object_uid="n1",local_id=0,m__whole_nucleus__mean=float(image[correct].mean()),
             m__whole_nucleus__integrated=float(image[correct].sum()),m__whole_nucleus__n_pixels=int(correct.sum()))
    item=region(row,np.roll(correct,1,axis=1),"nucleus","whole_nucleus","m")
    audit,_=audit_regions([item],image,image.shape)
    assert audit.iloc[0].status=="mismatch"
    assert not audit.iloc[0].mean_matches and audit.iloc[0].area_matches


def test_qc_exact_masks_hierarchy_intensity_and_resume(dataset,tmp_path,renderer):
    enable_qc(dataset,["sc35","myhc","nuclear_aberration"])
    directory,manifest=run_pipeline(dataset,tmp_path/"visual")
    assert manifest["visualisation_status"]=="complete",manifest
    assert manifest["visualisation_views"]==6
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    views=run.uns["qc"]["visualisations"]
    assert set(views.patch_index)=={1,2}
    assert set(views.status)=={"complete"}
    assert (directory/"qc_visualisations/index.html").is_file()
    assert len(json.loads((directory/"inputs/selected_fields.json").read_text()))==1
    sc=views[(views.marker=="sc35")&(views.patch_index==1)].iloc[0]
    audit=pd.read_csv(directory/sc.audit,sep="\t")
    assert set(audit.status)=={"pass"}
    assert {"object","inside_parent_nucleus","structures","nuclear_structures","cytoplasmic_structures"}.issubset(set(audit.compartment))
    source=json.loads((directory/sc.metadata).read_text())["source_path"]
    image=tifffile.imread(source)
    for _,row in audit.iterrows():
        mask=read_qc_mask(directory/sc.masks,row.mask_key)
        assert mask.sum()==row.n_pixels
        if mask.any():
            assert np.isclose(image[mask].mean(dtype=np.float64),row.saved_mean)
    nucleus=audit[(audit.label=="N0")&(audit.compartment=="structures")].iloc[0]
    assert nucleus.saved_mean==16.4
    structures=audit[(audit.object_level=="structure")&(audit.compartment=="object")]
    assert len(structures)==4 and structures.parent_nucleus_uid.notna().sum()==3
    myo=views[(views.marker=="myhc")&(views.patch_index==1)].iloc[0]
    ma=pd.read_csv(directory/myo.audit,sep="\t")
    nmasks=[read_qc_mask(directory/myo.masks,r.mask_key) for _,r in ma[ma.object_level=="nucleus"].iterrows()]
    for _,row in ma[ma.compartment=="cytoplasm"].iterrows():
        mask=read_qc_mask(directory/myo.masks,row.mask_key)
        assert not (mask & np.logical_or.reduce(nmasks)).any()
    prob=views[(views.marker=="nuclear_aberration")&(views.patch_index==1)].iloc[0]
    pa=pd.read_csv(directory/prob.audit,sep="\t")
    np.testing.assert_allclose(pa.saved_mean,96/255,rtol=1e-6)
    assert set(pa.status)=={"pass"}
    checkpoint=next((directory/"checkpoints").glob("*.h5ad"))
    before=checkpoint.stat().st_mtime_ns
    assert execute_task(directory,0)["status"]=="resumed"
    assert checkpoint.stat().st_mtime_ns==before
    (directory/sc.png).unlink()
    assert finalize(directory)["visualisation_status"]=="failed"
    assert execute_task(directory,0)["status"]=="complete"
    assert (directory/sc.png).is_file()


def test_missing_structure_qc_is_explicit_and_does_not_create_zero_masks(dataset,tmp_path,renderer):
    config=load_config(dataset)
    for path in Path(config["segmentations"]["structures"]["sc35"]["directory"]).glob("*.txt"):
        path.unlink()
    enable_qc(dataset,["sc35"],max_patches_per_field=1)
    directory,result=run_pipeline(dataset,tmp_path/"missing_qc")
    assert result["visualisation_views"]==1 and result["visualisation_status"]=="complete"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    view=run.uns["qc"]["visualisations"].iloc[0]
    assert view.detail=="missing"
    audit=pd.read_csv(directory/view.audit,sep="\t")
    assert not audit.object_level.eq("structure").any()
    assert not audit.compartment.eq("structures").any()


def test_render_failure_retains_measurements_and_is_reported(dataset,tmp_path,monkeypatch,renderer):
    import myonis_vis
    enable_qc(dataset,["sc35"],max_patches_per_field=1)
    def fail(*args,**kwargs):raise RuntimeError("injected renderer failure")
    monkeypatch.setattr(myonis_vis,"render_contours",fail)
    directory,result=run_pipeline(dataset,tmp_path/"broken_render")
    assert result["visualisation_status"]=="failed"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    assert run.n_obs==4
    assert "injected renderer failure" in run.uns["qc"]["visualisations"].iloc[0].detail


def test_sigmoid_qc_uses_prediction_grid_and_available_model_ensemble(dataset,tmp_path,renderer):
    from PIL import Image
    config=load_config(dataset)
    for i,folder in enumerate(config["nuclear_aberration"]["prediction_directories"]):
        for path in Path(folder).glob("*.png"):
            Image.fromarray(np.full((16,16),64+64*i,dtype=np.uint8)).save(path)
    second=Path(config["nuclear_aberration"]["prediction_directories"][1])
    sorted(second.glob("*.png"))[0].unlink()
    enable_qc(dataset,["nuclear_aberration"],max_patches_per_field=1)
    directory,result=run_pipeline(dataset,tmp_path/"prediction_grid")
    assert result["visualisation_status"]=="complete"
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    view=run.uns["qc"]["visualisations"].iloc[0]
    audit=pd.read_csv(directory/view.audit,sep="\t")
    assert set(audit.status)=={"pass"}
    with np.load(directory/view.masks,allow_pickle=False) as archive:
        assert tuple(archive["shape"])==(16,16)
        np.testing.assert_allclose(archive["source_values"],64/255,rtol=1e-6)


def test_parallel_qc_keeps_separate_field_outputs(dataset,tmp_path,renderer):
    enable_qc(dataset,["sc35"],max_patches_per_field=1)
    config=yaml.safe_load(dataset.read_text());config["selection"]["max_fields"]=2
    dataset.write_text(yaml.safe_dump(config))
    directory,result=run_pipeline(dataset,tmp_path/"parallel_qc",workers=2)
    assert result["visualisation_status"]=="complete" and result["visualisation_views"]==2
    run=ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    views=run.uns["qc"]["visualisations"]
    assert run.n_obs==8 and views.field_key.nunique()==2
    assert len({Path(p).parts[1] for p in views.png})==2
    for _,row in views.iterrows():
        assert (directory/row.png).is_file()
        assert pd.read_csv(directory/row.audit,sep="\t").status.eq("pass").all()
