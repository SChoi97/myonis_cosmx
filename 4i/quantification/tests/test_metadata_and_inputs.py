from pathlib import Path

import pytest
import yaml

from fouri_quantification.config import load_config
from fouri_quantification.inputs.metadata import parse_filename
from fouri_quantification.inputs.manifests import build_index
from conftest import canonical


def test_metadata_and_physical_row(dataset):
    config=load_config(dataset)
    meta=parse_filename(canonical(1,4,"647","53bp1",1,2),"fixture",config["plate_layout_data"])
    assert (meta["cell_line"],meta["condition"],meta["repeat"],meta["plate_id"]) == ("K32Del","K32Del",5,1)
    assert (meta["patch_index"],meta["patch_column"],meta["field_reference_x"]) == (2,2,32)
    index=build_index(config)
    assert set(index["images"].cell_line)=={"K32Del"}
    assert set(index["images"].legacy_cell_line_row)=={"3"}
    assert len(index["images"])==16
    for row,line in [(2,"NCRM1"),(3,"NCRM5"),(7,"L302P-Corr")]:
        assert parse_filename(canonical(1,4,"647","53bp1").replace("r04c02",f"r{row:02d}c02"),"fixture",config["plate_layout_data"])["cell_line"]==line


def test_conflicting_tokens_and_repeat_fail(dataset):
    config=load_config(dataset)
    for bad in [canonical(1,4,"647","53bp1").replace("__field_01__","__field_02__"),
                canonical(1,4,"647","53bp1").replace("rep-05","rep-07")]:
        with pytest.raises(ValueError): parse_filename(bad,"fixture",config["plate_layout_data"])


def test_repeat_is_part_of_global_identity(dataset):
    config=load_config(dataset)
    name=canonical(1,4,"647","53bp1")
    one=parse_filename(name,"fixture",config["plate_layout_data"])
    two=parse_filename(name.replace("rep-05","rep-06"),"fixture",config["plate_layout_data"])
    assert one["field_key"] != two["field_key"]
    assert one["spatial_key"] != two["spatial_key"]


def test_manifest_cannot_mislabel_acquisition(dataset):
    import pandas as pd
    config=load_config(dataset)
    path=Path(config["preprocessing"]["image_manifest"])
    frame=pd.read_csv(path); frame.loc[0,"acquisition_id"]="measurement_01_647_53bp1";frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="acquisition_id"):
        build_index(config)


def test_missing_is_not_empty(dataset):
    config=load_config(dataset)
    paths=sorted(Path(config["segmentations"]["structures"]["sc35"]["directory"]).glob("*.txt"))
    paths[0].write_text("")
    paths[1].unlink()
    index=build_index(config)
    statuses=index["assets"][index["assets"].source_key=="structures.sc35"].status
    assert (statuses=="present").sum()==3
    assert (statuses=="missing").sum()==1


def test_rebase_manifest_paths(dataset):
    config=load_config(dataset)
    manifest=Path(config["preprocessing"]["image_manifest"])
    original=manifest.read_text()
    manifest.write_text(original.replace(config["preprocessing"]["root"],"/nemo/nonexistent/old/run"))
    assert build_index(config)["images"].available.all()


def test_duplicate_and_cross_cohort_reference_rejected(dataset):
    config=load_config(dataset)
    manifest=Path(config["preprocessing"]["image_manifest"])
    text=manifest.read_text();manifest.write_text(text+text.splitlines()[1]+"\n")
    with pytest.raises(ValueError,match="Duplicate"):build_index(config)
    manifest.write_text(text)
    config["segmentations"]["nuclei"]["acquisition_id"]="measurement_09_dapi"
    with pytest.raises(ValueError,match="cross-cohort"):build_index(config)
