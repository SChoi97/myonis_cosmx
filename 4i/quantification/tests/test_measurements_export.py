import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fouri_quantification.outputs.measurements import make_measurements_h5ad, make_split_measurements_h5ad


def test_measurements_export_keeps_parent_features_and_drops_object_payload(tmp_path):
    obs = pd.DataFrame({"parent_myotube_uid": ["m1", "m1"], "qc_pass": [True, False],
                        "cell_line": ["K32Del", "K32Del"]}, index=["n1", "n2"])
    obs.index.name = "object_uid"
    run = ad.AnnData(np.array([[4.0], [8.0]], dtype=np.float32), obs=obs,
                     var=pd.DataFrame({"marker": ["dapi"]},
                         index=pd.Index(["measurement_02_dapi"], name="acquisition_id")))
    run.obsm["structure_features"] = pd.DataFrame({"sc35__structures__count": [2.0, 3.0],
        "sc35__structures__mean_object_intensity": [10.0, 12.0]}, index=run.obs_names)
    myotubes = pd.DataFrame({"n_nuclei": [2], "dapi__whole_myotube__mean": [6.0],
        "source_path": ["/very/long/source/path"], "image_name": ["long-image-name"],
        "resident_nuclei__sc35__structures__count__mean": [2.5],
        "resident_nuclei__sc35__structures__count__n_valid": [2],
        "resident_nuclei__sc35__structures__count__median": [2.5],
        "resident_nuclei__sc35__structures__count__std": [0.5]},
        index=pd.Index(["m1"], name="object_uid"))
    run.uns["myotubes"] = myotubes
    run.uns["feature_dictionary"] = pd.DataFrame({"feature": ["sc35__structures__count", "sc35__object__mean",
        "resident_nuclei__sc35__structures__count__mean", "resident_nuclei__sc35__structures__count__median"],
        "storage_location": ['obsm["structure_features"]', 'uns["structures"]["sc35"]',
                             'uns["myotubes"]', 'uns["myotubes"]']})
    run.uns["structures"] = {"sc35": pd.DataFrame({"parent_nucleus_uid": ["n1"]}, index=["s1"])}
    run.uns["geometry"] = {"points": np.array([[1.0, 2.0]], dtype=np.float32)}
    run.uns["object_registry"] = pd.DataFrame({"object_uid": ["s1"]})
    run.uns["relationships"] = {"candidates": pd.DataFrame({"child_uid": ["s1"]})}
    run.uns["qc"] = {"field_summary": pd.DataFrame({"n_nuclei": [2]}),
                     "nucleus_qc": {"rounds": pd.DataFrame({"round": [1]})}}
    source = tmp_path / "myonis_4i_data.h5ad"
    run.write_h5ad(source)

    output = make_measurements_h5ad(source)
    assert output.name == "myonis_4i_data_measurements.h5ad"
    slim = ad.read_h5ad(output)
    np.testing.assert_array_equal(slim.X, run.X)
    pd.testing.assert_frame_equal(slim.obs, run.obs)
    pd.testing.assert_frame_equal(slim.obsm["structure_features"], run.obsm["structure_features"])
    pd.testing.assert_frame_equal(slim.uns["myotubes"], run.uns["myotubes"])
    assert list(slim.uns["feature_dictionary"].feature) == ["sc35__structures__count",
        "resident_nuclei__sc35__structures__count__mean", "resident_nuclei__sc35__structures__count__median"]
    assert "field_summary" in slim.uns["qc"]
    assert "nucleus_qc" not in slim.uns["qc"]
    for key in ["structures", "geometry", "object_registry", "relationships"]:
        assert key not in slim.uns
    assert source.exists()

    nuclei_path, myotubes_path = make_split_measurements_h5ad(source, compact_myotubes=True)
    nuclei = ad.read_h5ad(nuclei_path)
    tubes = ad.read_h5ad(myotubes_path)
    assert nuclei.obs.parent_myotube_uid.eq("m1").all()
    assert nuclei.obs.qc_pass.tolist() == [True, False]
    assert "myotubes" not in nuclei.uns
    assert tubes.obs.index.tolist() == ["m1"]
    assert int(tubes.obs.n_nuclei.iloc[0]) == 2
    np.testing.assert_allclose(tubes.X, [[6.0]])
    assert "resident_nuclei__sc35__structures__count__mean" in tubes.obs
    assert "resident_nuclei__sc35__structures__count__n_valid" in tubes.obs
    for name in ["resident_nuclei__sc35__structures__count__median",
                 "resident_nuclei__sc35__structures__count__std", "source_path", "image_name"]:
        assert name not in tubes.obs
    assert set(tubes.uns["feature_dictionary"].storage_location) == {"obs"}

    make_split_measurements_h5ad(source)
    full_tubes = ad.read_h5ad(myotubes_path)
    assert "resident_nuclei__sc35__structures__count__median" in full_tubes.obs
    assert "source_path" in full_tubes.obs


def test_repeat_export_filters_both_object_levels_and_sample_qc(tmp_path):
    obs = pd.DataFrame({
        "repeat": [5, 6, 5],
        "parent_myotube_uid": ["m5", "m6", ""],
        "qc_pass": [True, False, True],
    }, index=pd.Index(["n5a", "n6", "n5b"], name="object_uid"))
    run = ad.AnnData(
        np.array([[5.0], [6.0], [7.0]], dtype=np.float32), obs=obs,
        var=pd.DataFrame({"marker": ["dapi"]},
                         index=pd.Index(["dapi_ref"], name="acquisition_id")))
    run.obsm["structure_features"] = pd.DataFrame(
        {"sc35__structures__count": [2.0, 3.0, 4.0]}, index=run.obs_names)
    run.uns["myotubes"] = pd.DataFrame({
        "object_uid": ["m6", "m5"], "repeat": [6, 5], "n_nuclei": [1, 1],
        "dapi__whole_myotube__mean": [60.0, 50.0],
        "resident_nuclei__sc35__structures__count__mean": [3.0, 2.0],
        "source_path": ["path6", "path5"],
    }, index=pd.Index(["m6", "m5"], name="object_uid"))
    run.uns["sample_metadata"] = pd.DataFrame({"repeat": [5, 6], "sample_id": ["s5", "s6"]})
    run.uns["qc"] = {
        "field_summary": pd.DataFrame({"field_key": ["d|p|w|5|c|1", "d|p|w|6|c|1"],
                                       "n_nuclei": [2, 1]}),
        "excluded_fields": pd.DataFrame({"repeat": [5, 6], "reason": ["x", "y"]}),
        "scope": "Selected fields",
    }
    source = tmp_path / "myonis_4i_data.h5ad"
    run.write_h5ad(source, compression="gzip")

    output = make_measurements_h5ad(source, repeat=5)
    assert output.name == "myonis_4i_data_measurements_rep_5.h5ad"
    slim = ad.read_h5ad(output)
    assert slim.obs.index.tolist() == ["n5a", "n5b"]
    assert slim.obs.repeat.tolist() == [5, 5]
    np.testing.assert_allclose(slim.X[:, 0], [5.0, 7.0])
    assert slim.obsm["structure_features"]["sc35__structures__count"].tolist() == [2.0, 4.0]
    assert slim.uns["myotubes"].index.tolist() == ["m5"]
    assert slim.uns["myotubes"].repeat.tolist() == [5]
    assert slim.uns["sample_metadata"].sample_id.tolist() == ["s5"]
    assert slim.uns["qc"]["field_summary"].n_nuclei.tolist() == [2]
    assert slim.uns["qc"]["excluded_fields"].reason.tolist() == ["x"]
    assert slim.obs.parent_myotube_uid.tolist() == ["m5", ""]

    nuclei_path, tubes_path = make_split_measurements_h5ad(source, compact_myotubes=True,
                                                            repeat=6)
    assert nuclei_path.name == "myonis_4i_data_nuclei_measurements_rep_6.h5ad"
    assert tubes_path.name == "myonis_4i_data_myotubes_measurements_rep_6.h5ad"
    nuclei, tubes = ad.read_h5ad(nuclei_path), ad.read_h5ad(tubes_path)
    assert nuclei.obs.index.tolist() == ["n6"]
    assert nuclei.obs.parent_myotube_uid.tolist() == ["m6"]
    assert tubes.obs.index.tolist() == ["m6"]
    assert tubes.obs.repeat.tolist() == [6]
    np.testing.assert_allclose(tubes.X, [[60.0]])
    assert "source_path" not in tubes.obs
    assert tubes.uns["sample_metadata"].sample_id.tolist() == ["s6"]
    assert tubes.uns["qc"]["excluded_fields"].reason.tolist() == ["y"]

    with pytest.raises(ValueError, match="has no nuclei or myotubes"):
        make_measurements_h5ad(source, repeat=7)
