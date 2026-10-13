from pathlib import Path
import csv

import numpy as np
from PIL import Image
import pytest
import tifffile
import yaml


def canonical(measurement, channel, fluor, marker, field=1, patch=1):
    x = (patch-1)*32
    return (f"plate-sc_rep5-6__rep-05__well-r04c02__field-f{field:02d}__cohort-m01-08__measurement-{measurement:02d}"
            f"__source-ch{channel:02d}__fluor-{fluor}__marker-{marker}__m01_08__field_{field:02d}__patch_{patch:04d}"
            f"__pr01-pc{patch:02d}__fieldrefy-0000-fieldrefx-{x:04d}__refy-000000-refx-{x:06d}")


def polygon_line(coords, size=32):
    return "0 "+" ".join(str(v/size) for xy in coords for v in xy)


def box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path/"input"
    root.mkdir()
    for name in ["nuclei", "myotubes", "sc35", "53bp1", "pred1", "pred2"]:
        (root/name).mkdir()
    rows = []
    specs = [("measurement_05_dapi",5,1,"dapi","dapi"), ("measurement_02_555_myhc",2,3,"555","myhc"),
             ("measurement_05_555_sc35",5,2,"555","sc35"), ("measurement_01_647_53bp1",1,4,"647","53bp1")]
    for field in [1,2]:
        for patch in [1,2]:
            for acq, m, ch, fluor, marker in specs:
                stem = canonical(m,ch,fluor,marker,field,patch)
                folder = root/"quantitative_patches"/acq; folder.mkdir(parents=True,exist_ok=True)
                image = np.full((32,32),2,dtype=np.float32)
                image[3:6,3:6] = 10
                image[3:7,8:12] = 20
                image[4:7,20:23] = 30
                image[24:27,5:8] = 40
                tifffile.imwrite(folder/(stem+".ome.tif"),image)
                rows.append(dict(quantitative_path=str(folder/(stem+".ome.tif")),acquisition_id=acq,patch_size=32,
                                 row=4,column=2,repeat=5,cell_line_row=3,field_id=field,cohort_id="m01_08",patch_index=patch))
                polygons = None
                if marker=="dapi":
                    polygons=[box(2,2,14,14),box(18,2,28,14)]; kind="nuclei"
                    Image.fromarray(np.full((32,32),64,dtype=np.uint8)).save(root/"pred1"/(stem+".png"))
                    Image.fromarray(np.full((32,32),128,dtype=np.uint8)).save(root/"pred2"/(stem+".png"))
                elif marker=="myhc":
                    polygons=[box(1,1,30,16),box(16,20,30,30)];kind="myotubes"
                else:
                    polygons=[box(3,3,5,5),box(8,3,11,6),box(20,4,22,6),box(5,24,7,26)];kind=marker
                (root/kind/(stem+".txt")).write_text("\n".join(polygon_line(p) for p in polygons)+"\n")
    with (root/"manifest.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    panel=dict(include_dapi=True,markers={
        "myhc":dict(acquisition_id="measurement_02_555_myhc"),
        "sc35":dict(acquisition_id="measurement_05_555_sc35",structure_source="sc35"),
        "53bp1":dict(acquisition_id="measurement_01_647_53bp1",structure_source="53bp1")})
    (tmp_path/"markers.yaml").write_text(yaml.safe_dump(panel))
    config=dict(schema_version=1,dataset_id="fixture",preprocessing=dict(root=str(root),image_manifest="manifest.csv",display_manifest=None),
                marker_config="markers.yaml",output_root="results",selection=dict(cohorts=["m01_08"]),
                geometry=dict(pixel_size_um=0.5,rim_width_px=2,perinuclear_width_px=3),
                segmentations=dict(nuclei=dict(directory="nuclei",acquisition_id="measurement_05_dapi",segmentation_id="n1"),
                                   myotubes=dict(directory="myotubes",acquisition_id="measurement_02_555_myhc",segmentation_id="m1"),
                                   structures={"sc35":dict(directory="sc35",acquisition_id="measurement_05_555_sc35",segmentation_id="s1"),
                                               "53bp1":dict(directory="53bp1",acquisition_id="measurement_01_647_53bp1",segmentation_id="b1")}),
                nuclear_aberration=dict(enabled=True,acquisition_id="measurement_05_dapi",prediction_directories=["pred1","pred2"]),
                texture=dict(enabled=True,levels=16,distances_px=[1],fallback_range=[0,40]),
                execution=dict(workers=1,hash_inputs=True),outputs=dict(pdf=True,qc_overlays=0))
    path=tmp_path/"dataset.yaml";path.write_text(yaml.safe_dump(config))
    return path
