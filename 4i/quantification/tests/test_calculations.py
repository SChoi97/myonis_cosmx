import ast
from pathlib import Path

import numpy as np
import pytest

from fouri_quantification.compatibility import _geometry, _aberration, _filtering
from fouri_quantification.objects.relationships import assign
from fouri_quantification.features.structures import structure_summary
from fouri_quantification.features.texture import texture


def obj(coords):
    p=np.array(coords,dtype=np.float32)
    return dict(object_uid=str(coords),polygon=p,local_polygon=p,bbox=_geometry.compute_bbox(p))


def test_legacy_function_bodies_are_unchanged():
    repo=Path(__file__).resolve().parents[3]
    specs=[(_geometry,"cosmx_utils.py"),(_aberration,"cosmx_aberration_utils.py"),(_filtering,"cosmx_h5ad_utils.py")]
    for module,original in specs:
        src=repo/"generate_h5ad/utils"/original
        if not src.exists():pytest.skip("Original checkout unavailable in installed distribution")
        original_tree=ast.parse(src.read_text())
        originals={n.name:ast.dump(n,include_attributes=False) for n in original_tree.body if isinstance(n,ast.FunctionDef)}
        for node in ast.parse(Path(module.__file__).read_text()).body:
            if isinstance(node,ast.FunctionDef):assert ast.dump(node,include_attributes=False)==originals[node.name]


def test_strict_threshold_and_tie_order():
    child=obj([(0,0),(10,0),(10,10),(0,10)])
    exactly=obj([(0,0),(8,0),(8,10),(0,10)])
    assert assign([child],[exactly],.8)[0]=={}
    mapping,rows,evidence=assign([child],[child,child],.8)
    assert mapping=={0:0}
    assert rows[0]["overlap_tie"] and rows[0]["competing_parent"]


def test_legacy_morphology_known_square():
    square=obj([(0,0),(10,0),(10,10),(0,10)])
    vals,names=_geometry.compute_morphology_features([square["polygon"]])
    np.testing.assert_allclose(vals[0],[100,40,np.pi/4,np.sqrt(200)],rtol=1e-6)


def test_overlap_union_and_weighted_intensity():
    parent=np.ones((4,4),bool);image=np.arange(16,dtype=float).reshape(4,4)
    items=[dict(pixels=np.array([0,1,2,3]),area_px2=4,mean_in_parent=1.5),
           dict(pixels=np.array([2,3,4,5]),area_px2=4,mean_in_parent=3.5)]
    result=structure_summary(items,image,parent)
    assert result["count"]==2
    assert result["occupied_area_px2"]==6
    assert result["integrated_structure_intensity"]==15
    assert result["mean_object_intensity"]==2.5
    assert np.isnan(structure_summary(items,image,parent,False)["count"])
    empty=structure_summary([],image,parent)
    assert empty["count"]==0 and empty["integrated_structure_intensity"]==0
    assert np.isnan(empty["mean_object_intensity"])


def test_texture_ignores_exterior_and_marks_small_objects():
    settings=dict(distances_px=[1],min_pixels=4,min_pairs=4,levels=8)
    mask=np.zeros((10,10),bool);mask[2:8,2:8]=True
    a=np.full((10,10),1000.);a[mask]=4
    b=np.full((10,10),-900.);b[mask]=4
    ra,rb=texture(a,mask,settings,[0,8]),texture(b,mask,settings,[0,8])
    assert ra["entropy"]==rb["entropy"]==0
    assert ra["glcm_contrast_d1px"]==0 and ra["glcm_homogeneity_d1px"]==1
    assert np.isnan(ra["glcm_correlation_d1px"])
    assert np.isnan(texture(a,np.eye(10,dtype=bool)[:,:]&False,settings,[0,8])["entropy"])
