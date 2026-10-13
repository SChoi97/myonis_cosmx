from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from ..compatibility._geometry import compute_morphology_features


def legacy_morphology(objects):
    values, names = compute_morphology_features([o["local_polygon"] for o in objects])
    return [dict(zip(names, row.tolist())) for row in values]


def additional_morphology(mask, pixel_size=None, myotube=False):
    result = {}
    if not mask.any():
        return result
    yy, xx = np.nonzero(mask)
    crop = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    props = regionprops(crop.astype(np.uint8))[0]
    for key in ["eccentricity", "solidity", "extent", "orientation", "axis_major_length", "axis_minor_length"]:
        result[key] = float(getattr(props, key))
    result["raster_area_px2"] = int(mask.sum())
    if pixel_size:
        result["raster_area_um2"] = mask.sum()*pixel_size**2
        result["ellipse_major_axis_um"] = props.axis_major_length*pixel_size
        result["ellipse_minor_axis_um"] = props.axis_minor_length*pixel_size
    if myotube:
        padded = np.pad(crop, 1)
        skeleton = skeletonize(padded)
        widths = 2*distance_transform_edt(padded)[skeleton]
        result["skeleton_pixels"] = int(skeleton.sum())
        result["skeleton_width_median_px"] = float(np.median(widths)) if widths.size else np.nan
        if pixel_size:
            result["skeleton_width_median_um"] = result["skeleton_width_median_px"]*pixel_size
    return result
