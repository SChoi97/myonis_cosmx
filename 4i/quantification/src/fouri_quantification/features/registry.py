"""Definitions shared by the H5AD schema and report."""
from __future__ import annotations

import re

MORPHOLOGY_COLUMNS = ["area_px2", "perimeter_px", "circularity", "major_axis_length_px"]
STAT_DEFINITIONS = {
    "mean": "Arithmetic mean of finite pixels; missing if any masked pixel is nonfinite.",
    "median": "Median of masked pixels; missing if any masked pixel is nonfinite.",
    "integrated": "Sum of masked pixel intensities; no background subtraction or exposure normalization.",
    "std": "Population standard deviation (ddof=0).",
    "p10": "10th percentile of masked pixels.", "p90": "90th percentile of masked pixels.",
    "p95": "95th percentile of masked pixels.", "min": "Minimum masked intensity.", "max": "Maximum masked intensity.",
    "n_pixels": "Number of pixels in the measurement compartment.",
    "n_valid_pixels": "Number of finite intensity pixels in the compartment.",
    "valid_fraction": "Finite pixel count divided by compartment pixel count.",
    "count": "Number of structures assigned by maximum child-area overlap strictly above threshold.",
    "count_total": "Unique assigned nuclear plus directly assigned cytoplasmic structures.",
    "count_per_nucleus": "Nuclear structure count divided by number of resident nuclei, including assessed zero-count nuclei.",
    "n_assessed_nuclei": "Resident nuclei with an available structure segmentation, including valid empty results.",
    "fraction_nuclei_positive": "Resident nuclei with at least one assigned structure divided by assessed resident nuclei.",
    "area_mean_px2": "Mean original polygon area of assigned structures, not clipped to the parent.",
    "area_median_px2": "Median original polygon area of assigned structures.",
    "area_max_px2": "Largest original polygon area of assigned structures.",
    "area_std_px2": "Population standard deviation of original assigned-structure polygon areas.",
    "mean_object_intensity": "Equal-object average of within-parent structure mean intensities; missing if any object mean is missing.",
    "std_object_intensity": "Population standard deviation of within-parent structure mean intensities.",
    "mean_structure_pixel_intensity": "Mean over the union of assigned structure measurement pixels clipped to the parent measurement mask; overlap policy applied first.",
    "integrated_structure_intensity": "Intensity sum over the union of structure measurement pixels inside the parent measurement mask; overlap policy applied first.",
    "occupied_area_px2": "Raster area of the union of assigned structure measurement masks intersected with the parent measurement mask.",
    "coverage_fraction": "Union occupied measurement area divided by retained parent measurement area.",
    "ratio": "Whole-nucleus mean divided by local perinuclear mean, only above the configured denominator floor.",
    "entropy": "Base-2 entropy of masked intensities after fixed acquisition-wide quantization.",
}
LEGACY_DEFINITIONS = {
    "overlap_area_px2": "Original raster pixels shared with at least one peer object on the same patch, within the same object type or structure source.",
    "overlap_fraction": "Shared raster area divided by original raster area, irrespective of the selected measurement policy.",
    "measurement_area_px2": "Whole-object raster area after applying measurement.overlap_policy on the quantitative image grid.",
    "measurement_empty": "True when the whole-object measurement mask has no retained pixels; the object and its assignments are retained.",
    "nuclear_aberration_measurement_area_px2": "Retained whole-object raster area on the native sigmoid prediction grid; saved for overlap exclusion or registered prediction measurement. NaN if the prediction is missing.",
    "area_px2": "Existing CosMx polygon area from the absolute shoelace formula.",
    "perimeter_px": "Existing CosMx sum of closed polygon edge lengths.",
    "circularity": "Existing CosMx 4*pi*area/perimeter^2 clipped to [0,1].",
    "major_axis_length_px": "Existing CosMx maximum pairwise polygon-vertex distance; not ellipse major axis.",
    "n_nuclei": "Number of nuclei assigned with the unchanged CosMx overlap rule.",
    "assigned_nuclei_area_px2": "Sum of original polygon areas of assigned nuclei.",
    "area_per_nucleus_px2": "Myotube polygon area divided by assigned nucleus count; undefined when count is zero.",
    "nuclei_density_per_1000px2": "Assigned nucleus count per 1000 square pixels of myotube polygon area.",
    "nuclear_area_fraction": "Union of assigned nuclear raster masks intersected with myotube, divided by myotube raster area.",
    "nuclear_nearest_neighbour_mean_px": "Mean nearest-neighbour centroid distance among resident nuclei; missing for fewer than two.",
    "intensity_mean_in_parent": "Mean intensity in the retained structure-mask intersection with its assigned nuclear measurement mask, or myotube cytoplasm for a direct link.",
    "area_in_parent_px2": "Retained pixel count of the structure-mask intersection with its assigned nuclear measurement mask, or myotube cytoplasm for a direct link.",
    "nuclear_aberration_mean_sigmoid": "Sigmoid sum divided by retained nuclear prediction-grid area; include preserves the existing CosMx mask, exclude removes peer overlap.",
    "nuclear_aberration_total_sigmoid": "Sum of sigmoid pixels in the nuclear measurement mask under the configured overlap policy.",
    "nuclear_aberration_myotube_total_sigmoid": "Sum of sigmoid pixels in the whole-myotube measurement mask under the configured overlap policy.",
    "nuclear_aberration_myotube_mean_sigmoid": "Mean sigmoid in the whole-myotube measurement mask under the configured overlap policy.",
    "nuclear_aberration_assigned_nuclei_total_sigmoid": "Existing sum of resident-nucleus sigmoid totals; NaN if any assigned nucleus lacks prediction; zero for no nuclei.",
    "nuclear_aberration_assigned_nuclei_total_per_nuclear_area": "Resident-nucleus sigmoid sum divided by summed retained resident prediction-grid nuclear areas under the configured overlap policy.",
    "nuclear_aberration_myotube_total_per_nuclear_area": "Whole-myotube sigmoid sum divided by summed retained resident prediction-grid nuclear areas under the configured overlap policy.",
}


def feature_group(column):
    if column in MORPHOLOGY_COLUMNS:
        return "morphology_features"
    if column.startswith("geometry__"):
        return "additional_morphology_features"
    if "__structures__" in column:
        return "structure_features"
    if "__" in column and not column.startswith("resident_nuclei__"):
        return "texture_features" if "__glcm_" in column or column.endswith("__entropy") else "intensity_features"
    return None


def describe(column, scope, location):
    if column not in LEGACY_DEFINITIONS and "__" not in column:
        return None
    parts = column.split("__")
    marker = parts[0] if len(parts) >= 3 else ""
    compartment = parts[1] if len(parts) >= 3 else scope
    stat = parts[-1]
    formula = LEGACY_DEFINITIONS.get(column, STAT_DEFINITIONS.get(stat, ""))
    if column.startswith("resident_nuclei__"):
        marker, compartment = parts[1], "resident_nuclei/"+parts[2]
        formula = f"{stat} across finite resident-nucleus values of {'__'.join(parts[1:-1])}; n_valid records the denominator."
    if stat.startswith("glcm_"):
        formula = "Symmetric GLCM pooled over horizontal, vertical and both diagonal offsets; only pairs with both pixels in the mask. "
        if "contrast" in stat: formula += "Sum P(i,j)*(i-j)^2."
        elif "homogeneity" in stat: formula += "Sum P(i,j)/(1+(i-j)^2)."
        elif "energy" in stat: formula += "sqrt(sum P(i,j)^2)."
        elif "correlation" in stat: formula += "Normalized covariance; undefined when either marginal variance is zero."
        else: formula += "Count of valid directed pairs before symmetrization."
    if not formula and column.startswith("geometry__"):
        formula = {"eccentricity": "Region eccentricity from second moments.", "solidity": "Raster area / convex hull area.",
                   "extent": "Raster area / bounding box area.", "orientation": "Second-moment ellipse orientation in radians.",
                   "axis_major_length": "Major axis of the second-moment ellipse in pixels.",
                   "axis_minor_length": "Minor axis of the second-moment ellipse in pixels.",
                   "raster_area_px2": "Number of rasterized object pixels.", "raster_area_um2": "Raster area times squared pixel size.",
                   "ellipse_major_axis_um": "Ellipse major axis times pixel size.", "ellipse_minor_axis_um": "Ellipse minor axis times pixel size.",
                   "skeleton_pixels": "Count of skeleton pixels; this is not a calibrated skeleton length.",
                   "skeleton_width_median_px": "Median of twice the distance to exterior at skeleton pixels.",
                   "skeleton_width_median_um": "Median skeleton width in pixels times pixel size."}.get(stat, stat)
    units = "dimensionless"
    if column.startswith("resident_nuclei__"):
        base = describe("__".join(parts[1:-1]), "nucleus", location)
        units = "count" if stat == "n_valid" else base["units"]
    elif column == "nuclei_density_per_1000px2": units = "nuclei / 1000 px^2"
    elif "px2" in column: units = "px^2"
    elif "um2" in column: units = "um^2"
    elif stat.endswith("_um"): units = "um"
    elif stat.endswith("_px") or stat.startswith("axis_"): units = "px"
    elif stat == "entropy": units = "bits"
    elif stat == "orientation": units = "radians"
    elif "integrated" in stat: units = "image intensity units * pixels"
    elif "intensity" in stat or stat in {"mean", "median", "std", "p10", "p90", "p95", "min", "max"}: units = "image intensity units"
    elif "count" in stat or stat in {"n_pixels", "n_valid_pixels", "n_assessed_nuclei", "n_valid", "skeleton_pixels"}: units = "count"
    elif "sigmoid" in column: units = "sigmoid score or score sum (see formula)"
    return dict(feature=column, object_level=scope, marker=marker, compartment=compartment,
                statistic=stat, units=units, definition=formula, storage_location=location,
                missing_policy="NaN means unavailable or undefined; count zero requires an available segmentation.")
