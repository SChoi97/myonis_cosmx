from __future__ import annotations

import numpy as np
import pandas as pd

from ..inputs.metadata import SAMPLE_COLUMNS
from .h5ad import make_adata


def combine_geometry(parts):
    points, uids, offsets, fields, cursor = [], [], [0], [], 0
    for part in parts:
        points.append(part["points"])
        uids.extend(part["object_uids"].tolist())
        fields.append(part["field_offsets"])
        offsets.extend((part["offsets"][1:]+cursor).tolist())
        cursor += len(part["points"])
    return dict(object_uids=np.asarray(uids, dtype=str),
                points=np.concatenate(points) if points else np.empty((0, 2), dtype=np.float32),
                offsets=np.asarray(offsets, dtype=np.int64),
                field_offsets=np.concatenate(fields) if fields else np.empty((0, 2), dtype=np.float32))


def subset_geometry(geometry, uids):
    parts = []
    for i, uid in enumerate(geometry["object_uids"]):
        if uid in uids:
            points = geometry["points"][geometry["offsets"][i]:geometry["offsets"][i+1]]
            parts.append(dict(points=points, offsets=np.array([0, len(points)]), object_uids=np.array([uid]), field_offsets=geometry["field_offsets"][i:i+1]))
    return combine_geometry(parts)


def _concat(frames, columns=()):
    result = pd.concat(frames, axis=0, sort=False) if frames else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in result:
            result[col] = pd.Series(index=result.index, dtype=object)
    return result


def aggregate(parts, index, config, provenance, failures):
    nuclei = _concat([p["nuclei"] for p in parts], ["object_uid", "parent_myotube_uid", *SAMPLE_COLUMNS])
    if nuclei.empty:
        for column, dtype in [("qc_pass", bool), ("qc_status", object), ("qc_exclusion_reason", object),
                              ("is_persistent", float), ("qc_is_edge", float), ("is_edge", bool), ("is_edge_geometry", bool)]:
            nuclei[column] = pd.Series(index=nuclei.index, dtype=dtype)
    myotubes = _concat([p["myotubes"] for p in parts], ["object_uid", "n_nuclei", *SAMPLE_COLUMNS])
    registry = _concat([p["registry"] for p in parts], ["object_uid"])
    names = sorted({k for p in parts for k in p["structures"]})
    structures = {key: _concat([p["structures"][key] for p in parts if key in p["structures"]],
                              ["object_uid", "parent_nucleus_uid", "parent_myotube_uid", *SAMPLE_COLUMNS]) for key in names}
    relationships = {key: _concat([p["relationships"][key] for p in parts], ["child_uid"])
                     for key in ["nuclei_to_myotubes", "structures_to_nuclei", "structures_to_myotubes", "candidates"]}
    for frame in [nuclei, myotubes, registry, *structures.values()]:
        frame.index = pd.Index(frame.object_uid.astype(str), name="object_uid")
        if not frame.index.is_unique:
            raise ValueError("Duplicate object UIDs while combining partitions")
    # Referential and count integrity are required before publishing a result.
    parents = set(nuclei.parent_myotube_uid.dropna()) - {""}
    if not parents.issubset(myotubes.index):
        raise ValueError("Nucleus references a missing myotube")
    counts = nuclei.parent_myotube_uid.value_counts()
    for uid, row in myotubes.iterrows():
        if int(row.n_nuclei) != int(counts.get(uid, 0)):
            raise ValueError(f"Resident nucleus count mismatch for {uid}")
    for table in structures.values():
        for _, row in table.iterrows():
            if row.parent_nucleus_uid:
                if row.parent_nucleus_uid not in nuclei.index:
                    raise ValueError("Structure references a missing nucleus")
                if row.parent_myotube_uid != nuclei.loc[row.parent_nucleus_uid, "parent_myotube_uid"]:
                    raise ValueError("Structure ancestry disagrees with its parent nucleus")
            elif row.parent_myotube_uid and row.parent_myotube_uid not in myotubes.index:
                raise ValueError("Structure references a missing direct myotube")
    measurements = _concat([p["qc"] for p in parts])
    images = index["images"]
    samples = images[SAMPLE_COLUMNS].drop_duplicates().reset_index(drop=True)
    for scope, frame in [("nuclei", nuclei), ("myotubes", myotubes)]:
        counts = frame.groupby("sample_id").size() if "sample_id" in frame else pd.Series(dtype=int)
        samples["n_"+scope] = samples.sample_id.map(counts).fillna(0).astype(int)
    samples["n_selected_fields"] = samples.sample_id.map(images.groupby("sample_id").field_key.nunique()).fillna(0).astype(int)
    completed_keys = {p["field_key"] for p in parts}
    completed_images = images[images.field_key.isin(completed_keys)]
    samples["n_completed_fields"] = samples.sample_id.map(completed_images.groupby("sample_id").field_key.nunique()).fillna(0).astype(int)
    samples["n_excluded_fields"] = samples.n_selected_fields-samples.n_completed_fields
    field_rows = []
    for p in parts:
        field_rows.append(dict(field_key=p["field_key"], n_nuclei=len(p["nuclei"]), n_myotubes=len(p["myotubes"]),
                               duration_seconds=p["duration_seconds"], rasterizer=p["rasterizer"]))
    fingerprints = pd.DataFrame([dict(path=k, **v) for k, v in index["fingerprints"].items()])
    nucleus_qc = index.get("nucleus_qc", {})
    if config.get("registration_qc", {}).get("enabled"):
        sources = [p["nucleus_qc"] for p in parts if "nucleus_qc" in p]
        nucleus_qc = {key: _concat([p[key] for p in sources]).reset_index(drop=True)
                      for key in ["objects", "rounds", "patches"]}
        nucleus_qc["manifest"] = dict(mode="integrated", settings_json=__import__("json").dumps(config["registration_qc"]))
    extra = dict(object_registry=registry, geometry=combine_geometry([p["geometry"] for p in parts]),
                 source_geometry=combine_geometry([p["source_geometry"] for p in parts]),
                 relationships=relationships, sample_metadata=samples,
                 metadata_mappings=dict(layout_json=__import__("json").dumps(config["plate_layout_data"]),
                                        cell_line_rule="Use physical row parsed from well-rXXcYY; legacy_cell_line_row is provenance only."),
                 inputs=dict(images=images, assets=index["assets"], fingerprints=fingerprints, field_selection=index.get("field_selection", {})),
                 qc=dict(input_validation=index["issues"], measurement_status=measurements, field_summary=pd.DataFrame(field_rows),
                         nucleus_qc=nucleus_qc,
                         visualisation_selection=index.get("visualisation_selection", {}),
                         visualisations=_concat([p["visualisations"] for p in parts]),
                         failures=pd.DataFrame(failures), excluded_fields=index.get("excluded_fields", pd.DataFrame()),
                         scope="Selected fields; object measurements cover completed fields only"),
                 provenance=provenance,
                 measurement=dict(overlap_policy=config.get("measurement", {}).get("overlap_policy", "include"),
                                  overlap_scope="Within each patch, separately for nuclei, myotubes and each structure source.",
                                  morphology="Original masks and polygons; IDs, counts and parent assignments are unchanged.",
                                  compartments="Exclude trims original compartments; removed nuclear pixels never become cytoplasm.",
                                  sigmoid="Policy applied on the native prediction grid; normalization uses retained nuclear area."),
                 nuclear_aberration_ensembles={p["field_key"]: p["aberration_provenance"] for p in parts})
    extra["registration"] = dict(enabled=config.get("registration_qc", {}).get("enabled", False),
        transforms=_concat([p["registration"] for p in parts if "registration" in p]).reset_index(drop=True),
        coordinate_frame="Reference DAPI patch" if config.get("registration_qc", {}).get("enabled") else "Preprocessing patch",
        source_geometry="Original input polygons in uns['source_geometry']; source IDs retained.",
        application="Same source-to-reference DAPI translation for every channel, contour and sigmoid map in that patch/round. "
                    "Images: linear float32 interpolation; invalid borders NaN. No DAPI brightness correction is applied to markers.")
    extra["measurement"]["morphology"] = "Original source polygons/masks. Overlap policy does not change IDs, counts or assignments; enabled registration aligns geometry before assignment."
    run = make_adata(nuclei, myotubes, structures, index["markers"], extra)
    for col in ["source_channel", "fluorophore", "cohort_id"]:
        values = images.groupby("acquisition_id")[col].agg(lambda s: ";".join(str(v) for v in sorted(s.unique())))
        run.var[col] = run.var_names.map(values).fillna("").astype(object)
    return run
