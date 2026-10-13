"""Create a small, standard H5AD containing nucleus and myotube measurements."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import uuid

import h5py
from anndata.io import read_elem, write_elem
import numpy as np


ROOT_KEYS = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers")
UNS_KEYS = ("myotubes", "morphology_feature_columns", "sample_metadata",
            "metadata_mappings", "provenance", "measurement", "schema_version", "X_definition")
QC_KEYS = ("field_summary", "excluded_fields", "failures", "scope")


def _repeat_number(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _object_id(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _repeat_indices(frame: h5py.Group, repeat: int, label: str) -> np.ndarray:
    if "repeat" not in frame:
        raise ValueError(f"{label} has no repeat column")
    values = np.asarray(read_elem(frame["repeat"]))
    return np.flatnonzero(np.fromiter(
        (_repeat_number(value) == repeat for value in values), dtype=bool, count=len(values)))


def _copy_selected(parent: h5py.Group, name: str, target: h5py.Group,
                   indices: np.ndarray, row_count: int) -> None:
    """Copy one H5AD element, selecting rows without reading whole feature tables."""
    source = parent[name]
    if isinstance(source, h5py.Dataset):
        if not source.shape or source.shape[0] != row_count:
            parent.copy(name, target)
            return
        dest = target.create_dataset(name, shape=(len(indices), *source.shape[1:]),
                                     dtype=source.dtype, compression=source.compression,
                                     compression_opts=source.compression_opts,
                                     shuffle=source.shuffle, fletcher32=source.fletcher32)
        dest.attrs.update(source.attrs)
        for start in range(0, row_count, 8192):
            stop = min(start + 8192, row_count)
            left, right = np.searchsorted(indices, [start, stop])
            if right > left:
                dest[left:right] = source[start:stop][indices[left:right] - start]
        return
    copied = target.create_group(name)
    copied.attrs.update(source.attrs)
    for child in source:
        if source.attrs.get("encoding-type") == "categorical" and child == "categories":
            source.copy(child, copied)
        else:
            _copy_selected(source, child, copied, indices, row_count)


def _copy_root(original: h5py.File, slim: h5py.File,
               nucleus_indices: np.ndarray | None) -> None:
    total = len(original["obs"][original["obs"].attrs["_index"]])
    for key in ROOT_KEYS:
        if key not in original:
            continue
        if nucleus_indices is not None and key in {"X", "obs", "obsm", "layers"}:
            _copy_selected(original, key, slim, nucleus_indices, total)
        elif nucleus_indices is not None and key == "obsp" and len(original[key]):
            raise ValueError("Replicate export does not support a nonempty obsp matrix")
        else:
            original.copy(key, slim)


def _filter_table(frame, repeat: int, label: str):
    if "repeat" in frame:
        values = frame["repeat"]
    elif "field_key" in frame:
        values = frame["field_key"].astype(str).str.split("|").str[3]
    elif frame.index.name == "field_key":
        values = frame.index.astype(str).str.split("|").str[3]
    elif frame.empty:
        return frame
    else:
        print(f"Measurements H5AD: omitting {label}; rows cannot be assigned to a repeat", flush=True)
        return None
    mask = np.fromiter((_repeat_number(value) == repeat for value in values),
                       dtype=bool, count=len(frame))
    return frame.iloc[np.flatnonzero(mask)].copy()


def _copy_uns_table(parent: h5py.Group, key: str, target: h5py.Group,
                    repeat: int | None) -> None:
    if repeat is None:
        parent.copy(key, target)
    else:
        frame = _filter_table(read_elem(parent[key]), repeat, key)
        if frame is not None:
            write_elem(target, key, frame, dataset_kwargs={"compression": "gzip"})


def _copy_qc(source_uns: h5py.Group, target_uns: h5py.Group,
             repeat: int | None) -> None:
    if "qc" not in source_uns:
        return
    source_qc = source_uns["qc"]
    target_qc = target_uns.create_group("qc")
    target_qc.attrs.update(source_qc.attrs)
    for key in QC_KEYS:
        if key in source_qc:
            if key == "scope" or repeat is None:
                source_qc.copy(key, target_qc)
            else:
                _copy_uns_table(source_qc, key, target_qc, repeat)


def _selection(original: h5py.File, repeat: int | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if repeat is None:
        return None, None
    if repeat < 1:
        raise ValueError("--repeat must be a positive integer")
    if "uns" not in original or "myotubes" not in original["uns"]:
        raise ValueError("Input is not a full MYONIS 4i H5AD: missing uns/myotubes")
    nuclei = _repeat_indices(original["obs"], repeat, "Nucleus table")
    myotubes = _repeat_indices(original["uns/myotubes"], repeat, "Myotube table")
    if not len(nuclei) and not len(myotubes):
        raise ValueError(f"Repeat {repeat} has no nuclei or myotubes in the input H5AD")
    if "parent_myotube_uid" in original["obs"]:
        parents = {_object_id(value) for value in
                   np.asarray(read_elem(original["obs/parent_myotube_uid"]))[nuclei]}
        myotube_frame = original["uns/myotubes"]
        index_name = myotube_frame.attrs["_index"]
        tube_ids = {_object_id(value) for value in
                    np.asarray(read_elem(myotube_frame[index_name]))[myotubes]}
        missing = parents - {""} - tube_ids
        if missing:
            raise ValueError(f"Repeat {repeat} has {len(missing)} parent myotube IDs outside the selected replicate")
    print(f"Measurements H5AD: repeat {repeat}: {len(nuclei):,} nuclei, "
          f"{len(myotubes):,} myotubes", flush=True)
    return nuclei, myotubes


def default_measurements_path(source: str | Path, repeat: int | None = None) -> Path:
    source = Path(source)
    suffix = f"_rep_{repeat}" if repeat is not None else ""
    return source.with_name(f"{source.stem}_measurements{suffix}{source.suffix}")


def _write_local(source: Path, local: Path, repeat: int | None = None) -> None:
    with h5py.File(source, "r") as original, h5py.File(local, "w") as slim:
        if "uns" not in original or "myotubes" not in original["uns"]:
            raise ValueError("Input is not a full MYONIS 4i H5AD: missing uns/myotubes")
        for key in ("X", "obs", "var", "obsm"):
            if key not in original:
                raise ValueError(f"Input H5AD is missing {key}")
        nuclei, myotubes = _selection(original, repeat)
        slim.attrs.update(original.attrs)
        print("Measurements H5AD: copying nucleus measurements...", flush=True)
        _copy_root(original, slim, nuclei)
        source_uns = original["uns"]
        target_uns = slim.create_group("uns")
        target_uns.attrs.update(source_uns.attrs)
        for key in UNS_KEYS:
            if key in source_uns:
                if key == "myotubes":
                    print("Measurements H5AD: copying myotube measurements...", flush=True)
                    if myotubes is not None:
                        _copy_selected(source_uns, key, target_uns, myotubes,
                                       len(source_uns[key]["object_uid"]))
                    else:
                        source_uns.copy(key, target_uns)
                elif key == "sample_metadata":
                    _copy_uns_table(source_uns, key, target_uns, repeat)
                else:
                    source_uns.copy(key, target_uns)
        if "feature_dictionary" in source_uns:
            dictionary = read_elem(source_uns["feature_dictionary"])
            dictionary = dictionary.loc[
                ~dictionary.storage_location.str.startswith('uns["structures"]')
            ].reset_index(drop=True)
            dictionary.index.name = "row_id"
            write_elem(target_uns, "feature_dictionary", dictionary, dataset_kwargs={"compression": "gzip"})
        _copy_qc(source_uns, target_uns, repeat)
        write_elem(target_uns, "measurements_view", {
            "source_file": source.name,
            "scope": "Nucleus and myotube measurements; all original objects and saved aggregates.",
            "omitted": "Individual structure tables, contours, object registry, relationship audits, per-round QC evidence and input manifests.",
            "qc_note": "Myotube resident-nucleus aggregates include all assigned reference nuclei; filtering obs.qc_pass does not recalculate them.",
            "repeat": repeat if repeat is not None else "all",
        })
        if slim["X"].shape[0] != (len(nuclei) if nuclei is not None else original["X"].shape[0]):
            raise RuntimeError("Nucleus matrix shape changed during export")
        if len(slim["uns/myotubes"]["object_uid"]) != (len(myotubes) if myotubes is not None else len(source_uns["myotubes"]["object_uid"])):
            raise RuntimeError("Myotube count changed during export")
        if "structures" in target_uns or "geometry" in target_uns:
            raise RuntimeError("Full-resolution payload leaked into measurements export")


def make_measurements_h5ad(source: str | Path, output: str | Path | None = None,
                           repeat: int | None = None) -> Path:
    """Copy analysis tables via local scratch, then publish beside the full archive.

    Existing nucleus and myotube values are copied without recomputation. The
    original archive is never changed; the output is replaced atomically only
    after a structural check succeeds.
    """
    source = Path(source)
    output = Path(output) if output is not None else default_measurements_path(source, repeat)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("Measurements output must differ from the full H5AD")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        with tempfile.TemporaryDirectory(prefix="myonis-4i-measurements-", dir=os.environ.get("SLURM_TMPDIR")) as scratch:
            local = Path(scratch) / "measurements.h5ad"
            _write_local(source, local, repeat)
            print("Measurements H5AD: copying completed file to results...", flush=True)
            shutil.copyfile(local, temporary)
        os.replace(temporary, output)
        print(f"Measurements H5AD: complete ({output.stat().st_size / 2**30:.2f} GiB): {output}", flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def default_split_paths(source: str | Path, repeat: int | None = None) -> tuple[Path, Path]:
    source = Path(source)
    suffix = f"_rep_{repeat}" if repeat is not None else ""
    return (source.with_name(f"{source.stem}_nuclei_measurements{suffix}{source.suffix}"),
            source.with_name(f"{source.stem}_myotubes_measurements{suffix}{source.suffix}"))


def _empty_mapping(slim: h5py.File, original: h5py.File, name: str) -> None:
    group = slim.create_group(name)
    if name in original:
        group.attrs.update(original[name].attrs)
    else:
        group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})


def _split_uns(original: h5py.File, slim: h5py.File, scope: str,
               kept_myotube_columns: set[str] | None = None,
               repeat: int | None = None) -> None:
    source_uns = original["uns"]
    target_uns = slim.create_group("uns")
    target_uns.attrs.update(source_uns.attrs)
    for key in UNS_KEYS:
        if key not in {"myotubes", "morphology_feature_columns", "X_definition"} and key in source_uns:
            if key == "sample_metadata":
                _copy_uns_table(source_uns, key, target_uns, repeat)
            else:
                source_uns.copy(key, target_uns)
    if scope == "nuclei":
        for key in ("morphology_feature_columns", "X_definition"):
            if key in source_uns:
                source_uns.copy(key, target_uns)
    else:
        write_elem(target_uns, "X_definition", "Whole-myotube mean intensity per acquisition.")
    if "feature_dictionary" in source_uns:
        dictionary = read_elem(source_uns["feature_dictionary"])
        if scope == "nuclei":
            dictionary = dictionary.loc[dictionary.storage_location.eq("obs") |
                dictionary.storage_location.str.startswith("obsm[")].copy()
        else:
            dictionary = dictionary.loc[dictionary.storage_location.eq('uns["myotubes"]')].copy()
            if kept_myotube_columns is not None:
                dictionary = dictionary.loc[dictionary.feature.isin(kept_myotube_columns)]
            dictionary["storage_location"] = "obs"
        dictionary = dictionary.reset_index(drop=True)
        dictionary.index.name = "row_id"
        write_elem(target_uns, "feature_dictionary", dictionary, dataset_kwargs={"compression": "gzip"})
    _copy_qc(source_uns, target_uns, repeat)
    write_elem(target_uns, "measurements_view", {
        "source_file": Path(original.filename).name,
        "scope": f"{scope.capitalize()} measurements; one row per original {'nucleus' if scope == 'nuclei' else 'myotube'} object.",
        "join_key": "Nucleus obs.parent_myotube_uid matches myotube obs.index; an empty parent ID means unassigned.",
        "qc_note": "Myotube resident-nucleus aggregates include all assigned reference nuclei; filtering qc_pass does not recalculate them.",
        "repeat": repeat if repeat is not None else "all",
    })


def _write_nuclei_local(source: Path, local: Path, repeat: int | None = None) -> None:
    with h5py.File(source, "r") as original, h5py.File(local, "w") as slim:
        nuclei, _ = _selection(original, repeat)
        slim.attrs.update(original.attrs)
        _copy_root(original, slim, nuclei)
        _split_uns(original, slim, "nuclei", repeat=repeat)
        if slim["X"].shape[0] != (len(nuclei) if nuclei is not None else original["X"].shape[0]):
            raise RuntimeError("Nucleus matrix shape changed during split export")


def _write_myotubes_local(source: Path, local: Path, compact: bool,
                          repeat: int | None = None) -> None:
    with h5py.File(source, "r") as original, h5py.File(local, "w") as slim:
        source_myotubes = original["uns/myotubes"]
        _, myotubes = _selection(original, repeat)
        count = len(myotubes) if myotubes is not None else len(source_myotubes["object_uid"])
        slim.attrs.update(original.attrs)
        original.copy("var", slim)
        for key in ("obsm", "varm", "obsp", "varp", "layers"):
            _empty_mapping(slim, original, key)
        kept = None
        if compact:
            kept = {str(name) for name in source_myotubes.attrs["column-order"]
                    if name not in {"source_path", "image_name"} and not
                    (name.startswith("resident_nuclei__") and name.rsplit("__", 1)[-1] in {"median", "std"})}
            kept.add(str(source_myotubes.attrs["_index"]))
            obs = slim.create_group("obs")
            obs.attrs.update(source_myotubes.attrs)
            obs.attrs["column-order"] = np.asarray(
                [str(name) for name in source_myotubes.attrs["column-order"] if name in kept],
                dtype=h5py.string_dtype(encoding="utf-8"))
            for name in source_myotubes:
                if name in kept:
                    if myotubes is None:
                        source_myotubes.copy(name, obs)
                    else:
                        _copy_selected(source_myotubes, name, obs, myotubes,
                                       len(source_myotubes["object_uid"]))
        else:
            if myotubes is None:
                original.copy("uns/myotubes", slim, name="obs")
            else:
                _copy_selected(original["uns"], "myotubes", slim, myotubes,
                               len(source_myotubes["object_uid"]))
                slim.move("myotubes", "obs")
        markers = read_elem(original["var"])["marker"].astype(str)
        matrix = np.full((count, len(markers)), np.nan, dtype=np.float32)
        for j, marker in enumerate(markers):
            column = f"{marker}__whole_myotube__mean"
            if column in source_myotubes:
                values = source_myotubes[column]
                matrix[:, j] = (values[:] if myotubes is None else values[myotubes]).astype(np.float32)
        write_elem(slim, "X", matrix, dataset_kwargs={"compression": "gzip"})
        _split_uns(original, slim, "myotubes", kept, repeat)
        if len(slim["obs/object_uid"]) != matrix.shape[0]:
            raise RuntimeError("Myotube count changed during split export")


def make_split_measurements_h5ad(source: str | Path, compact_myotubes: bool = False,
                                 repeat: int | None = None) -> tuple[Path, Path]:
    """Write separate nucleus/myotube H5ADs; compact mode drops redundant myotube summaries."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    nucleus_output, myotube_output = default_split_paths(source, repeat)
    temporary_paths = [p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp.h5ad")
                       for p in (nucleus_output, myotube_output)]
    try:
        with tempfile.TemporaryDirectory(prefix="myonis-4i-split-", dir=os.environ.get("SLURM_TMPDIR")) as scratch:
            local_nuclei = Path(scratch) / "nuclei.h5ad"
            local_myotubes = Path(scratch) / "myotubes.h5ad"
            print("Measurements H5AD: copying nucleus measurements...", flush=True)
            _write_nuclei_local(source, local_nuclei, repeat)
            print("Measurements H5AD: copying myotube measurements...", flush=True)
            _write_myotubes_local(source, local_myotubes, compact_myotubes, repeat)
            for local, temporary in zip((local_nuclei, local_myotubes), temporary_paths):
                shutil.copyfile(local, temporary)
        for temporary, output in zip(temporary_paths, (nucleus_output, myotube_output)):
            os.replace(temporary, output)
            print(f"Measurements H5AD: complete ({output.stat().st_size / 2**30:.2f} GiB): {output}", flush=True)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
    return nucleus_output, myotube_output
