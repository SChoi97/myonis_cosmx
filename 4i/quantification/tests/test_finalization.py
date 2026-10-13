import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pypdf import PdfReader
import pytest

from fouri_quantification import pipeline
from fouri_quantification.cli import main
from fouri_quantification.finalization import check_finalizer_changes, local_checkpoint
from fouri_quantification.reporting.pdf import build_report


def test_local_copy_verifies_exact_bytes_and_cleans_up(tmp_path):
    source = tmp_path/"source.h5ad"
    source.write_bytes(b"checkpoint payload")
    scratch = tmp_path/"scratch"; scratch.mkdir()
    with local_checkpoint(source, scratch, hashlib.sha256(source.read_bytes()).hexdigest()) as local:
        assert local.parent.parent == scratch
        assert local.read_bytes() == source.read_bytes()
    assert not list(scratch.iterdir())
    with pytest.raises(ValueError, match="checksum"):
        with local_checkpoint(source, scratch, "wrong"):
            pytest.fail("corrupt checkpoint must not be opened")
    assert not list(scratch.iterdir())


def test_finalizer_upgrade_does_not_allow_measurement_changes():
    original = {"pipeline.py": "old", "features/intensity.py": "unchanged"}
    current = dict(original, **{"pipeline.py": "new", "finalization.py": "new"})
    with pytest.raises(ValueError, match="code changed"):
        check_finalizer_changes(original, current, False)
    assert check_finalizer_changes(original, current, True) == ["finalization.py", "pipeline.py"]
    current["features/intensity.py"] = "changed"
    with pytest.raises(ValueError, match="measurement code"):
        check_finalizer_changes(original, current, True)


def test_partial_archive_reports_specific_registration_exclusion(dataset, tmp_path, monkeypatch):
    directory = pipeline.prepare(dataset, tmp_path/"partial")
    pipeline.execute_task(directory, 0)
    task = json.loads((directory/"tasks.json").read_text())[1]
    (directory/"logs"/(task["task_id"]+".error.json")).write_text(json.dumps(dict(error="Registration failed")))
    folder = directory/"registration_qc"/task["task_id"]; folder.mkdir(parents=True)
    (folder/"summary.json").write_text(json.dumps(dict(failed=[dict(
        patch_key="field__patch_0002__pr01-pc02", error="Round 1: low_global_correlation")])) )
    # Strict mode must stop before opening any HDF5 payloads.
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "read_partition", lambda *a: pytest.fail("HDF5 opened before missing-field check"))
        with pytest.raises(ValueError, match="1/2 field checkpoints"):
            pipeline.finalize(directory, require_complete=True)
    result = pipeline.finalize(directory, workers=2)
    assert result["status"] == "partial" and result["pdf_status"] == "complete", result
    assert not (directory/"logs/finalize.error.json").exists()
    run = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    exclusions = run.uns["qc"]["excluded_fields"]
    assert len(exclusions) == 1
    assert exclusions.iloc[0].failed_patch_indices == "2"
    assert exclusions.iloc[0].excluded_patches == 2  # fixture has two patches per field
    assert "Round 1: low_global_correlation" in exclusions.iloc[0].reason
    assert task["field_key"] not in set(run.obs.field_key)
    assert run.uns["sample_metadata"].n_excluded_fields.sum() == 1
    assert len(pd.read_csv(directory/"excluded_fields.csv")) == 1
    pdf = PdfReader(directory/"run_summary.pdf")
    assert len(pdf.pages) == 4
    text = pdf.pages[-1].extract_text()
    assert "low_global_correlation" in text and "K32Del" in text
    assert "4 / 4" in text
    # Many exclusions must all appear, with no page overflow or silent truncation.
    many = pd.concat([exclusions]*35, ignore_index=True)
    many["field_id"] = np.arange(1, 36)
    run.uns["qc"]["excluded_fields"] = many
    with monkeypatch.context() as patch:
        patch.setattr("fouri_quantification.reporting.pdf.ad.read_h5ad", lambda *a: pytest.fail("PDF reloaded full archive"))
        build_report("unused", tmp_path/"many-exclusions.pdf", run=run)
    text = "\n".join(p.extract_text() for p in PdfReader(tmp_path/"many-exclusions.pdf").pages[3:])
    assert text.count("low_global_correlation") == 35


def test_parallel_finalization_preserves_exact_measurements_and_old_provenance(dataset, tmp_path):
    directory, _ = pipeline.run_pipeline(dataset, tmp_path/"run")
    expected = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    manifest = json.loads((directory/"run_manifest.json").read_text())
    manifest["code_hashes"]["pipeline.py"] = "old-finalizer"
    (directory/"run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="code changed"):
        pipeline.finalize(directory)
    result = pipeline.finalize(directory, workers=2, allow_finalizer_update=True)
    assert result["status"] == "complete"
    assert result["code_hashes"]["pipeline.py"] == "old-finalizer"
    assert result["finalization"]["changed_code_files"] == ["pipeline.py"]
    actual = ad.read_h5ad(directory/"myonis_4i_data.h5ad")
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    np.testing.assert_array_equal(actual.X, expected.X)
    for name in actual.obsm:
        if isinstance(actual.obsm[name], pd.DataFrame):
            pd.testing.assert_frame_equal(actual.obsm[name], expected.obsm[name])
        else:
            np.testing.assert_array_equal(actual.obsm[name], expected.obsm[name])
    pd.testing.assert_frame_equal(actual.uns["myotubes"], expected.uns["myotubes"])
    for name in actual.uns["structures"]:
        pd.testing.assert_frame_equal(actual.uns["structures"][name], expected.uns["structures"][name])
    for name in actual.uns["relationships"]:
        pd.testing.assert_frame_equal(actual.uns["relationships"][name], expected.uns["relationships"][name])


def test_no_valid_fields_refuses_empty_archive(dataset, tmp_path):
    directory = pipeline.prepare(dataset, tmp_path/"empty")
    with pytest.raises(ValueError, match="2/2 field checkpoints"):
        pipeline.finalize(directory)
    assert not (directory/"myonis_4i_data.h5ad").exists()


@pytest.mark.parametrize("allow,expected", [(False, 1), (True, 0)])
def test_explicit_partial_finalization_exit_status(monkeypatch, allow, expected):
    monkeypatch.setattr(pipeline, "finalize", lambda *a, **kw: dict(status="partial"))
    args = ["finalize", "--run-dir", "unused"] + (["--allow-partial"] if allow else [])
    assert main(args) == expected


def test_reader_does_not_treat_scratch_exhaustion_as_field_exclusion(tmp_path, monkeypatch):
    import errno
    from contextlib import contextmanager
    from fouri_quantification import finalization
    (tmp_path/"checkpoints").mkdir()
    (tmp_path/"checkpoints/one.done.json").write_text(json.dumps(dict(fingerprint="same", sha256="unused")))
    @contextmanager
    def full_disk(*args):
        raise OSError(errno.ENOSPC, "Scratch disk is full")
        yield  # contextmanager protocol
    monkeypatch.setattr(finalization, "local_checkpoint", full_disk)
    with pytest.raises(OSError, match="Scratch disk is full"):
        finalization._read_one((tmp_path, dict(task_id="one", field_key="field"), "same", None))


def test_single_very_long_exclusion_reason_is_fully_paginated():
    from fouri_quantification.reporting.pdf import exclusion_pages
    reason = "Registration detail. " * 2000
    rows = pd.DataFrame([dict(plate_name="plate", repeat=6, cell_line="L302P-Corr", well_id="r07c11",
                              field_id=11, cohort_id="m01_08", excluded_patches=4, reason=reason)])
    pages = exclusion_pages(rows)
    assert len(pages) > 3
    for page in pages:
        assert sum(block.wrap(510, 100000)[1]+14 for block in page) <= 570
