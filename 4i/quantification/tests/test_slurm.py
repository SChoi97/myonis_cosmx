import json
import shlex
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fouri_quantification import pipeline, slurm
from fouri_quantification.cli import main
from test_registration import registered_dataset


@pytest.mark.parametrize("fields,jobs", [(1, 1), (2, 8), (50, 4), (101, 7), (0, 1)])
def test_batches_cover_every_field_exactly_once(fields, jobs):
    groups = [slurm.batch_indices(fields, i, jobs) for i in range(jobs)]
    assert sorted(i for group in groups for i in group) == list(range(fields))
    assert max(map(len, groups))-min(map(len, groups)) <= 1


@pytest.mark.parametrize("index,count", [(-1, 4), (4, 4), (0, 0)])
def test_invalid_batch_indices_fail(index, count):
    with pytest.raises(ValueError, match="batch-count"):
        slurm.batch_indices(10, index, count)


@pytest.fixture
def fake_scheduler(tmp_path, monkeypatch):
    folder = tmp_path/"run with spaces and 'quotes'"
    folder.mkdir()
    (folder/"tasks.json").write_text(json.dumps([dict(task_id=str(i)) for i in range(5)]))
    (folder/"run_manifest.json").write_text(json.dumps(dict(worker_count=2)))
    calls, preparations = [], []
    def prepare(*args, **kwargs):
        preparations.append((args, kwargs))
        return folder
    def submit(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=f"{100+len(calls)};nemo\n", stderr="")
    monkeypatch.setattr(pipeline, "prepare", prepare)
    monkeypatch.setattr(slurm.shutil, "which", lambda _: "/mock/sbatch")
    monkeypatch.setattr(slurm.subprocess, "run", submit)
    monkeypatch.setenv("SLURM_JOB_ID", "90")
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(tmp_path/"submission folder"))
    return folder, calls, preparations


def test_submission_resources_dependencies_environment_and_paths(fake_scheduler):
    folder, calls, preparations = fake_scheduler
    result = slurm.submit_batch("dataset.yaml", jobs=3, workers=2, memory="16G", finalize_memory="48G",
                                troubleshoot=True, visualisation_fields=5)
    assert len(preparations) == 1
    assert preparations[0][1]["troubleshoot"] is True
    assert preparations[0][1]["visualisation_fields"] == 5
    assert len(calls) == 2
    worker, final = calls
    assert {"--array=0-2%3", "--cpus-per-task=2", "--mem=16G", "--export=ALL"} <= set(worker[0])
    assert {"--dependency=afterany:101", "--cpus-per-task=4", "--mem=48G"} <= set(final[0])
    log_dir = (folder.parent/"submission folder").resolve()
    assert log_dir.is_dir()
    for args, pattern in [(worker[0], "myonis_4i_batch-%A_%a.out"), (final[0], "myonis_4i_finalize-%j.out")]:
        assert f"--output={log_dir/pattern}" in args
        assert f"--error={log_dir/pattern}" in args
    assert result["log_dir"] == str(log_dir)
    assert result["worker_logs"] == [str(log_dir/f"myonis_4i_batch-101_{i}.out") for i in range(3)]
    assert result["finalize_log"] == str(log_dir/"myonis_4i_finalize-102.out")
    command = shlex.split(worker[1]["input"].splitlines()[-1])
    assert command[command.index("--run-dir")+1] == str(folder)
    assert command[command.index("--batch-count")+1] == "3"
    assert command[command.index("--batch-index")+1] == "${SLURM_ARRAY_TASK_ID:?}"
    assert "--allow-partial" in final[1]["input"]
    assert "--workers 4" in final[1]["input"]
    assert "OPENBLAS_NUM_THREADS=1" in worker[1]["input"]
    assert result["worker_array_job_id"] == "101" and result["finalize_job_id"] == "102"
    assert result["fields_per_job"] == [2, 2, 1]
    assert json.loads((folder/"batch_submission.json").read_text()) == result
    manifest = json.loads((folder/"run_manifest.json").read_text())
    assert manifest["worker_count"] == 5
    assert manifest["batch_execution"]["jobs"] == 3


def test_submission_does_not_create_empty_jobs(fake_scheduler):
    _, calls, _ = fake_scheduler
    receipt = slurm.submit_batch("dataset.yaml", jobs=20, workers=2)
    assert receipt["jobs"] == 5
    assert "--array=0-4%5" in calls[0][0]


@pytest.mark.parametrize("mode", ["absolute", "relative", "working_directory"])
def test_log_directory_override_and_local_fallback(fake_scheduler, monkeypatch, mode):
    folder, calls, _ = fake_scheduler
    if mode == "absolute":
        chosen = str(folder.parent/"explicit logs")
        expected = folder.parent/"explicit logs"
    elif mode == "relative":
        chosen = "relative logs"
        expected = folder.parent/"submission folder"/"relative logs"
    else:
        chosen = None
        expected = folder.parent
        monkeypatch.delenv("SLURM_SUBMIT_DIR")
        monkeypatch.chdir(expected)
    receipt = slurm.submit_batch("dataset.yaml", jobs=2, workers=2, log_dir=chosen)
    assert receipt["log_dir"] == str(expected.resolve())
    assert f"--output={expected.resolve()}/myonis_4i_batch-%A_%a.out" in calls[0][0]


def test_final_submission_failure_retains_worker_id(fake_scheduler, monkeypatch):
    folder, _, _ = fake_scheduler
    responses = iter(["101", None])
    def submit(*args):
        value = next(responses)
        if value is None:
            raise RuntimeError("Scheduler unavailable")
        return value
    monkeypatch.setattr(slurm, "_submit", submit)
    with pytest.raises(RuntimeError, match="recorded job IDs"):
        slurm.submit_batch("dataset.yaml", jobs=3, workers=2)
    result = json.loads((folder/"batch_submission.json").read_text())
    assert result["status"] == "submission_failed" and result["worker_array_job_id"] == "101"
    assert "finalize_job_id" not in result


def test_failed_batch_returns_nonzero_cli_exit(monkeypatch):
    monkeypatch.setattr(slurm, "run_batch", lambda *args: dict(complete=1, resumed=0, failed=1))
    assert main(["worker-batch", "--run-dir", "unused", "--batch-index", "0",
                 "--batch-count", "1", "--workers", "2"]) == 1


def test_batch_recovery_rejects_changed_partition(fake_scheduler):
    folder, _, _ = fake_scheduler
    (folder/"run_manifest.json").write_text(json.dumps(dict(batch_execution=dict(jobs=3))))
    with pytest.raises(ValueError, match="split into 3 batches"):
        slurm.run_batch(folder, 0, 2, 1)


def test_registered_batch_results_match_single_job_and_require_all_fields(registered_dataset, tmp_path, monkeypatch):
    pytest.importorskip("myonis_vis")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    path, _ = registered_dataset
    single, status = pipeline.run_pipeline(path, tmp_path/"single", workers=2, troubleshoot_timing=True)
    assert status["status"] == "complete"
    batched = pipeline.prepare(path, tmp_path/"batched", workers=2, troubleshoot_timing=True)
    assert slurm.run_batch(batched, 0, 2, 2)["complete"] == 1
    with pytest.raises(ValueError, match="1/2 field checkpoints"):
        pipeline.finalize(batched, require_complete=True)
    assert not (batched/"myonis_4i_data.h5ad").exists()
    assert (batched/"logs/finalize.error.json").is_file()
    assert slurm.run_batch(batched, 1, 2, 2)["complete"] == 1
    assert slurm.run_batch(batched, 0, 2, 2)["resumed"] == 1
    result = pipeline.finalize(batched, require_complete=True)
    assert result["status"] == "complete" and result["failed_tasks"] == 0
    assert result["timing_summary"]["timed_fields"] == 2
    assert not (batched/"logs/finalize.error.json").exists()
    a, b = ad.read_h5ad(single/"myonis_4i_data.h5ad"), ad.read_h5ad(batched/"myonis_4i_data.h5ad")
    pd.testing.assert_frame_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.X, b.X)
    for key in a.obsm:
        if isinstance(a.obsm[key], pd.DataFrame):
            pd.testing.assert_frame_equal(a.obsm[key], b.obsm[key])
        else:
            np.testing.assert_array_equal(a.obsm[key], b.obsm[key])
    pd.testing.assert_frame_equal(a.uns["myotubes"], b.uns["myotubes"])
    for key in a.uns["structures"]:
        pd.testing.assert_frame_equal(a.uns["structures"][key], b.uns["structures"][key])
    for key in a.uns["relationships"]:
        pd.testing.assert_frame_equal(a.uns["relationships"][key], b.uns["relationships"][key])
    pd.testing.assert_frame_equal(a.uns["registration"]["transforms"], b.uns["registration"]["transforms"])


def test_strict_final_submission_and_custom_reader_count(fake_scheduler):
    _, calls, _ = fake_scheduler
    receipt = slurm.submit_batch("dataset.yaml", jobs=2, workers=2, finalize_workers=6, require_complete=True)
    assert "--cpus-per-task=6" in calls[1][0]
    assert "--require-complete" in calls[1][1]["input"]
    assert "--allow-partial" not in calls[1][1]["input"]
    assert receipt["finalize_workers"] == 6 and receipt["require_complete"] is True
