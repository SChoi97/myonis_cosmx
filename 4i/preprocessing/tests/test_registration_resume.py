"""Recovery pins processing code and submits workers with the active environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_registration_resume as recovery
from utils.manifests import sha256_file
from utils.registry import ACQUISITIONS, PLATES


PIPELINE = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ("submit_preprocessing_1080_global_registration.sh", "field"),
    ("submit_preprocessing_1080.sh", "patch"),
)


class RegistrationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.helper = self.source / "4i/preprocessing/prepare_registration_resume.py"
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text("resume helper\n")
        self.run = self.root / "run"
        self.provenance = self.run / "provenance"
        self.provenance.mkdir(parents=True)
        config = {
            "registration": {"level": "field_level"},
            "test_mode": False,
            "patch_size": 1024,
        }
        (self.provenance / "pipeline_config.json").write_text(json.dumps(config))
        (self.provenance / "illumination_method_selection.csv").write_text("selection\n")
        (self.provenance / "clip_limits.xlsx").write_bytes(b"limits")
        (self.provenance / "registration_overrides.csv").write_text("overrides\n")
        self.manifest = {
            "run_mode": "field_level_production",
            "snapshots": {
                key: sha256_file(self.provenance / name)
                for name, key in (
                    ("pipeline_config.json", "config_sha256"),
                    ("illumination_method_selection.csv", "selection_sha256"),
                    ("clip_limits.xlsx", "clip_limits_sha256"),
                )
            },
            "slurm_job_graph": {
                "real_data_smoke_test": "100",
                "illumination": "200",
                "registration": "300",
            },
        }
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.worker = self.source / "4i/preprocessing/register_and_crop.py"
        self.worker.write_text("original processing code\n")
        self.expected_code = {}
        for relative in sorted(recovery.RESUME_SUBMISSION_SCRIPTS):
            source = self.source / relative
            source.write_text("original submission settings\n")
            self.expected_code[relative] = sha256_file(source)
            source.write_text("updated submission settings\n")
        self.expected_code[str(self.worker.relative_to(self.source))] = sha256_file(self.worker)
        self.code_path = self.provenance / "code_sha256.json"
        self.code_path.write_text(json.dumps(self.expected_code))
        for acquisition in ACQUISITIONS:
            for plate in PLATES:
                folder = self.run / "corrected" / acquisition.acquisition_id
                (folder / f"{plate.name}.ome.zarr").mkdir(parents=True)
                (folder / f"{plate.name}_SUCCESS.json").write_text(
                    json.dumps({"status": "complete", "task_hash": "corrected-hash"})
                )

    def prepare(self, **kwargs):
        with patch.object(recovery, "__file__", str(self.helper)):
            return recovery.prepare_resume(
                self.run, "field_level_production", 1024, **kwargs
            )

    def test_recovery_preserves_original_hashes_inputs_and_job_graph(self):
        old_code = self.code_path.read_bytes()
        old_manifest = (self.run / "run_manifest.json").read_bytes()
        jobs = self.prepare()
        self.assertEqual(jobs, ("100", "200"))
        history, = (self.provenance / "registration_resume").iterdir()
        self.assertEqual((history / "code_sha256_before.json").read_bytes(), old_code)
        self.assertEqual((history / "run_manifest_before.json").read_bytes(), old_manifest)
        self.assertEqual((self.run / "run_manifest.json").read_bytes(), old_manifest)
        new_code = json.loads(self.code_path.read_text())
        self.assertEqual(set(new_code), set(self.expected_code))
        self.assertEqual(new_code[str(self.worker.relative_to(self.source))], sha256_file(self.worker))
        for relative in recovery.RESUME_SUBMISSION_SCRIPTS:
            self.assertEqual(new_code[relative], sha256_file(self.source / relative))
        for name, key in (
            ("pipeline_config.json", "config_sha256"),
            ("illumination_method_selection.csv", "selection_sha256"),
            ("clip_limits.xlsx", "clip_limits_sha256"),
        ):
            self.assertEqual(sha256_file(self.provenance / name), self.manifest["snapshots"][key])

    def test_processing_code_change_is_rejected_without_mutating_provenance(self):
        old_code = self.code_path.read_bytes()
        self.worker.write_text("changed processing code\n")
        with self.assertRaisesRegex(RuntimeError, "changed:.*register_and_crop"):
            self.prepare()
        self.assertEqual(self.code_path.read_bytes(), old_code)
        self.assertFalse((self.provenance / "registration_resume").exists())

    def test_missing_corrected_store_is_rejected_before_hash_update(self):
        old_code = self.code_path.read_bytes()
        store = self.run / "corrected" / ACQUISITIONS[0].acquisition_id / f"{PLATES[0].name}.ome.zarr"
        store.rmdir()
        with self.assertRaisesRegex(RuntimeError, "completed corrected inputs"):
            self.prepare()
        self.assertEqual(self.code_path.read_bytes(), old_code)
        self.assertFalse((self.provenance / "registration_resume").exists())

    def test_check_only_keeps_actual_snapshot_unchanged(self):
        old_code = self.code_path.read_bytes()
        self.assertEqual(self.prepare(check_only=True), ("100", "200"))
        self.assertEqual(self.code_path.read_bytes(), old_code)
        self.assertFalse((self.provenance / "registration_resume").exists())

    def test_different_registration_mode_is_rejected(self):
        with patch.object(recovery, "__file__", str(self.helper)):
            with self.assertRaisesRegex(ValueError, "must match RUN_MODE"):
                recovery.prepare_resume(self.run, "patch_level_production", 1024)
        self.assertFalse((self.provenance / "registration_resume").exists())

    def test_changed_saved_input_is_rejected(self):
        (self.provenance / "illumination_method_selection.csv").write_text("changed selection\n")
        with self.assertRaisesRegex(RuntimeError, "Run input differs"):
            self.prepare()
        self.assertFalse((self.provenance / "registration_resume").exists())


class SubmissionChainTests(unittest.TestCase):
    def test_both_controllers_resume_existing_runs_and_keep_new_run_support(self):
        for filename, mode in SCRIPTS:
            for resume in (True, False):
                with self.subTest(script=filename, resume=resume):
                    self.check_controller(filename, mode, resume)

    def check_controller(self, filename, mode, resume):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            commands = folder / "bin"
            commands.mkdir()
            script = folder / filename
            # Redirect cluster-only paths for the new-run branch in this fixture.
            original_root = "/nemo/lab/tedescos/home/users/chois1/computational_tools/myonis_cosmx"
            script.write_text((PIPELINE / filename).read_text().replace(original_root, str(folder / "repo")))
            shim = commands / "python"
            shim.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
folder = Path(os.environ["TEST_FOLDER"])
script, *args = sys.argv[1:]
with (folder / "python_calls.jsonl").open("a") as f:
    f.write(json.dumps([script, *args]) + "\\n")
if script == "prepare_registration_resume.py":
    print("100 200")
elif "--count-tasks" in args:
    print({"real_data_smoke_test.py": 2, "apply_illumination_correction.py": 40,
           "register_and_crop.py": 120, "clip_and_export_png.py": 40}[script])
elif script == "initialize_run.py":
    print(folder / "new-run")
elif script in {"prepare_method_selection.py", "record_job_graph.py"}:
    pass
else:
    raise RuntimeError("Unexpected pipeline invocation: " + script)
''')
            sbatch = commands / "sbatch"
            sbatch.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
folder = Path(os.environ["TEST_FOLDER"])
log = folder / "sbatch_calls.jsonl"
prior = log.read_text().splitlines() if log.exists() else []
with log.open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
keys = ("FOURI_RUN_DIR", "FOURI_PYTHON", "FOURI_CLEANUP_STAGE", "PATH",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "ACTIVE_ENV_SENTINEL",
        "SBATCH_GET_USER_ENV", "SLURM_GET_USER_ENV", "SLURM_EXPORT_ENV")
with (folder / "sbatch_environment.jsonl").open("a") as f:
    f.write(json.dumps({key: os.environ.get(key) for key in keys}) + "\\n")
print(1000 + len(prior))
''')
            shim.chmod(0o755)
            sbatch.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                # A stale interpreter must not override the currently active PATH.
                "FOURI_PYTHON": "/old/environment/bin/python",
                "CONDA_PREFIX": str(folder / "active-environment"),
                "CONDA_DEFAULT_ENV": "active-test-env",
                "ACTIVE_ENV_SENTINEL": "inherited-from-submitting-shell",
                "SBATCH_GET_USER_ENV": "1",
                "SLURM_GET_USER_ENV": "1",
                "SLURM_EXPORT_ENV": "NONE",
                "FOURI_CLEANUP_STAGE": "old-stage",
                "TEST_FOLDER": str(folder),
            }
            environment.pop("FOURI_RESUME_RUN_DIR", None)
            if not resume:
                environment["FOURI_RESUME_RUN_DIR"] = ""
            result = subprocess.run(
                ["bash", str(script)], cwd=folder, env=environment,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (folder / "sbatch_calls.jsonl").read_text().splitlines()]
            submitted_environments = [json.loads(line) for line in (folder / "sbatch_environment.jsonl").read_text().splitlines()]
            python_calls = [json.loads(line) for line in (folder / "python_calls.jsonl").read_text().splitlines()]
            workers = [call[-1] for call in calls]
            downstream = ["registration_worker.sh", "cleanup_worker.sh", "patch_worker.sh",
                          "cleanup_worker.sh", "clipping_worker.sh", "report_worker.sh"]
            self.assertEqual(workers, downstream if resume else ["smoke_worker.sh", "illumination_worker.sh", *downstream])
            registration_index = 0 if resume else 2
            if resume:
                run_dir = str(folder / "repo/4i/preprocessing/results/runs" / f"4i_{mode}_level_production_20261004_204300_f6ecb24")
                self.assertNotIn("initialize_run.py", [call[0] for call in python_calls])
                self.assertNotIn("prepare_method_selection.py", [call[0] for call in python_calls])
                self.assertFalse(any(arg.startswith("--dependency=") for arg in calls[0]))
            else:
                run_dir = str(folder / "new-run")
                self.assertIn("--dependency=afterok:1001", calls[2])
            self.assertIn(f"Python from submission environment: {shim}", result.stdout)
            self.assertIn("Conda environment: active-test-env", result.stdout)
            self.assertEqual(len(submitted_environments), len(calls))
            for index, (call, submitted) in enumerate(zip(calls, submitted_environments)):
                self.assertEqual([arg for arg in call if arg.startswith("--export=")], ["--export=ALL"])
                self.assertFalse(any(arg.startswith("--get-user-env") for arg in call))
                self.assertEqual(submitted["FOURI_RUN_DIR"], run_dir)
                self.assertEqual(submitted["FOURI_PYTHON"], str(shim))
                for key in ("PATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "ACTIVE_ENV_SENTINEL"):
                    self.assertEqual(submitted[key], environment[key])
                self.assertIsNone(submitted["SBATCH_GET_USER_ENV"])
                self.assertIsNone(submitted["SLURM_GET_USER_ENV"])
                self.assertEqual(submitted["SLURM_EXPORT_ENV"], "ALL")
                expected_stage = {registration_index + 1: "corrected",
                                  registration_index + 3: "registered_crops"}.get(index)
                self.assertEqual(submitted["FOURI_CLEANUP_STAGE"], expected_stage)
            for index in range(registration_index + 1, len(calls) - 1):
                self.assertIn(f"--dependency=afterok:{1000 + index - 1}", calls[index])
            report_jobs = ":".join(str(1000 + index) for index in range(len(calls) - 1))
            self.assertIn(f"--dependency=afterany:{report_jobs}", calls[-1])
            self.assertIn("--array=0-119%8", calls[registration_index])
            self.assertIn("--array=0-39%8", calls[-2])
            record = next(call for call in python_calls if call[0] == "record_job_graph.py")
            self.assertEqual(record[record.index("--registration-job") + 1], str(1000 + registration_index))
            self.assertEqual(record[record.index("--smoke-job") + 1], "100" if resume else "1000")
            self.assertEqual(record[record.index("--illumination-job") + 1], "200" if resume else "1001")


if __name__ == "__main__":
    unittest.main()
