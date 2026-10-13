from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from utils.manifests import write_csv, write_json
from utils.registry import ACQUISITIONS


def test_partial_html_and_pdf_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    provenance = run_dir / "provenance"
    provenance.mkdir(parents=True)
    config = {
        "data_root": str(tmp_path / "data"),
        "evaluation_dir": str(tmp_path / "evaluation"),
        "patch_size": 1024,
        "random_seed": 1,
        "correction_batch_size": 2,
    }
    (provenance / "pipeline_config.json").write_text(json.dumps(config), encoding="utf-8")
    selections = []
    for acquisition in ACQUISITIONS:
        selections.append(
            {
                "acquisition_id": acquisition.acquisition_id,
                "measurement": acquisition.measurement,
                "marker": acquisition.marker,
                "chosen_method": "Harmony-like",
                "review_status": "approved",
                "basic_model_sha256": "a" * 64,
                "harmony_profile_sha256": "b" * 64,
            }
        )
    write_csv(provenance / "illumination_method_selection.csv", selections)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "synthetic_test",
            "status": "initialized",
            "started": {"local": "2026-08-20T12:00:00+01:00"},
            "hostname": "test-host",
            "git": {"commit": "test"},
        },
    )
    script = Path(__file__).resolve().parents[1] / "build_run_report.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "run_report.html").is_file()
    assert (run_dir / "run_report.pdf").is_file()
    assert (run_dir / "run_report.pdf").stat().st_size > 1000
    report = (run_dir / "run_report.html").read_text(encoding="utf-8")
    assert "synthetic_test" in report
    assert "Illumination selections" in report

