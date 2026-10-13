import pytest

from fouri_quantification.diagnostics import field_timing_summary, summarize_timing, report_timing_summary


def timing(**stages):
    return [dict(stage=k, seconds=v) for k, v in stages.items()]


def measurement(patch, channel, status="measured"):
    return dict(spatial_key=patch, acquisition_id=channel, status=status)


def test_worker_rates_use_measured_images_and_do_not_double_count_child_timers(capsys):
    ordinary = field_timing_summary(
        [measurement("p1", "dapi"), measurement("p1", "myhc"), measurement("p2", "dapi"),
         measurement("p2", "myhc", "missing_image"), measurement("p1", "myhc", "structures_available")],
        timing(quantification_total=20, registration_qc_total=10, texture_features=8, qc_scoring=6,
               input_and_software_verification=3, write_checkpoint=2, registration_artifact_checksums=.5))
    diagnostic = field_timing_summary(
        [measurement("p3", "dapi"), measurement("p3", "myhc")],
        timing(quantification_total=50, registration_qc_total=30, measurement_visualisations=20,
               prediction_visualisations=5, qc_rendering=8, qc_comparison_exports=12))
    summary = summarize_timing([ordinary, diagnostic], 2, 120, dict(complete=2, resumed=0, failed=0))
    plain, debug = summary["groups"]["without_diagnostics"], summary["groups"]["with_diagnostics"]
    assert (plain["patches"], plain["images"], plain["missing_images"]) == (2, 3, 1)
    assert plain["core_seconds_per_patch"] == 15
    assert plain["accounted_seconds_per_patch"] == 17.75
    assert debug["core_seconds_per_patch"] == 35
    assert debug["diagnostic_seconds_per_patch"] == 45
    assert summary["total"]["core_seconds_per_patch"] == pytest.approx(65/3)
    assert summary["total"]["measured_seconds_per_image"] == 9
    assert summary["wall"]["seconds_per_patch"] == 40
    assert summary["wall"]["seconds_per_image"] == 24
    report_timing_summary(summary)
    output = capsys.readouterr().out
    assert "3 spatial patches; 5 quantified channel images (1 missing)" in output
    assert "Without diagnostic exports (1 fields)" in output
    assert "With diagnostic exports (1 fields)" in output
    assert "9.00 s/quantified channel image" in output


@pytest.mark.parametrize("counts,elapsed", [
    (dict(complete=1, resumed=1, failed=0), 10),
    (dict(complete=2, resumed=0, failed=1), 10),
    ({}, None),
])
def test_resumed_partial_or_standalone_finalization_does_not_claim_wall_throughput(counts, elapsed):
    field = field_timing_summary([measurement("p", "dapi")], timing(quantification_total=5))
    summary = summarize_timing([field, field], 2, elapsed, counts)
    assert summary["total"]["core_seconds_per_patch"] == 5
    assert summary["wall"]["seconds_per_patch"] is None
    assert summary["wall"]["seconds_per_image"] is None


def test_missing_images_or_missing_timers_have_no_fabricated_rates():
    field = field_timing_summary([measurement("p", "dapi", "missing_image")], timing(quantification_total=5))
    summary = summarize_timing([field], 1)
    assert summary["total"]["measured_seconds_per_image"] is None
    assert field_timing_summary([measurement("p", "dapi")], []) is None
    summary = summarize_timing([], 1, 10, dict(complete=1))
    assert summary["timed_fields"] == 0
    assert summary["total"]["core_seconds_per_patch"] is None
    assert summary["wall"]["seconds_per_patch"] is None


def test_incompletely_timed_run_has_worker_averages_but_no_wall_rate():
    field = field_timing_summary([measurement("p", "dapi")], timing(quantification_total=5))
    summary = summarize_timing([field], 2, 10, dict(complete=2, resumed=0, failed=0))
    assert summary["timed_fields"] == 1
    assert summary["total"]["core_seconds_per_patch"] == 5
    assert summary["wall"]["seconds_per_patch"] is None
