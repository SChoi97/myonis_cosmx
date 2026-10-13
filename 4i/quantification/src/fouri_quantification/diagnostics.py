"""Optional timing and bounded, reproducible diagnostic selection."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import re
import sys
import time


class StageTimer:
    def __init__(self, enabled=False, label=""):
        self.enabled, self.label = enabled, label
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    def call(self, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        with self.stage(name):
            return function(*args, **kwargs)

    def report(self):
        if self.enabled:
            values = ", ".join(f"{key}={value:.2f}s" for key, value in sorted(self.seconds.items(), key=lambda x: -x[1]))
            print(f"[timing] {self.label}: {values}", flush=True)

    def rows(self):
        return [dict(scope=self.label, stage=key, seconds=value, calls=self.calls[key])
                for key, value in self.seconds.items()]


def field_timing_summary(measurements, timings):
    """Amortise field work over spatial patches and measured channel images.

    Parent timers include their children. Subtract only disjoint diagnostic
    stages; do not add feature timers to quantification_total again.
    """
    stages = {row["stage"]: float(row["seconds"]) for row in timings}
    if "quantification_total" not in stages:
        return None  # A checkpoint created without timing cannot supply rates.
    images = [r for r in measurements if r["status"] in {"measured", "missing_image"}]
    key = lambda r: (r["spatial_key"], r["acquisition_id"])
    qc_exports = sum(stages.get(k, 0) for k in ("qc_rendering", "qc_comparison_exports"))
    measurement_exports = sum(stages.get(k, 0) for k in ("measurement_visualisations", "prediction_visualisations"))
    return dict(
        fields=1, patches=len({r["spatial_key"] for r in images}),
        images=len({key(r) for r in images if r["status"] == "measured"}),
        missing_images=len({key(r) for r in images if r["status"] == "missing_image"}),
        registration_seconds=max(0.0, stages.get("registration_qc_total", 0)-qc_exports),
        measurement_seconds=max(0.0, stages["quantification_total"]-measurement_exports),
        diagnostic_seconds=qc_exports+measurement_exports,
        overhead_seconds=sum(stages.get(k, 0) for k in
                             ("input_and_software_verification", "write_checkpoint", "registration_artifact_checksums")))


def summarize_timing(fields, completed_fields, elapsed_seconds=None, execution_counts=None):
    """Worker averages may include resumed checkpoints; wall rates must not."""
    def aggregate(group):
        values = {key: sum(row[key] for row in group) for key in
                  ("fields", "patches", "images", "missing_images", "registration_seconds",
                   "measurement_seconds", "diagnostic_seconds", "overhead_seconds")}
        core = values["registration_seconds"]+values["measurement_seconds"]
        values.update(
            core_seconds_per_patch=core/values["patches"] if values["patches"] else None,
            measured_seconds_per_image=values["measurement_seconds"]/values["images"] if values["images"] else None,
            diagnostic_seconds_per_patch=values["diagnostic_seconds"]/values["patches"] if values["patches"] else None,
            accounted_seconds_per_patch=(core+values["diagnostic_seconds"]+values["overhead_seconds"])/values["patches"]
                                        if values["patches"] else None)
        return values

    groups = {name: aggregate([r for r in fields if bool(r["diagnostic_seconds"]) == rendered])
              for name, rendered in [("without_diagnostics", False), ("with_diagnostics", True)]}
    total = aggregate(fields)
    counts = execution_counts or {}
    fresh_complete_run = (elapsed_seconds is not None and counts.get("complete") == completed_fields
                          and counts.get("resumed", 0) == 0 and counts.get("failed", 0) == 0
                          and len(fields) == completed_fields and completed_fields > 0)
    return dict(
        completed_fields=completed_fields, timed_fields=len(fields), groups=groups, total=total,
        definitions=dict(patch="One spatial patch across all quantified channels",
                         image="One measured channel image; reference DAPI only, excludes registration-only DAPI and sigmoid maps",
                         worker_rates="Field times amortised across patches/images, not individually timed image latencies. Core excludes diagnostic exports and verification/checkpoint overhead. Measurement includes shared object/feature work.",
                         wall_rates="Full pipeline elapsed time divided by completed units, including preparation, diagnostics and finalization; available only for fresh complete runs"),
        wall=dict(elapsed_seconds=elapsed_seconds, execution_counts=counts,
                  seconds_per_patch=elapsed_seconds/total["patches"] if fresh_complete_run and total["patches"] else None,
                  seconds_per_image=elapsed_seconds/total["images"] if fresh_complete_run and total["images"] else None))


def report_timing_summary(summary):
    total = summary["total"]
    fmt = lambda value: "n/a" if value is None else f"{value:.2f}"
    print(f"[timing summary] {summary['timed_fields']}/{summary['completed_fields']} completed fields with timings; "
          f"{total['patches']} spatial patches; {total['images']} quantified channel images "
          f"({total['missing_images']} missing).", flush=True)
    for name, label in [("without_diagnostics", "Without diagnostic exports"),
                        ("with_diagnostics", "With diagnostic exports")]:
        group = summary["groups"][name]
        if group["fields"]:
            print(f"  {label} ({group['fields']} fields): core {fmt(group['core_seconds_per_patch'])} s/patch; "
                  f"diagnostics {fmt(group['diagnostic_seconds_per_patch'])} s/patch; "
                  f"including recorded verification/checkpoint overhead {fmt(group['accounted_seconds_per_patch'])} s/patch.", flush=True)
    print(f"  Measurement work: {fmt(total['measured_seconds_per_image'])} s/quantified channel image. "
          "Worker averages, with shared field work amortised; patches span all quantified channels.", flush=True)
    wall = summary["wall"]
    if wall["elapsed_seconds"] is not None:
        print(f"  Pipeline wall time: {wall['elapsed_seconds']/60:.2f} min. "
              + (f"Effective throughput including all stages: {fmt(wall['seconds_per_patch'])} s/patch, "
                 f"{fmt(wall['seconds_per_image'])} s/channel image across all workers."
                 if wall["seconds_per_patch"] is not None else
                 "No full-run throughput estimate for resumed, partial or incompletely timed runs."), flush=True)
    print("  Image average includes reference DAPI; registration-only DAPI and sigmoid maps are not counted as extra images. "
          "Summary saved in run_manifest.json under timing_summary.", flush=True)


def sample_visualisation_fields(field_keys, config):
    vis = config["visualisation"]
    if not vis["enabled"]:
        return []
    keys = sorted(set(field_keys), key=lambda key: hashlib.sha256(f"visualisation|{vis['field_seed']}|{key}".encode()).hexdigest())
    return keys if vis["max_fields"] is None else keys[:vis["max_fields"]]


def field_label(meta):
    # Escape rather than truncate so identities cannot collide after sanitizing.
    def token(value):
        return re.sub(r"[^A-Za-z0-9_-]", lambda m: f"~{ord(m[0]):x}~", str(value))
    return (f"plate-{token(meta['plate_name'])}__rep-{int(meta['repeat']):02d}"
            f"__cell-{token(meta['cell_line'])}__well-{token(meta['well_id'])}"
            f"__field-f{int(meta['field_id']):02d}__cohort-{token(meta['cohort_id'])}")


def progress(iterable, total, description, unit="field"):
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"{description}: {total} {unit}s (tqdm unavailable; showing counts)", file=sys.stderr, flush=True)
        for number, item in enumerate(iterable, 1):
            yield item
            if number == 1 or number == total or number % max(1, total // 20) == 0:
                print(f"{description}: {number}/{total}", file=sys.stderr, flush=True)
    else:
        yield from tqdm(iterable, total=total, desc=description, unit=unit,
                        mininterval=1 if sys.stderr.isatty() else 15, dynamic_ncols=True)
