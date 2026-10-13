from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import pandas as pd

from .config import load_config, stable_hash
from .inputs.manifests import build_index, file_hash
from .outputs.h5ad import write_partition, read_partition, atomic_write
from .diagnostics import (StageTimer, sample_visualisation_fields, field_label, progress,
                          field_timing_summary, summarize_timing, report_timing_summary)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n")
    os.replace(temporary, path)


def code_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.suffix in {".py", ".yaml", ".json"}}


def verify_software_snapshot(run_dir, allow_finalizer_update=False):
    manifest = json.loads((Path(run_dir)/"run_manifest.json").read_text())
    from .finalization import check_finalizer_changes
    changes = check_finalizer_changes(manifest["code_hashes"], code_hashes(), allow_finalizer_update)
    for package, version in manifest["software_versions"].items():
        if metadata.version(package) != version:
            raise ValueError(f"Software version changed after prepare: {package}; prepare a new run")
    if manifest.get("visualisation_renderer"):
        from .reporting.visualisation import renderer_identity
        if renderer_identity() != manifest["visualisation_renderer"]:
            raise ValueError("MYONIS-VisTools changed after prepare; prepare a new run")
    from .registration import verify_sources
    verify_sources(json.loads((Path(run_dir)/"config/resolved.json").read_text()))
    return changes


def analysis_fingerprint(config, index):
    numerical = {k: v for k, v in config.items() if k not in {"config_path", "output_root", "outputs", "execution"}}
    versions = {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-image", "shapely", "anndata", "h5py"]}
    renderer = {}
    if config["visualisation"]["enabled"] or config.get("registration_qc", {}).get("enabled"):
        from .reporting.visualisation import renderer_identity
        renderer = renderer_identity()
    return stable_hash(dict(config=numerical, troubleshoot=config["execution"].get("troubleshoot", False),
                            fingerprints=index["fingerprints"], code=code_hashes(), versions=versions, renderer=renderer))


def prepare(config_path, run_dir=None, resume=False, workers=None, troubleshoot=False,
            troubleshoot_timing=False, visualisation_fields=None):
    config = load_config(config_path, troubleshoot, troubleshoot_timing, visualisation_fields)
    timer = StageTimer(config["execution"]["troubleshoot_timing"], "prepare")
    renderer = {}
    if config["visualisation"]["enabled"] or config.get("registration_qc", {}).get("enabled"):
        from .reporting.visualisation import renderer_identity
        renderer = renderer_identity()
    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be positive")
        config["execution"]["workers"] = workers
    print("Indexing selected images and segmentation assets...", flush=True)
    index = timer.call("index_inputs", build_index, config)
    errors = index["issues"][index["issues"].severity == "error"]
    if len(errors) and not config["execution"]["allow_partial"]:
        raise ValueError(f"Preflight found {len(errors)} required input errors. Use validate for details, or explicitly enable execution.allow_partial.")
    fingerprint = timer.call("fingerprint", analysis_fingerprint, config, index)
    snapshot_start = time.perf_counter()
    if run_dir is None:
        if resume:
            raise ValueError("--resume requires --run-dir")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config["output_root"]) / f"run_{stamp}_{fingerprint[:8]}"
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {run_dir}; use --resume to reuse a matching run")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume: configuration, code or selected inputs changed")
    else:
        run_dir.mkdir(parents=True)
    for folder in ["config", "inputs", "checkpoints", "tasks", "logs", "qc_overlays"]:
        (run_dir/folder).mkdir(exist_ok=True)
    write_json(run_dir/"config/resolved.json", config)
    write_partition(dict(images=index["images"], assets=index["assets"], issues=index["issues"], field_selection=index["field_selection"],
                         nucleus_qc=index["nucleus_qc"], registration_inputs=index["registration_inputs"]), run_dir/"inputs/index.h5ad")
    field_columns = ["field_key", "plate_name", "repeat", "well_id", "cell_line", "cohort_id", "field_id", "field_selection_rank"]
    write_json(run_dir/"inputs/selected_fields.json", index["images"][field_columns].drop_duplicates().sort_values("field_selection_rank").to_dict("records"))
    write_json(run_dir/"inputs/fingerprints.json", index["fingerprints"])
    write_json(run_dir/"config/markers.resolved.json", index["markers"])
    tasks = []
    visual_fields = sample_visualisation_fields(index["images"].field_key, config)
    visual_selection = dict(selected_fields=len(visual_fields), eligible_fields=int(index["images"].field_key.nunique()),
                            max_fields=config["visualisation"]["max_fields"], seed=config["visualisation"]["field_seed"],
                            field_keys=visual_fields)
    write_json(run_dir/"inputs/visualisation_fields.json", visual_selection)
    groups = index["images"].groupby("field_key", sort=True)
    for i, (field_key, group) in enumerate(progress(groups, groups.ngroups, "Prepare fields")):
        task_id = stable_hash(field_key)[:20]
        task = dict(task_id=task_id, field_key=field_key, fingerprint=fingerprint,
                    images=group.to_dict("records"), assets=index["assets"][index["assets"].field_key == field_key].to_dict("records"),
                    visualisation=field_key in visual_fields, label=field_label(group.iloc[0]),
                    overlay=bool(field_key in visual_fields and i < int(config["outputs"]["qc_overlays"])))
        qc_rows = index["nucleus_qc"]["objects"]
        task["nucleus_qc_rows"] = qc_rows[qc_rows.field_key == field_key].to_dict("records")
        task["nucleus_qc_files"] = index["nucleus_qc"]["files"]
        task["registration_inputs"] = index["registration_inputs"][index["registration_inputs"].field_key == field_key].to_dict("records")
        write_json(run_dir/"tasks"/(task_id+".json"), task)
        tasks.append(dict(task_id=task_id, field_key=field_key))
    write_json(run_dir/"tasks.json", tasks)
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-image", "shapely", "anndata", "h5py", "tifffile", "reportlab"]:
        versions[package] = metadata.version(package)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if timer.enabled:
        timer.seconds["write_run_snapshots"] += time.perf_counter()-snapshot_start
        timer.calls["write_run_snapshots"] += 1
    manifest = dict(previous, schema_version=1, dataset_id=config["dataset_id"], fingerprint=fingerprint,
                    status="prepared", n_tasks=len(tasks), input_issue_counts=index["issues"].status.value_counts().to_dict(),
                    started_utc=previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
                    worker_count=config["execution"]["workers"], code_hashes=code_hashes(), software_versions=versions,
                    field_selection=index["field_selection"], visualisation_renderer=renderer,
                    visualisation_selection=visual_selection, timings=timer.rows(),
                    measurement_overlap_policy=config["measurement"]["overlap_policy"])
    write_json(manifest_path, manifest)
    timer.report()
    print(f"Prepared {len(tasks)} fields; visualisations for {len(visual_fields)} fields. Run: {run_dir}", flush=True)
    return run_dir


def execute_task(run_dir, task_index):
    from .worker import process_field
    run_dir = Path(run_dir)
    tasks = json.loads((run_dir/"tasks.json").read_text())
    entry = tasks[int(task_index)]
    task_id = entry["task_id"]
    task = json.loads((run_dir/"tasks"/(task_id+".json")).read_text())
    config = json.loads((run_dir/"config/resolved.json").read_text())
    result_path = run_dir/"checkpoints"/(task_id+".h5ad")
    done_path = result_path.with_suffix(".done.json")
    markers = json.loads((run_dir/"config/markers.resolved.json").read_text())
    timer = StageTimer(config["execution"].get("troubleshoot_timing", False), task.get("label", task["field_key"]))
    verification_start = time.perf_counter()
    try:
        verify_software_snapshot(run_dir)
        fingerprints = json.loads((run_dir/"inputs/fingerprints.json").read_text())
        paths = {r["resolved_path"] for r in task["images"]} | {r["resolved_path"] for r in task["assets"] if r["resolved_path"]}
        paths.update(task.get("nucleus_qc_files", []))
        paths.update(row["path"] for row in task.get("registration_inputs", []))
        if config["myhc_filter"]["enabled"]:
            paths.add(config["myhc_filter"]["metadata_csv"])
        for path in paths:
            before = fingerprints[path]
            p = Path(path)
            if before.get("missing"):
                if p.exists(): raise ValueError(f"Input appeared after prepare: {path}; prepare a new run")
                continue
            if not p.is_file(): raise ValueError(f"Input disappeared after prepare: {path}")
            stat = p.stat()
            if (stat.st_size != before["size"] or (before.get("sha256") and file_hash(p) != before["sha256"])
                    or (not before.get("sha256") and stat.st_mtime_ns != before["mtime_ns"])):
                raise ValueError(f"Input changed after prepare: {path}; prepare a new run")
        if result_path.is_file() and done_path.is_file():
            done = json.loads(done_path.read_text())
            visualisations_intact = done.get("visualisation_status") != "failed" and all(
                (run_dir/path).is_file() and file_hash(run_dir/path) == checksum
                for path, checksum in done.get("visualisation_files", {}).items())
            registration_intact = all((run_dir/path).is_file() and file_hash(run_dir/path) == checksum
                                     for path, checksum in done.get("registration_files", {}).items())
            if done["fingerprint"] == task["fingerprint"] and done["sha256"] == file_hash(result_path) and visualisations_intact and registration_intact:
                return dict(task_id=task_id, status="resumed")
        transforms, imported_qc, registration_files = None, None, {}
        if timer.enabled:
            timer.seconds["input_and_software_verification"] += time.perf_counter()-verification_start
            timer.calls["input_and_software_verification"] += 1
        selected_vis = task.get("visualisation", config["visualisation"]["enabled"])
        if config["execution"].get("troubleshoot"):
            config["registration_qc"]["fast_mode"] = not selected_vis
            config["registration_qc"]["compare_metrics"] = selected_vis
        elif not selected_vis:
            config["registration_qc"]["fast_mode"] = True
        registration_start = time.perf_counter()
        if config.get("registration_qc", {}).get("enabled"):
            from .registration import run_field
            qc_folder = run_dir/"registration_qc"/task_id
            transforms, imported_qc = timer.call("registration_qc_total", run_field, config, task, qc_folder, timer=timer)
            with timer.stage("registration_artifact_checksums"):
                registration_files = {str(p.relative_to(run_dir)): file_hash(p) for p in qc_folder.rglob("*") if p.is_file()}
        registration_duration = time.perf_counter()-registration_start
        result = timer.call("quantification_total", process_field, config, task["images"], task["assets"], markers,
                               str(run_dir/"qc_overlays"/(task_id+".png")) if task["overlay"] else None,
                               str(run_dir/"qc_visualisations"/task.get("label", task_id)) if selected_vis else None,
                               nucleus_qc_rows=imported_qc["objects"].to_dict("records") if imported_qc is not None else task.get("nucleus_qc_rows"),
                               registration_rows=transforms.to_dict("records") if transforms is not None else None, timer=timer)
        if imported_qc is not None:
            result["nucleus_qc"] = imported_qc
            result["registration"] = transforms
            result["registration_duration_seconds"] = registration_duration
            result["duration_seconds"] += registration_duration
        timer.call("write_checkpoint", write_partition, result, result_path)
        vis = result["visualisations"]
        vis_files = {row[col]: file_hash(run_dir/row[col]) for row in vis.to_dict("records")
                     for col in ["png", "audit", "masks", "metadata"] if row.get(col)}
        vis_failed = bool(len(vis) and vis.status.isin(["failed", "mismatch"]).any())
        write_json(done_path, dict(fingerprint=task["fingerprint"], sha256=file_hash(result_path), duration_seconds=result["duration_seconds"],
                                  timings=timer.rows(),
                                  registration_files=registration_files,
                                  visualisation_status="failed" if vis_failed else "complete", visualisation_files=vis_files))
        error_path = run_dir/"logs"/(task_id+".error.json")
        if error_path.exists(): error_path.unlink()
        timer.report()
        return dict(task_id=task_id, status="complete")
    except Exception as exc:
        write_json(run_dir/"logs"/(task_id+".error.json"), dict(task_id=task_id, field_key=task["field_key"], error=str(exc), traceback=traceback.format_exc()))
        raise


def finalize(run_dir, started_perf_counter=None, execution_counts=None, require_complete=False,
             workers=1, scratch_dir=None, allow_finalizer_update=False):
    from .outputs.aggregate import aggregate
    from .access import get_myotubes
    from .finalization import (preflight_checkpoints, read_checkpoints, local_checkpoint, excluded_fields)
    if workers < 1:
        raise ValueError("Finalisation workers must be positive")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and workers > int(allocated):
        raise ValueError(f"Requested {workers} finalisation workers but Slurm allocated only {allocated} CPUs")
    if scratch_dir is not None and not Path(scratch_dir).is_dir():
        raise ValueError("scratch-dir must be an existing directory with space for temporary checkpoints")
    run_dir = Path(run_dir)
    changes = verify_software_snapshot(run_dir, allow_finalizer_update=allow_finalizer_update)
    config = json.loads((run_dir/"config/resolved.json").read_text())
    timer = StageTimer(config["execution"].get("troubleshoot_timing", False), "finalize")
    manifest = json.loads((run_dir/"run_manifest.json").read_text())
    tasks = json.loads((run_dir/"tasks.json").read_text())
    manifest["finalization"] = dict(workers=workers, checkpoint_reads="verified_local_copies",
        require_complete=require_complete, allow_finalizer_update=allow_finalizer_update,
        changed_code_files=changes, code_hashes=code_hashes(), started_utc=datetime.now(timezone.utc).isoformat())
    ready, failures = timer.call("preflight_checkpoints", preflight_checkpoints, run_dir,
                                 progress(tasks, len(tasks), "Check checkpoints"), manifest["fingerprint"])

    def refuse_incomplete(completed):
        manifest.update(status="failed", completed_tasks=completed, failed_tasks=len(failures))
        write_json(run_dir/"run_manifest.json", manifest)
        write_json(run_dir/"logs/finalize.error.json", dict(failures=failures))
        raise ValueError(f"Cannot finalize: {len(failures)}/{len(tasks)} field checkpoints are missing or invalid. "
                         "The final H5AD was not updated. See logs/finalize.error.json. "
                         "Use --allow-partial to explicitly assemble valid fields, or rerun failed fields.")

    if not ready or (require_complete and failures):
        refuse_incomplete(0)
    print(f"Finalisation: {len(ready)}/{len(tasks)} checkpoints ready; {len(failures)} fields excluded. "
          f"Reading verified local copies with {min(workers, len(ready))} workers.", flush=True)
    with timer.stage("read_input_index"):
        with local_checkpoint(run_dir/"inputs/index.h5ad", scratch_dir) as local:
            index = read_partition(local)
    index["visualisation_selection"] = manifest.get("visualisation_selection", {})
    index["markers"] = json.loads((run_dir/"config/markers.resolved.json").read_text())
    index["fingerprints"] = json.loads((run_dir/"inputs/fingerprints.json").read_text())
    parts, field_timings = [], []
    timings = list(manifest.get("timings", []))
    with timer.stage("read_and_check_checkpoints"):
        results = read_checkpoints(run_dir, ready, manifest["fingerprint"], min(workers, len(ready)), scratch_dir)
        for part, done, failure in progress(results, len(ready), "Read checkpoints"):
            if failure is not None:
                failures.append(failure)
                continue
            timings.extend(done.get("timings", []))
            parts.append(part)
            if timer.enabled:
                field_timing = field_timing_summary(part["qc"].to_dict("records"), done.get("timings", []))
                if field_timing is not None:
                    field_timings.append(field_timing)
    if not parts or (require_complete and failures):
        refuse_incomplete(len(parts))
    exclusions = excluded_fields(failures, index["images"])
    index["excluded_fields"] = exclusions
    manifest["excluded_fields"] = exclusions.to_dict("records")
    manifest["excluded_patches"] = int(exclusions.excluded_patches.sum())
    if len(exclusions):
        exclusions.to_csv(run_dir/"excluded_fields.csv", index=False)
        print(f"Assembling {len(parts)} completed fields; excluding {len(exclusions)} fields "
              f"({manifest['excluded_patches']} patches). Reasons: excluded_fields.csv and the PDF.", flush=True)
    elif (run_dir/"excluded_fields.csv").exists():
        (run_dir/"excluded_fields.csv").unlink()
    measured_issues = sum(int((p["qc"].status.isin(["missing_image", "structures_unavailable"])).sum()) for p in parts)
    visualisations = pd.concat([p["visualisations"] for p in parts], ignore_index=True) if parts else pd.DataFrame()
    vis_failures = int(visualisations.status.isin(["failed", "mismatch"]).sum()) if len(visualisations) else 0
    vis_status = "disabled"
    if config["visualisation"]["enabled"]:
        from .reporting.visualisation import build_gallery
        selected_count = manifest.get("visualisation_selection", {}).get("selected_fields", 1)
        if selected_count:
            build_gallery(run_dir, visualisations, manifest.get("visualisation_selection", manifest["field_selection"]))
            vis_status = "failed" if vis_failures or not len(visualisations) else ("partial" if failures else "complete")
        else:
            vis_status = "not_requested"
    measured_issues += vis_failures
    warnings = int(index["issues"].severity.isin(["warning", "error"]).sum())
    status = "failed" if not parts else ("partial" if failures else ("complete_with_omissions" if warnings or measured_issues else "complete"))
    manifest.update(status=status, completed_tasks=len(parts), failed_tasks=len(failures),
                    output_file="myonis_4i_data.h5ad",
                    n_nuclei=sum(len(p["nuclei"]) for p in parts), n_myotubes=sum(len(p["myotubes"]) for p in parts))
    manifest.update(visualisation_status=vis_status, visualisation_views=len(visualisations), visualisation_failures=vis_failures)
    provenance = dict(run_manifest_json=json.dumps(manifest), config_json=json.dumps(config),
                      markers_json=json.dumps(index["markers"]), status=status,
                      note="Supplied patch objects are retained; edge myotubes may be fragments. Cohorts are never implicitly joined.")
    from .registration import export_run
    timer.call("combine_registration_qc", export_run, run_dir, parts, config, status)
    run = timer.call("aggregate_features", aggregate, parts, index, config, provenance, failures)
    timer.call("write_h5ad", atomic_write, run, run_dir/"myonis_4i_data.h5ad")
    manifest["n_nuclei"], manifest["n_myotubes"] = run.n_obs, len(run.uns["myotubes"])
    if config["outputs"]["object_h5ad"]:
        target = run_dir/"exports"; target.mkdir(exist_ok=True)
        nuclei_export = run.copy()
        for key in ["myotubes", "structures", "geometry", "object_registry", "relationships"]:
            nuclei_export.uns.pop(key, None)
        atomic_write(nuclei_export, target/"nuclei.h5ad")
        atomic_write(get_myotubes(run, as_anndata=True), target/"myotubes.h5ad")
    if config["outputs"]["pdf"]:
        try:
            from .reporting.pdf import build_report
            timer.call("pdf_report", build_report, run_dir/"myonis_4i_data.h5ad", run_dir/"run_summary.pdf", run=run)
            manifest["pdf_status"] = "complete"
        except Exception as exc:
            manifest["pdf_status"], manifest["pdf_error"] = "failed", str(exc)
            write_json(run_dir/"logs/report.error.json", dict(error=str(exc), traceback=traceback.format_exc()))
    else:
        manifest["pdf_status"] = "disabled"
    if config["outputs"].get("measurements_h5ad", True):
        try:
            from .outputs.measurements import make_measurements_h5ad
            measurements_path = timer.call("write_measurements_h5ad", make_measurements_h5ad,
                                           run_dir/"myonis_4i_data.h5ad")
            manifest["measurements_h5ad_status"] = "complete"
            manifest["measurements_h5ad_file"] = measurements_path.name
            manifest["measurements_h5ad_bytes"] = measurements_path.stat().st_size
            error_path = run_dir/"logs/measurements_h5ad.error.json"
            if error_path.exists():
                error_path.unlink()
        except Exception as exc:
            manifest["measurements_h5ad_status"], manifest["measurements_h5ad_error"] = "failed", str(exc)
            write_json(run_dir/"logs/measurements_h5ad.error.json",
                       dict(error=str(exc), traceback=traceback.format_exc()))
    else:
        manifest["measurements_h5ad_status"] = "disabled"
    manifest.pop("timing_summary", None)
    if timer.enabled:
        pd.DataFrame(timings+timer.rows()).to_csv(run_dir/"logs/timings.csv", index=False)
        timer.report()
        counts = dict(execution_counts or {})
        counts["failed"] = max(counts.get("failed", 0), len(failures))
        manifest["timing_summary"] = summarize_timing(
            field_timings, len(parts),
            time.perf_counter()-started_perf_counter if started_perf_counter is not None else None, counts)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir/"run_manifest.json", manifest)
    error_path = run_dir/"logs/finalize.error.json"
    if error_path.exists():
        error_path.unlink()
    if timer.enabled:
        report_timing_summary(manifest["timing_summary"])
    print(f"{status}: nuclei={run.n_obs}, myotubes={len(run.uns['myotubes'])}; {run_dir/'myonis_4i_data.h5ad'}", flush=True)
    return manifest


def run_prepared_tasks(run_dir, task_indices, workers):
    """Use the same field executor locally and within each independent batch job."""
    run_dir = Path(run_dir)
    tasks = json.loads((run_dir/"tasks.json").read_text())
    task_indices = list(task_indices)
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(set(task_indices)) != len(task_indices) or any(i < 0 or i >= len(tasks) for i in task_indices):
        raise ValueError("Task indices must be unique and within the prepared task list")
    workers = min(workers, max(1, len(task_indices)))
    execution_counts = dict(complete=0, resumed=0, failed=0)
    if workers == 1:
        for i in progress(task_indices, len(task_indices), "Quantify fields"):
            task = tasks[i]
            try:
                result = execute_task(run_dir, i)
                execution_counts[result["status"]] += 1
            except Exception as exc:
                execution_counts["failed"] += 1
                print(f"[{i+1}/{len(tasks)}] FAILED {task['field_key']}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(execute_task, str(run_dir), i): tasks[i] for i in task_indices}
            for future in progress(as_completed(futures), len(futures), "Quantify fields"):
                task = futures[future]
                try:
                    result = future.result()
                    execution_counts[result["status"]] += 1
                except Exception as exc:
                    execution_counts["failed"] += 1
                    print(f"FAILED {task['field_key']}: {exc}", flush=True)
    return execution_counts


def run_pipeline(config_path, run_dir=None, resume=False, workers=None, troubleshoot=False,
                 troubleshoot_timing=False, visualisation_fields=None):
    started = time.perf_counter()
    run_dir = prepare(config_path, run_dir, resume, workers, troubleshoot, troubleshoot_timing, visualisation_fields)
    config = json.loads((run_dir/"config/resolved.json").read_text())
    tasks = json.loads((run_dir/"tasks.json").read_text())
    execution_counts = run_prepared_tasks(run_dir, range(len(tasks)), int(config["execution"]["workers"]))
    return run_dir, finalize(run_dir, started_perf_counter=started, execution_counts=execution_counts)
