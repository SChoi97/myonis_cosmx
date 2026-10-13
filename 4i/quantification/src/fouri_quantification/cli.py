from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Bound native thread pools before importing numerical libraries in worker processes.
for _variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_variable, "1")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MYONIS 4i: linked nuclei, myotubes and substructures")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Index inputs and show missing/incompatible assets")
    validate.add_argument("--config", required=True)
    for name in ["run", "prepare", "submit-batch"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--run-dir")
        if name != "submit-batch":
            sub.add_argument("--resume", action="store_true")
        sub.add_argument("--workers", type=int, required=name == "submit-batch")
        if name == "submit-batch":
            sub.add_argument("--jobs", type=int, required=True, help="Maximum simultaneous batch jobs")
            sub.add_argument("--partition", default="ncpu")
            sub.add_argument("--log-dir", help="Folder for worker/finalisation Slurm logs; default: SLURM_SUBMIT_DIR or current directory")
            sub.add_argument("--memory", default="32G", help="Memory per batch job, not per worker")
            sub.add_argument("--time", default="24:00:00", help="Time limit per batch job")
            sub.add_argument("--finalize-workers", type=int, default=4, help="Parallel checkpoint readers in the final assembly job")
            sub.add_argument("--require-complete", action="store_true", help="Refuse partial final output; default assembles valid fields with exclusions")
            sub.add_argument("--finalize-memory", default="64G")
            sub.add_argument("--finalize-time", default="04:00:00")
        sub.add_argument("--troubleshoot", "--optimise", action="store_true",
                         help="Detailed QC figures, all marker/patch views and audits for sampled visualisation fields; includes timing")
        sub.add_argument("--troubleshoot_timing", "--troubleshoot-timing", action="store_true",
                         help="Print/save stage timings without enabling extra visualisations")
        sub.add_argument("--visualisation_fields", "--visualisation-fields", type=int,
                         help="Random field/cohort visualisation limit, independent of quantification; 0 disables figures")
    worker = commands.add_parser("worker", help="Run one prepared Slurm array task")
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--task-index", required=True, type=int)
    batch = commands.add_parser("worker-batch", help="Run one group of prepared fields using several workers")
    batch.add_argument("--run-dir", required=True)
    batch.add_argument("--batch-index", required=True, type=int)
    batch.add_argument("--batch-count", required=True, type=int)
    batch.add_argument("--workers", required=True, type=int)
    final = commands.add_parser("finalize", help="Combine completed tasks and generate report")
    final.add_argument("--run-dir", required=True)
    policy = final.add_mutually_exclusive_group()
    policy.add_argument("--require-complete", action="store_true", help="Refuse output if any field checkpoint is missing or invalid")
    policy.add_argument("--allow-partial", action="store_true", help="Assemble valid fields with explicit exclusions; a partial archive is a successful finalisation")
    final.add_argument("--workers", type=int, default=1, help="Parallel checkpoint readers; match the Slurm CPU allocation")
    final.add_argument("--scratch-dir", help="Existing local temporary directory; default SLURM_TMPDIR or system temporary directory")
    final.add_argument("--allow-finalizer-update", action="store_true", help="Allow assembly/reporting code updates for old checkpoints; measurement code and versions must still match")
    report = commands.add_parser("report", help="Regenerate the PDF from a saved H5AD")
    report.add_argument("--input", required=True)
    report.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            from .config import load_config
            from .inputs.manifests import build_index
            config = load_config(args.config)
            if config["visualisation"]["enabled"] or config.get("registration_qc", {}).get("enabled"):
                from .reporting.visualisation import renderer_identity
                print("QC renderer:", renderer_identity())
            index = build_index(config)
            print(json.dumps(dict(images=len(index["images"]), field_cohorts=index["images"].field_key.nunique(),
                                  assets=len(index["assets"]), markers=[m["key"] for m in index["markers"]],
                                  field_selection=index["field_selection"]), indent=2))
            if len(index["issues"]):
                print(index["issues"].groupby(["severity", "source", "status"]).size().to_string())
                print("First input issues:")
                print(index["issues"].head(12).to_string(index=False))
            return 1 if (index["issues"].severity == "error").any() else 0
        if args.command in {"run", "prepare"}:
            from .pipeline import run_pipeline, prepare
            if args.command == "prepare":
                print(prepare(args.config, args.run_dir, args.resume, args.workers,
                              args.troubleshoot, args.troubleshoot_timing, args.visualisation_fields))
                return 0
            _, result = run_pipeline(args.config, args.run_dir, args.resume, args.workers,
                                     args.troubleshoot, args.troubleshoot_timing, args.visualisation_fields)
            return 1 if result["status"] in {"partial", "failed"} or result.get("pdf_status") == "failed" or result.get("visualisation_status") == "failed" or result.get("measurements_h5ad_status") == "failed" else 0
        if args.command == "worker":
            from .pipeline import execute_task
            print(execute_task(args.run_dir, args.task_index)); return 0
        if args.command == "submit-batch":
            from .slurm import submit_batch
            submit_batch(args.config, args.jobs, args.workers, args.partition, args.memory, args.time,
                         args.finalize_memory, args.finalize_time, args.run_dir,
                         args.troubleshoot, args.troubleshoot_timing, args.visualisation_fields, log_dir=args.log_dir,
                         finalize_workers=args.finalize_workers, require_complete=args.require_complete)
            return 0
        if args.command == "worker-batch":
            from .slurm import run_batch
            result = run_batch(args.run_dir, args.batch_index, args.batch_count, args.workers)
            return 1 if result["failed"] else 0
        if args.command == "finalize":
            from .pipeline import finalize
            result = finalize(args.run_dir, require_complete=args.require_complete, workers=args.workers,
                              scratch_dir=args.scratch_dir, allow_finalizer_update=args.allow_finalizer_update)
            return 1 if (result["status"] == "failed" or (result["status"] == "partial" and not args.allow_partial)) or result.get("pdf_status") == "failed" or result.get("visualisation_status") == "failed" or result.get("measurements_h5ad_status") == "failed" else 0
        if args.command == "report":
            from .reporting.pdf import build_report
            print(build_report(args.input, args.output or Path(args.input).with_name("run_summary.pdf")))
            return 0
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
