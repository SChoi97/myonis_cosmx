#!/usr/bin/env python3
"""Diagnose scVI preprocessing, tensors, and first-batch CUDA failures."""

import argparse
import gc
import os
import platform
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import scipy.sparse as sp
import torch
from scvi import REGISTRY_KEYS
from scvi.model import SCVI

import scvi_main

try:
    from scvi.module._constants import MODULE_KEYS
except Exception:
    MODULE_KEYS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the MYONIS scVI preprocessing path, then probe the exact tensors "
            "and first-batch model operations that can fail on CUDA."
        )
    )
    parser.add_argument(
        "--h5ad-paths",
        "--h5ad-path",
        nargs="+",
        required=True,
        dest="h5ad_paths",
        help="One or more input .h5ad files.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["auto", "nuclei", "myotubes"],
        default="auto",
    )
    parser.add_argument(
        "--is-myonucleus",
        "--is_myonucleus",
        type=scvi_main.parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--covariates-to-remove",
        nargs="*",
        default=[],
        help="Same meaning as scvi_main.py.",
    )
    parser.add_argument("--counts-layer", default=None)
    parser.add_argument("--var-join", choices=["inner", "outer"], default="inner")
    parser.add_argument("--min-counts", type=float, default=100.0)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--control-probe-pattern", default="SystemControl|Negative")
    parser.add_argument("--min-nuclei", type=int, default=1)
    parser.add_argument("--nuclei-h5ad-paths", nargs="*", default=None)

    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--cuda-lgamma-mode",
        choices=["auto", "torch", "stirling"],
        default="auto",
        help="Same meaning as scvi_main.py.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--continuous-covariate-transform",
        choices=["none", "zscore", "log1p-zscore"],
        default="log1p-zscore",
    )
    parser.add_argument("--skip-count-validation", action="store_true")
    parser.add_argument("--no-cast-counts-float32", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=["high", "medium", "default"],
        default="high",
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")

    parser.add_argument(
        "--probe-likelihoods",
        nargs="+",
        choices=["zinb", "nb", "poisson"],
        default=["nb", "zinb"],
        help="scVI likelihoods to probe on the first batch.",
    )
    parser.add_argument(
        "--also-test-no-covariates",
        action="store_true",
        help="Repeat first-batch probes without registered covariates.",
    )
    parser.add_argument(
        "--n-probe-obs",
        type=int,
        default=4096,
        help="Number of observations used to build deterministic probe batches.",
    )
    parser.add_argument(
        "--max-probe-batches",
        type=int,
        default=2,
        help="Number of deterministic batches to inspect before model probes.",
    )
    return parser.parse_args()


def section(title):
    print(f"\n[diag] === {title} ===", flush=True)


def kv(key, value):
    print(f"[diag] {key}: {value}", flush=True)


def short_list(values, max_items=12):
    values = list(values)
    if len(values) <= max_items:
        return values
    return values[:max_items] + [f"... ({len(values) - max_items} more)"]


def quantile_summary(values):
    values = np.asarray(values)
    if values.size == 0:
        return "empty"
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "all_nonfinite"
    qs = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    names = ["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]
    return ", ".join(f"{name}={value:.6g}" for name, value in zip(names, qs))


def as_numeric_array(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.float64, copy=False)

    flat = pd.to_numeric(pd.Series(values.reshape(-1)), errors="coerce")
    return flat.to_numpy(dtype=np.float64).reshape(values.shape)


def numpy_stats(name, values):
    values = np.asarray(values)
    kv(f"{name}.shape", values.shape)
    kv(f"{name}.dtype", values.dtype)
    if values.size == 0:
        kv(f"{name}.stats", "empty")
        return
    numeric_values = as_numeric_array(values)
    finite = np.isfinite(numeric_values)
    kv(f"{name}.finite", f"{int(finite.sum())}/{values.size}")
    kv(f"{name}.nan", int(np.isnan(numeric_values).sum()))
    kv(f"{name}.inf", int(np.isinf(numeric_values).sum()))
    kv(f"{name}.negative", int((numeric_values < 0).sum()))
    kv(f"{name}.quantiles", quantile_summary(numeric_values))


def tensor_stats(name, tensor):
    if not torch.is_tensor(tensor):
        kv(name, f"{type(tensor).__name__}: {tensor}")
        return

    kv(f"{name}.shape", tuple(tensor.shape))
    kv(f"{name}.dtype", tensor.dtype)
    kv(f"{name}.device", tensor.device)
    kv(f"{name}.requires_grad", tensor.requires_grad)
    kv(f"{name}.is_sparse", tensor.is_sparse)
    kv(f"{name}.is_contiguous", tensor.is_contiguous() if not tensor.is_sparse else "n/a")
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        values = coalesced.values().detach()
        kv(f"{name}.nnz", int(coalesced._nnz()))
    else:
        values = tensor.detach().reshape(-1)
    if values.numel() == 0:
        kv(f"{name}.stats", "empty")
        return

    with torch.no_grad():
        values_cpu = values.float().cpu()
        kv(f"{name}.finite", f"{int(torch.isfinite(values_cpu).sum())}/{values_cpu.numel()}")
        kv(f"{name}.nan", int(torch.isnan(values_cpu).sum()))
        kv(f"{name}.inf", int(torch.isinf(values_cpu).sum()))
        kv(f"{name}.negative", int((values_cpu < 0).sum()))
        numpy_stats(f"{name}.values", values_cpu.numpy())


def summarize_batch(batch, prefix="batch"):
    if isinstance(batch, dict):
        kv(f"{prefix}.keys", list(batch.keys()))
        for key, value in batch.items():
            summarize_batch(value, f"{prefix}.{key}")
    elif isinstance(batch, (list, tuple)):
        kv(f"{prefix}.len", len(batch))
        for idx, value in enumerate(batch):
            summarize_batch(value, f"{prefix}[{idx}]")
    else:
        tensor_stats(prefix, batch)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def synchronize_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def run_probe(label, fn, device=None):
    print(f"[diag] BEGIN {label}", flush=True)
    try:
        out = fn()
        if device is not None:
            synchronize_if_cuda(device)
        print(f"[diag] PASS {label}", flush=True)
        return out, None
    except Exception as exc:
        print(f"[diag] FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return None, exc


def environment_report():
    section("Environment")
    kv("python", sys.version.replace("\n", " "))
    kv("executable", sys.executable)
    kv("platform", platform.platform())
    kv("cwd", os.getcwd())
    kv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    kv("CUDA_LAUNCH_BLOCKING", os.environ.get("CUDA_LAUNCH_BLOCKING", "<unset>"))
    kv("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID", "<unset>"))
    kv("SLURM_JOB_NODELIST", os.environ.get("SLURM_JOB_NODELIST", "<unset>"))
    kv("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_PER_TASK", "<unset>"))
    kv("anndata", ad.__version__)
    kv("numpy", np.__version__)
    kv("pandas", pd.__version__)
    kv("scipy", getattr(sp, "__version__", "<unknown>"))
    kv("scvi", scvi.__version__)
    kv("torch", torch.__version__)
    kv("torch.version.cuda", torch.version.cuda)
    kv("torch.cuda.is_available", torch.cuda.is_available())
    kv("torch.cuda.device_count", torch.cuda.device_count())
    kv("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    kv("torch.backends.cudnn.allow_tf32", torch.backends.cudnn.allow_tf32)
    kv("torch.backends.cudnn.benchmark", torch.backends.cudnn.benchmark)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            kv(f"cuda[{idx}].name", props.name)
            kv(f"cuda[{idx}].capability", f"{props.major}.{props.minor}")
            kv(f"cuda[{idx}].total_memory_gb", f"{props.total_memory / 1024 ** 3:.2f}")


def matrix_report(adata, counts_layer):
    section("AnnData and Count Matrix")
    matrix = scvi_main.get_matrix(adata, counts_layer)
    kv("adata.shape", adata.shape)
    kv("adata.obs_names_unique", adata.obs_names.is_unique)
    kv("adata.var_names_unique", adata.var_names.is_unique)
    kv("adata.obs_columns", short_list(adata.obs.columns))
    kv("adata.var_columns", short_list(adata.var.columns))
    kv("adata.layers", list(adata.layers.keys()))
    kv("adata.uns_keys", short_list(adata.uns.keys()))
    kv("X.type", type(matrix).__name__)
    kv("X.dtype", getattr(matrix, "dtype", None))
    kv("X.shape", getattr(matrix, "shape", None))
    kv("X.sparse", sp.issparse(matrix))
    if sp.issparse(matrix):
        kv("X.format", matrix.getformat())
        kv("X.nnz", int(matrix.nnz))
        numpy_stats("X.data", matrix.data)
    else:
        values = np.asarray(matrix)
        numpy_stats("X", values)
        kv("X.nnz", int(np.count_nonzero(values)))

    row_totals = scvi_main.row_sums(matrix)
    numpy_stats("row_total_counts", row_totals)
    nonzero_by_gene = scvi_main.col_nonzero_counts(matrix)
    numpy_stats("gene_nonzero_counts", nonzero_by_gene)

    top_idx = np.argsort(row_totals)[-10:][::-1]
    cols = [
        col
        for col in (
            "source_h5ad",
            "Slide Name",
            "Cell Line",
            "field",
            "patch_idx",
            "local_id",
            "n_nuclei",
            "n_counts",
        )
        if col in adata.obs.columns
    ]
    top = adata.obs.iloc[top_idx][cols].copy() if cols else pd.DataFrame(index=adata.obs.index[top_idx])
    top.insert(0, "row_total_counts", row_totals[top_idx])
    kv("top_row_total_counts", top.to_string())

    for col in ("source_h5ad", "Slide Name", "Cell Line", "n_nuclei", "n_counts"):
        if col not in adata.obs.columns:
            continue
        series = adata.obs[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numpy_stats(f"obs[{col}]", pd.to_numeric(series, errors="coerce").to_numpy())
        else:
            counts = series.astype("string").fillna("missing").value_counts(dropna=False).head(20)
            kv(f"obs[{col}].top_values", counts.to_string())


def build_training_adata(args):
    section("Preprocessing")
    scvi_main.configure_runtime(args)
    if args.use_gpu:
        scvi_main.check_gpu_or_raise()
        scvi_main.configure_cuda_lgamma(args.cuda_lgamma_mode)

    covariates = scvi_main.flatten_covariates(args.covariates_to_remove)
    kv("requested_covariates", covariates)
    adata, input_mode, summaries = scvi_main.load_h5ads(
        args.h5ad_paths,
        requested_mode=args.input_mode,
        is_myonucleus=args.is_myonucleus,
        var_join=args.var_join,
    )
    adata, filter_summary = scvi_main.apply_standard_filters(
        adata,
        counts_layer=args.counts_layer,
        min_counts=args.min_counts,
        min_cells=args.min_cells,
        control_probe_pattern=args.control_probe_pattern,
    )

    nuclei_sources = []
    nuclei_filter_summary = None
    nuclei_edge_summary = None
    myotube_assignment_summary = None
    if input_mode == "myotubes" and args.min_nuclei and args.min_nuclei > 0:
        nuclei_adata, nuclei_sources = scvi_main.load_paired_nuclei_h5ads(
            args.h5ad_paths,
            args.nuclei_h5ad_paths,
            args.var_join,
        )
        nuclei_adata, nuclei_edge_summary = scvi_main.filter_nuclei_edges_for_matching(nuclei_adata)
        nuclei_adata, nuclei_filter_summary = scvi_main.apply_standard_filters(
            nuclei_adata,
            counts_layer=args.counts_layer,
            min_counts=args.min_counts,
            min_cells=args.min_cells,
            control_probe_pattern=args.control_probe_pattern,
        )
        nuclei_counts, nuclei_source, myotube_assignment_summary = (
            scvi_main.compute_myotube_nucleus_counts_from_nuclei(
                adata,
                nuclei_adata,
                ", ".join(nuclei_sources),
            )
        )
        adata, _ = scvi_main.apply_min_nuclei_filter(
            adata,
            nuclei_counts,
            nuclei_source,
            args.min_nuclei,
        )
        del nuclei_adata
        gc.collect()

    if not args.skip_count_validation:
        scvi_main.validate_count_matrix(adata, args.counts_layer)
    if not args.no_cast_counts_float32:
        scvi_main.cast_count_matrix_to_float32(adata, args.counts_layer)

    continuous_keys, categorical_keys, covariate_transform_params = scvi_main.prepare_covariates(
        adata,
        covariates,
        args.counts_layer,
        continuous_covariate_transform=args.continuous_covariate_transform,
    )

    kv("input_mode", input_mode)
    kv("load_summaries", summaries)
    kv("filter_summary", filter_summary)
    kv("nuclei_sources", nuclei_sources)
    kv("nuclei_edge_summary", nuclei_edge_summary)
    kv("nuclei_filter_summary", nuclei_filter_summary)
    kv("myotube_assignment_summary", myotube_assignment_summary)
    kv("continuous_covariate_keys", continuous_keys)
    kv("categorical_covariate_keys", categorical_keys)
    kv("covariate_transform_params", covariate_transform_params)
    return adata, input_mode, continuous_keys, categorical_keys


def instantiate_model(adata, args, continuous_keys, categorical_keys, gene_likelihood):
    SCVI.setup_anndata(
        adata,
        layer=args.counts_layer,
        continuous_covariate_keys=continuous_keys or None,
        categorical_covariate_keys=categorical_keys or None,
    )
    model = SCVI(
        adata,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        gene_likelihood=gene_likelihood,
        dispersion="gene",
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
        dropout_rate=args.dropout_rate,
    )
    return model


def parameter_report(model, label):
    section(f"Model Parameters: {label}")
    n_total = 0
    n_nonfinite = 0
    for name, param in model.module.named_parameters():
        n_total += param.numel()
        with torch.no_grad():
            values = param.detach().float().cpu()
            bad = int((~torch.isfinite(values)).sum())
            n_nonfinite += bad
            if bad or name in {"px_r"} or "px_r" in name:
                tensor_stats(f"param[{name}]", values)
    kv("parameter_total", n_total)
    kv("parameter_nonfinite", n_nonfinite)


def make_probe_batches(model, adata, args):
    n_probe = min(int(args.n_probe_obs), int(adata.n_obs))
    indices = np.arange(n_probe)
    loader = model._make_data_loader(
        adata,
        indices=indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    batches = []
    for idx, batch in enumerate(loader):
        section(f"DataLoader Batch {idx}")
        summarize_batch(batch, prefix=f"batch[{idx}]")
        batches.append(batch)
        if idx + 1 >= args.max_probe_batches:
            break
    if not batches:
        raise RuntimeError("No batches produced by scVI AnnDataLoader.")
    return batches


def get_x_from_batch(batch):
    x_key = getattr(REGISTRY_KEYS, "X_KEY", "X")
    if x_key in batch:
        return batch[x_key]
    if "X" in batch:
        return batch["X"]
    raise KeyError(f"No X tensor found in batch keys: {list(batch.keys())}")


def find_px_distribution(generative_outputs):
    if MODULE_KEYS is not None:
        px_key = getattr(MODULE_KEYS, "PX_KEY", None)
        if px_key in generative_outputs:
            return px_key, generative_outputs[px_key]
    for key, value in generative_outputs.items():
        if hasattr(value, "log_prob"):
            return key, value
    raise KeyError(f"No distribution-like object found in generative outputs: {list(generative_outputs.keys())}")


def distribution_report(px, prefix):
    kv(f"{prefix}.class", type(px).__name__)
    for attr in ("mu", "theta", "zi_logits", "scale", "rate", "logits", "total_count", "probs"):
        if hasattr(px, attr):
            value = getattr(px, attr)
            if torch.is_tensor(value):
                tensor_stats(f"{prefix}.{attr}", value)
            else:
                kv(f"{prefix}.{attr}", value)


def direct_lgamma_tests(batch, device):
    def _run():
        x = get_x_from_batch(batch).to(device)
        tensor_stats("direct_lgamma.x", x)
        samples = torch.tensor(
            [0.0, 1.0, 2.0, 10.0, 118.0, 4100.0, 119316.0],
            device=device,
            dtype=torch.float32,
        )
        tensor_stats("direct_lgamma.sample_input", samples)
        tensor_stats("direct_lgamma.safe_sample_output", scvi_main.cuda_safe_lgamma_positive(samples))
        tensor_stats("direct_lgamma.safe_x_plus_1", scvi_main.cuda_safe_lgamma_positive(x + 1.0))
        tensor_stats("direct_lgamma.sample_output", torch.lgamma(samples))
        tensor_stats("direct_lgamma.x_plus_1", torch.lgamma(x + 1.0))
        tensor_stats("direct_lgamma.x_contiguous_plus_1", torch.lgamma(x.contiguous() + 1.0))
        theta_gene = torch.ones((x.shape[1],), device=device, dtype=x.dtype)
        tensor_stats("direct_lgamma.x_plus_gene_theta", torch.lgamma(x + theta_gene))
        theta_full = torch.ones_like(x)
        tensor_stats("direct_lgamma.x_plus_full_theta", torch.lgamma(x + theta_full))
        return True

    return run_probe(f"direct torch.lgamma tests on {device}", _run, device=device)


def forward_tests(model, batch, device, label):
    def _run_compute_loss_false():
        model.module.to(device)
        model.module.eval()
        batch_device = move_to_device(batch, device)
        with torch.no_grad():
            inference_outputs, generative_outputs = model.module(
                batch_device,
                compute_loss=False,
            )
        kv(f"{label}.inference_keys", list(inference_outputs.keys()))
        kv(f"{label}.generative_keys", list(generative_outputs.keys()))
        for key, value in inference_outputs.items():
            if torch.is_tensor(value):
                tensor_stats(f"{label}.inference[{key}]", value)
        px_key, px = find_px_distribution(generative_outputs)
        kv(f"{label}.px_key", px_key)
        distribution_report(px, f"{label}.px")
        return batch_device, inference_outputs, generative_outputs, px

    outputs, exc = run_probe(
        f"{label}: module forward compute_loss=False on {device}",
        _run_compute_loss_false,
        device=device,
    )
    if exc is not None:
        return

    batch_device, inference_outputs, generative_outputs, px = outputs

    def _run_direct_model_lgamma():
        x = get_x_from_batch(batch_device)
        theta = getattr(px, "theta", None)
        if theta is None:
            kv(f"{label}.theta_lgamma", "skipped; distribution has no theta")
            return True
        theta_for_broadcast = theta.view(1, theta.size(0)) if theta.ndim == 1 else theta
        tensor_stats(f"{label}.x_plus_theta", x + theta_for_broadcast)
        tensor_stats(
            f"{label}.safe_lgamma_x_plus_theta",
            scvi_main.cuda_safe_lgamma_positive(x + theta_for_broadcast),
        )
        tensor_stats(f"{label}.safe_lgamma_theta", scvi_main.cuda_safe_lgamma_positive(theta_for_broadcast))
        tensor_stats(f"{label}.safe_lgamma_x_plus_1", scvi_main.cuda_safe_lgamma_positive(x + 1.0))
        tensor_stats(f"{label}.lgamma_x_plus_theta", torch.lgamma(x + theta_for_broadcast))
        tensor_stats(f"{label}.lgamma_theta", torch.lgamma(theta_for_broadcast))
        tensor_stats(f"{label}.lgamma_x_plus_1", torch.lgamma(x + 1.0))
        return True

    run_probe(
        f"{label}: direct lgamma with model theta on {device}",
        _run_direct_model_lgamma,
        device=device,
    )

    def _run_px_log_prob():
        x = get_x_from_batch(batch_device)
        log_prob = px.log_prob(x)
        tensor_stats(f"{label}.px_log_prob", log_prob)
        tensor_stats(f"{label}.reconstruction_loss", -log_prob.sum(-1))
        return True

    run_probe(f"{label}: px.log_prob(x) on {device}", _run_px_log_prob, device=device)

    def _run_compute_loss_true():
        with torch.no_grad():
            _, _, losses = model.module(batch_device, compute_loss=True)
        if hasattr(losses, "loss"):
            tensor_stats(f"{label}.loss", losses.loss)
        else:
            kv(f"{label}.losses", losses)
        return True

    run_probe(f"{label}: module forward compute_loss=True on {device}", _run_compute_loss_true, device=device)


def probe_model_variant(adata, args, continuous_keys, categorical_keys, covariate_label, gene_likelihood):
    label = f"{covariate_label}, likelihood={gene_likelihood}"
    section(f"Probe Variant: {label}")
    model = instantiate_model(
        adata,
        args,
        continuous_keys=continuous_keys,
        categorical_keys=categorical_keys,
        gene_likelihood=gene_likelihood,
    )
    parameter_report(model, label)
    batches = make_probe_batches(model, adata, args)
    first_batch = batches[0]

    forward_tests(model, first_batch, torch.device("cpu"), f"{label}, cpu")
    if args.use_gpu:
        direct_lgamma_tests(first_batch, torch.device("cuda:0"))
        forward_tests(model, first_batch, torch.device("cuda:0"), f"{label}, cuda")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    environment_report()
    adata, input_mode, continuous_keys, categorical_keys = build_training_adata(args)
    matrix_report(adata, args.counts_layer)

    print(
        f"[diag] Final matrix for probes: n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
        f"mode={input_mode}, continuous_covariates={continuous_keys}, "
        f"categorical_covariates={categorical_keys}",
        flush=True,
    )

    for gene_likelihood in args.probe_likelihoods:
        probe_model_variant(
            adata,
            args,
            continuous_keys=continuous_keys,
            categorical_keys=categorical_keys,
            covariate_label="with_covariates",
            gene_likelihood=gene_likelihood,
        )

    if args.also_test_no_covariates:
        for gene_likelihood in args.probe_likelihoods:
            probe_model_variant(
                adata,
                args,
                continuous_keys=[],
                categorical_keys=[],
                covariate_label="without_covariates",
                gene_likelihood=gene_likelihood,
            )

    section("Diagnostic Complete")
    print("[diag] The diagnostic finished without an unhandled exception.", flush=True)


if __name__ == "__main__":
    main()
