#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(Matrix)
})

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else if (!is.null(sys.frames()[[1]]$ofile)) {
    dirname(normalizePath(sys.frames()[[1]]$ofile))
  } else {
    getwd()
  }
})

source(file.path(script_dir, "utils", "preprocessing_utils.R"))

# ---------------- USER INPUTS ----------------
# DEG_PATH can point to either:
#   1. the DEG output directory from run_deg_myotubes.R, or
#   2. myotube_deg_results.rds / myotube_deg_results.RData directly.
DEG_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/deg/archive"

# Optional. If available, the script reproduces the per-cell-line filtering counts
# from run_deg_myotubes.R so the summary includes myotube counts before/after filters.
INPUT_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/archive/processed_myotube_filtered.rds"
USE_INPUT_OBJECT <- TRUE

area_col_candidates <- c("area_px2", "Area", "area", "area_um2", "myotube_area", "cell_area")

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Supports:
#   --KEY value
#   --key=value
#   key=value
# Usage example:
# Rscript generate_deg_myotube_summary.R \
#   --DEG_PATH /path/to/deg \
#   --INPUT_PATH /path/to/processed_myotube_filtered.rds
overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
input_path_set_null <- FALSE
if ("INPUT_PATH" %in% names(overrides)) {
  raw_input <- tolower(trimws(overrides[["INPUT_PATH"]]))
  if (raw_input %in% c("null", "none", "na", "")) {
    input_path_set_null <- TRUE
  }
}
apply_cli_overrides(overrides)
if (input_path_set_null) {
  INPUT_PATH <- NULL
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

fmt_value <- function(x) {
  if (is.null(x)) return("NULL")
  if (length(x) == 0) return("<empty>")
  if (length(x) == 1) {
    if (is.logical(x)) return(ifelse(isTRUE(x), "TRUE", "FALSE"))
    return(as.character(x))
  }
  paste(as.character(x), collapse = ", ")
}

resolve_deg_result_path <- function(path) {
  if (is.null(path) || !nzchar(path)) {
    stop("DEG_PATH is required.")
  }

  if (dir.exists(path)) {
    candidates <- c(
      file.path(path, "myotube_deg_results.rds"),
      file.path(path, "myotube_deg_results.RData")
    )
    hits <- candidates[file.exists(candidates)]
    if (length(hits) == 0) {
      stop("No myotube DEG result file found in directory: ", path)
    }
    return(normalizePath(hits[1]))
  }

  if (file.exists(path)) {
    return(normalizePath(path))
  }

  stop("DEG_PATH not found: ", path)
}

extract_results_by_cell_line <- function(obj, source_label) {
  if (is.list(obj) && "results_by_cell_line" %in% names(obj)) {
    return(obj$results_by_cell_line)
  }

  if (
    is.list(obj) &&
    length(obj) > 0 &&
    !is.null(names(obj)) &&
    all(vapply(obj, function(x) {
      is.data.frame(x) || is.matrix(x) || inherits(x, "tbl_df")
    }, logical(1)))
  ) {
    return(obj)
  }

  stop("Could not extract `results_by_cell_line` from ", source_label)
}

load_deg_bundle <- function(path) {
  resolved <- resolve_deg_result_path(path)
  ext <- tolower(tools::file_ext(resolved))

  if (ext == "rds") {
    obj <- readRDS(resolved)
    return(list(
      source_path = resolved,
      source_format = "rds",
      stored_fields = names(obj),
      raw = obj,
      results_by_cell_line = extract_results_by_cell_line(obj, resolved),
      sig_genes_by_cell_line = obj[["sig_genes_by_cell_line"]],
      sig_genes_up_abnormal = obj[["sig_genes_up_abnormal"]],
      sig_genes_down_abnormal = obj[["sig_genes_down_abnormal"]],
      min_expr_cells = obj[["min_expr_cells"]],
      alpha_padj = obj[["alpha_padj"]],
      lfc_thr = obj[["lfc_thr"]],
      eps = obj[["eps"]],
      area_threshold = if ("area_threshold" %in% names(obj)) obj[["area_threshold"]] else NULL,
      y_max_clip = obj[["y_max_clip"]],
      n_labels = obj[["n_labels"]]
    ))
  }

  if (ext == "rdata") {
    env <- new.env(parent = emptyenv())
    load(resolved, envir = env)
    get_or_null <- function(name) {
      if (exists(name, envir = env, inherits = FALSE)) {
        get(name, envir = env, inherits = FALSE)
      } else {
        NULL
      }
    }

    raw_obj <- as.list(env)
    return(list(
      source_path = resolved,
      source_format = "RData",
      stored_fields = ls(env, all.names = TRUE),
      raw = raw_obj,
      results_by_cell_line = get_or_null("results_by_cell_line"),
      sig_genes_by_cell_line = get_or_null("sig_genes_by_cell_line"),
      sig_genes_up_abnormal = get_or_null("sig_genes_up_abnormal"),
      sig_genes_down_abnormal = get_or_null("sig_genes_down_abnormal"),
      min_expr_cells = get_or_null("min_expr_cells"),
      alpha_padj = get_or_null("alpha_padj"),
      lfc_thr = get_or_null("lfc_thr"),
      eps = get_or_null("eps"),
      area_threshold = if ("area_threshold" %in% ls(env, all.names = TRUE)) get_or_null("area_threshold") else NULL,
      y_max_clip = get_or_null("y_max_clip"),
      n_labels = get_or_null("n_labels")
    ))
  }

  stop("Unsupported DEG result format: ", resolved)
}

ensure_morphology_class <- function(adata) {
  cd0 <- as.data.frame(colData(adata))
  if ("morphology_class" %in% colnames(cd0)) {
    return(adata)
  }

  if (!all(c("n_normal_nuclei", "n_abnormal_nuclei") %in% colnames(cd0))) {
    stop("morphology_class missing and cannot be derived (n_normal_nuclei / n_abnormal_nuclei not found).")
  }

  morph <- rep(-1L, nrow(cd0))
  morph[cd0$n_normal_nuclei > 0 & cd0$n_abnormal_nuclei == 0] <- 1L
  morph[cd0$n_normal_nuclei == 0 & cd0$n_abnormal_nuclei > 0] <- 2L
  morph[cd0$n_normal_nuclei > 0 & cd0$n_abnormal_nuclei > 0] <- 3L
  colData(adata)$morphology_class <- morph
  adata
}

prepare_input_context <- function(input_path, use_input_object, area_threshold) {
  if (!isTRUE(use_input_object)) {
    return(list(ok = FALSE, reason = "USE_INPUT_OBJECT=FALSE"))
  }

  if (is.null(input_path) || !nzchar(input_path)) {
    return(list(ok = FALSE, reason = "INPUT_PATH not set"))
  }

  if (!file.exists(input_path)) {
    return(list(ok = FALSE, reason = paste0("Input file not found: ", input_path)))
  }

  adata <- tryCatch(readRDS(input_path), error = function(e) e)
  if (inherits(adata, "error")) {
    return(list(ok = FALSE, reason = paste0("Failed to read INPUT_PATH: ", conditionMessage(adata))))
  }

  adata <- tryCatch(ensure_morphology_class(adata), error = function(e) e)
  if (inherits(adata, "error")) {
    return(list(ok = FALSE, reason = conditionMessage(adata)))
  }

  cd0 <- as.data.frame(colData(adata))
  cell_line_col <- tryCatch(
    pick_col(cd0, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), "Cell Line"),
    error = function(e) e
  )
  if (inherits(cell_line_col, "error")) {
    return(list(ok = FALSE, reason = conditionMessage(cell_line_col)))
  }

  area_col <- tryCatch(
    pick_col(cd0, area_col_candidates, "Area", required = FALSE),
    error = function(e) e
  )
  if (inherits(area_col, "error")) {
    return(list(ok = FALSE, reason = conditionMessage(area_col)))
  }

  if (!is.null(area_threshold) && is.na(area_col)) {
    return(list(
      ok = FALSE,
      reason = "area_threshold is set in DEG results but no compatible area column was found in INPUT_PATH"
    ))
  }

  list(
    ok = TRUE,
    reason = NULL,
    adata = adata,
    cell_line_col = cell_line_col,
    area_col = area_col
  )
}

summarise_saved_results <- function(df, alpha_padj, lfc_thr) {
  if (is.null(df)) {
    return(list(
      tested_genes = NA_integer_,
      total_sig = NA_integer_,
      up_abnormal = NA_integer_,
      up_normal = NA_integer_
    ))
  }

  x <- as.data.frame(df)
  if (!all(c("pvals_adj", "log2fc_xenium_eps") %in% colnames(x))) {
    stop("Saved result is missing required columns: pvals_adj and/or log2fc_xenium_eps")
  }

  sig <- (x$pvals_adj < alpha_padj) & (abs(x$log2fc_xenium_eps) >= lfc_thr)
  sig_up <- sig & (x$log2fc_xenium_eps >= lfc_thr)
  sig_dn <- sig & (x$log2fc_xenium_eps <= -lfc_thr)

  list(
    tested_genes = nrow(x),
    total_sig = sum(sig, na.rm = TRUE),
    up_abnormal = sum(sig_up, na.rm = TRUE),
    up_normal = sum(sig_dn, na.rm = TRUE)
  )
}

summarise_cell_line_from_input <- function(adata, cl, cell_line_col, area_col, min_expr_cells, area_threshold) {
  out <- list(
    n_before = 0L,
    n_after_area = NA_integer_,
    n_after_morph = NA_integer_,
    n_abnormal = NA_integer_,
    n_normal = NA_integer_,
    genes_after_min_expr = NA_integer_,
    status = "ok",
    reason = NULL
  )

  mask_cl <- as.character(colData(adata)[[cell_line_col]]) == cl
  a <- adata[, mask_cl]
  out$n_before <- ncol(a)

  if (ncol(a) == 0) {
    out$status <- "skip"
    out$reason <- "no myotubes before filtering."
    return(out)
  }

  if (!is.null(area_threshold)) {
    area_vals <- suppressWarnings(as.numeric(as.character(colData(a)[[area_col]])))
    keep_area <- !is.na(area_vals) & area_vals >= area_threshold
    a <- a[, keep_area]
    out$n_after_area <- ncol(a)
    if (ncol(a) == 0) {
      out$status <- "skip"
      out$reason <- "no myotubes pass area_threshold."
      return(out)
    }
  }

  morph <- suppressWarnings(as.integer(as.character(colData(a)$morphology_class)))
  keep_grp <- morph %in% c(1L, 2L)
  a <- a[, keep_grp]
  morph <- morph[keep_grp]
  out$n_after_morph <- ncol(a)

  if (ncol(a) == 0) {
    out$status <- "skip"
    out$reason <- "no myotubes in morphology_class {1,2}."
    return(out)
  }

  group <- as.integer(morph == 2L)
  mask_abn <- group == 1
  mask_norm <- group == 0
  out$n_abnormal <- sum(mask_abn)
  out$n_normal <- sum(mask_norm)

  if (out$n_abnormal == 0 || out$n_normal == 0) {
    out$status <- "skip"
    out$reason <- "one group has zero myotubes."
    return(out)
  }

  cnt <- SummarizedExperiment::assay(a, "counts")
  nz_abn <- Matrix::rowSums(cnt[, mask_abn, drop = FALSE] > 0)
  nz_norm <- Matrix::rowSums(cnt[, mask_norm, drop = FALSE] > 0)
  keep_genes <- (nz_abn >= min_expr_cells) | (nz_norm >= min_expr_cells)
  out$genes_after_min_expr <- sum(keep_genes)

  if (!any(keep_genes)) {
    out$status <- "skip"
    out$reason <- "no genes pass min_expr_cells."
    return(out)
  }

  cnt <- cnt[keep_genes, , drop = FALSE]
  total_counts <- Matrix::colSums(cnt)
  nonzero_totals <- total_counts[total_counts > 0]
  if (length(nonzero_totals) == 0) {
    out$status <- "skip"
    out$reason <- "all myotubes zero counts."
    return(out)
  }

  median_total <- median(nonzero_totals)
  size_factors <- total_counts / median_total
  sf_pos <- size_factors[size_factors > 0]
  if (length(sf_pos) == 0) {
    out$status <- "skip"
    out$reason <- "no positive size factors."
    return(out)
  }

  out
}

bundle <- load_deg_bundle(DEG_PATH)
results_by_cell_line <- bundle$results_by_cell_line
if (is.null(results_by_cell_line)) {
  stop("`results_by_cell_line` was not found in DEG results: ", bundle$source_path)
}

stored_fields <- bundle$stored_fields %||% character(0)
min_expr_cells <- if ("min_expr_cells" %in% stored_fields) bundle$min_expr_cells else 100L
alpha_padj <- if ("alpha_padj" %in% stored_fields) bundle$alpha_padj else 0.05
lfc_thr <- if ("lfc_thr" %in% stored_fields) bundle$lfc_thr else 0.25
eps <- if ("eps" %in% stored_fields) bundle$eps else 1e-6
area_threshold <- if ("area_threshold" %in% stored_fields) bundle$area_threshold else NULL
y_max_clip <- if ("y_max_clip" %in% stored_fields) bundle$y_max_clip else 30
n_labels <- if ("n_labels" %in% stored_fields) bundle$n_labels else 15

input_ctx <- prepare_input_context(INPUT_PATH, USE_INPUT_OBJECT, area_threshold)
result_cell_lines <- sort(unique(names(results_by_cell_line)))
input_cell_lines <- character(0)
if (isTRUE(input_ctx$ok)) {
  input_cell_lines <- sort(unique(na.omit(as.character(colData(input_ctx$adata)[[input_ctx$cell_line_col]]))))
}
cell_lines <- if (length(input_cell_lines) > 0) input_cell_lines else result_cell_lines
extra_result_lines <- setdiff(result_cell_lines, cell_lines)
if (length(extra_result_lines) > 0) {
  cell_lines <- c(cell_lines, extra_result_lines)
}
n_cell_lines <- length(cell_lines)

message(sprintf("NB-GLM DE (myotubes) summary generated at %s",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
message("DEG source: ", bundle$source_path, " [", bundle$source_format, "]")
if (isTRUE(input_ctx$ok)) {
  message("Input object: ", normalizePath(INPUT_PATH))
} else {
  message("Input object unavailable; per-cell-line myotube counts skipped: ", input_ctx$reason)
}

message("Stored / inferred parameters:")
message(" - min_expr_cells: ", fmt_value(min_expr_cells))
message(" - alpha_padj: ", fmt_value(alpha_padj))
message(" - lfc_thr: ", fmt_value(lfc_thr))
message(" - eps: ", fmt_value(eps))
message(" - area_threshold: ", fmt_value(area_threshold))
message(" - y_max_clip: ", fmt_value(y_max_clip))
message(" - n_labels: ", fmt_value(n_labels))
message(" - slide_covariate: <not stored in myotube DEG output>")
message(" - INPUT_PATH used for summary reconstruction: ", fmt_value(INPUT_PATH))

message(sprintf("Cell lines with saved DEG results: %d", length(result_cell_lines)))
if (length(input_cell_lines) > 0) {
  message(sprintf("Cell lines in input object: %d", length(input_cell_lines)))
}

total_sig_all <- 0L
for (i in seq_along(cell_lines)) {
  cl <- cell_lines[[i]]
  message(sprintf("[%d/%d] %s: starting summary", i, n_cell_lines, cl))

  saved_df <- if (cl %in% names(results_by_cell_line)) results_by_cell_line[[cl]] else NULL
  sig_counts <- summarise_saved_results(saved_df, alpha_padj = alpha_padj, lfc_thr = lfc_thr)

  if (isTRUE(input_ctx$ok)) {
    line_summary <- summarise_cell_line_from_input(
      adata = input_ctx$adata,
      cl = cl,
      cell_line_col = input_ctx$cell_line_col,
      area_col = input_ctx$area_col,
      min_expr_cells = min_expr_cells,
      area_threshold = area_threshold
    )

    message(sprintf("[%d/%d] %s: %d myotubes before filtering", i, n_cell_lines, cl, line_summary$n_before))
    if (!is.null(area_threshold)) {
      message(sprintf("[%d/%d] %s: %d myotubes after area filter (%s >= %.3f)",
                      i, n_cell_lines, cl, line_summary$n_after_area, input_ctx$area_col, area_threshold))
    }
    if (!is.na(line_summary$n_after_morph)) {
      message(sprintf("[%d/%d] %s: %d myotubes after morphology_class filter (1/2)",
                      i, n_cell_lines, cl, line_summary$n_after_morph))
    }
    if (!is.na(line_summary$n_abnormal) && !is.na(line_summary$n_normal)) {
      message(sprintf("[%d/%d] %s: abnormal-only=%d, normal-only=%d",
                      i, n_cell_lines, cl, line_summary$n_abnormal, line_summary$n_normal))
    }
    if (!is.na(line_summary$genes_after_min_expr)) {
      message(sprintf("[%d/%d] %s: %d genes after min_expr_cells",
                      i, n_cell_lines, cl, line_summary$genes_after_min_expr))
    }
    if (!identical(line_summary$status, "ok")) {
      message(cl, ": ", line_summary$reason)
      if (!is.null(saved_df)) {
        message(sprintf("[%d/%d] %s: saved DEG results exist despite reconstructed stop (tested=%d, Total Sig=%d)",
                        i, n_cell_lines, cl, sig_counts$tested_genes, sig_counts$total_sig))
        total_sig_all <- total_sig_all + sig_counts$total_sig
      }
      next
    }
  }

  if (is.null(saved_df)) {
    message(cl, ": no saved DEG results found.")
    next
  }

  if (!isTRUE(input_ctx$ok) && !is.na(sig_counts$tested_genes)) {
    message(sprintf("[%d/%d] %s: %d genes after min_expr_cells (from saved results)",
                    i, n_cell_lines, cl, sig_counts$tested_genes))
  }

  if (isTRUE(input_ctx$ok) &&
      !is.na(line_summary$genes_after_min_expr) &&
      !is.na(sig_counts$tested_genes) &&
      !identical(as.integer(line_summary$genes_after_min_expr), as.integer(sig_counts$tested_genes))) {
    warning(
      sprintf(
        "%s: reconstructed genes_after_min_expr (%d) does not match saved tested genes (%d)",
        cl, line_summary$genes_after_min_expr, sig_counts$tested_genes
      ),
      call. = FALSE
    )
  }

  message(sprintf("[%d/%d] %s: Up in Abnormal=%d, Up in Normal=%d, Total Sig=%d",
                  i, n_cell_lines, cl,
                  sig_counts$up_abnormal,
                  sig_counts$up_normal,
                  sig_counts$total_sig))
  total_sig_all <- total_sig_all + sig_counts$total_sig
}

message(sprintf("Summary complete: cell lines listed=%d, saved result cell lines=%d, total significant genes across cell lines=%d",
                n_cell_lines, length(result_cell_lines), total_sig_all))
