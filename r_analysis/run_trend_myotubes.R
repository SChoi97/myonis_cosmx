#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(edgeR)
  library(Matrix)
  library(tibble)
  library(dplyr)
})

# ---------------- USER INPUTS ----------------
INPUT_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/r_dataset/processed_myotube_filtered.rds"
OUTPUT_DIR <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/r_dataset/trend_regression_myotubes"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------- LOAD INPUT ----------------
myotube_filtered <- readRDS(INPUT_PATH)

# Use filtered myotubes if present, else fallback to combined
adata <- if (exists("myotube_filtered")) myotube_filtered else myotube_combined
if (is.null(adata)) stop("No myotube object found (myotube_filtered or myotube_combined).")

# ---------------- CONFIG ----------------
min_expr_tubes <- 50
eps <- 1e-6
top_n_genes <- 50
use_ref_total <- "median"  # "median" or "mean"
prop_pseudo <- 0.5

if (!use_ref_total %in% c("median", "mean")) {
  stop("use_ref_total must be 'median' or 'mean'.")
}

pick_col <- function(df, candidates, label, required = TRUE) {
  hit <- candidates[candidates %in% colnames(df)][1]
  if (is.na(hit)) {
    if (required) stop("Missing required column: ", label, ". Available: ", paste(colnames(df), collapse = ", "))
    return(NA_character_)
  }
  hit
}

# Pull data once; avoid adata[,] subsetting for regression input
cd0 <- as.data.frame(SummarizedExperiment::colData(adata))
cnt0 <- SummarizedExperiment::assay(adata, "counts")
if (!inherits(cnt0, "dgCMatrix")) cnt0 <- as(cnt0, "dgCMatrix")
if (nrow(cd0) != ncol(cnt0)) stop("colData rows and count columns do not match.")

cell_line_col <- pick_col(cd0, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), "Cell Line")
slide_col <- pick_col(cd0, c("Slide Name", "Slide.Name", "slide_name", "slide"), "Slide Name", required = FALSE)

req_cols <- c("n_normal_nuclei", "n_abnormal_nuclei")
miss <- setdiff(req_cols, colnames(cd0))
if (length(miss) > 0) stop("Missing required myotube columns: ", paste(miss, collapse = ", "))

trend_results_by_cell_line <- list()
heatmap_pred_by_cell_line <- list()

cell_lines <- sort(unique(na.omit(as.character(cd0[[cell_line_col]]))))
n_cell_lines <- length(cell_lines)

t0_all <- Sys.time()
message(sprintf("Trend regression (myotubes; total_nuc + prop_abn) starting at %s; cell lines=%d",
                format(t0_all, "%Y-%m-%d %H:%M:%S"), n_cell_lines))

for (i in seq_along(cell_lines)) {
  cl <- cell_lines[i]
  cl_t0 <- Sys.time()
  message(sprintf("[%d/%d] %s: starting", i, n_cell_lines, cl))

  idx_cl <- which(as.character(cd0[[cell_line_col]]) == cl)
  if (length(idx_cl) == 0) next

  cd <- cd0[idx_cl, , drop = FALSE]
  cnt_all <- cnt0[, idx_cl, drop = FALSE]

  # Covariates (raw)
  n_abn <- suppressWarnings(as.numeric(as.character(cd$n_abnormal_nuclei)))
  n_norm <- suppressWarnings(as.numeric(as.character(cd$n_normal_nuclei)))
  total_nuc <- n_abn + n_norm

  # Keep only myotubes with >=1 nucleus
  keep_nuc <- total_nuc > 0 & is.finite(total_nuc)
  if (!any(keep_nuc)) {
    message(cl, ": no myotubes with nuclei.")
    next
  }
  cd <- cd[keep_nuc, , drop = FALSE]
  cnt_all <- cnt_all[, keep_nuc, drop = FALSE]

  # Recompute covariates after filtering
  n_abn <- suppressWarnings(as.numeric(as.character(cd$n_abnormal_nuclei)))
  n_norm <- suppressWarnings(as.numeric(as.character(cd$n_normal_nuclei)))
  total_nuc <- n_abn + n_norm

  # Stabilized abnormal proportion in [0,1]
  prop_abn <- (n_abn + prop_pseudo) / (total_nuc + 2 * prop_pseudo)

  # Robust scaling for total_nuc
  mean_total <- mean(total_nuc, na.rm = TRUE)
  sd_total <- sd(total_nuc, na.rm = TRUE)
  if (!is.finite(sd_total) || sd_total == 0) {
    total_nuc_sc <- rep(0, length(total_nuc))
  } else {
    total_nuc_sc <- as.numeric((total_nuc - mean_total) / sd_total)
  }

  # Size factors from all genes
  total_counts_all <- Matrix::colSums(cnt_all)
  nonzero_totals <- total_counts_all[total_counts_all > 0]
  if (length(nonzero_totals) == 0) {
    message(cl, ": all tubes zero counts (all genes).")
    next
  }
  median_total <- median(nonzero_totals)
  size_factors <- as.numeric(total_counts_all / median_total)
  sf_pos <- size_factors[size_factors > 0]
  if (length(sf_pos) == 0) {
    message(cl, ": no positive size factors.")
    next
  }
  size_factors[size_factors <= 0] <- min(sf_pos) * 1e-3

  # Regression/testing matrix
  cnt <- cnt_all
  nz <- Matrix::rowSums(cnt > 0)
  keep_genes <- nz >= min_expr_tubes
  if (!any(keep_genes)) {
    message(cl, ": no genes pass min_expr_tubes=", min_expr_tubes)
    next
  }
  cnt <- cnt[keep_genes, , drop = FALSE]

  message(cl, ": size factors from ", nrow(cnt_all), " total genes; regression on ", nrow(cnt), " genes.")

  # Design matrix: total_nuc + prop_abn (+ optional slide)
  if (!is.na(slide_col)) {
    slide_chr <- as.character(cd[[slide_col]])
    slide_chr[is.na(slide_chr) | trimws(slide_chr) == ""] <- "UNKNOWN"
    slide <- factor(slide_chr)
    if (nlevels(slide) > 1) {
      design <- model.matrix(~ total_nuc_sc + prop_abn + slide)
    } else {
      design <- model.matrix(~ total_nuc_sc + prop_abn)
      message(cl, ": slide column has one level; fitting without slide term.")
    }
  } else {
    design <- model.matrix(~ total_nuc_sc + prop_abn)
  }

  # EdgeR fit
  y <- edgeR::DGEList(counts = cnt)
  y$samples$lib.size <- as.numeric(size_factors)
  y$samples$norm.factors <- rep(1, ncol(cnt))
  y <- edgeR::estimateDisp(y, design)
  fit <- edgeR::glmFit(y, design)

  coef_names <- colnames(design)
  idx_total <- which(coef_names == "total_nuc_sc")
  idx_prop <- which(coef_names == "prop_abn")
  if (length(idx_total) != 1 || length(idx_prop) != 1) {
    stop("Could not find total_nuc_sc/prop_abn in design: ", paste(coef_names, collapse = ", "))
  }

  lrt_total <- edgeR::glmLRT(fit, coef = idx_total)
  lrt_prop <- edgeR::glmLRT(fit, coef = idx_prop)

  p_total <- lrt_total$table$PValue
  p_prop <- lrt_prop$table$PValue
  fdr_total <- p.adjust(p_total, method = "BH")
  fdr_prop <- p.adjust(p_prop, method = "BH")

  res <- tibble::tibble(
    gene = rownames(cnt),
    beta0 = as.numeric(fit$coefficients[, 1]),
    beta_total_nuc_sd = as.numeric(fit$coefficients[, idx_total]),
    beta_prop_abn = as.numeric(fit$coefficients[, idx_prop]),
    p_total = as.numeric(p_total),
    fdr_total = as.numeric(fdr_total),
    p_prop = as.numeric(p_prop),
    fdr_prop = as.numeric(fdr_prop),
    fc_per_1SD_total_nuc = exp(as.numeric(fit$coefficients[, idx_total])),
    fc_prop_0_to_1 = exp(as.numeric(fit$coefficients[, idx_prop])),
    fc_prop_plus_0.10 = exp(as.numeric(fit$coefficients[, idx_prop]) * 0.10)
  )

  trend_results_by_cell_line[[cl]] <- res

  message(cl, ": top 10 highest beta_prop_abn (with beta_total_nuc_sd):")
  print(
    res %>%
      dplyr::arrange(dplyr::desc(beta_prop_abn)) %>%
      dplyr::select(gene, beta_prop_abn, beta_total_nuc_sd, fdr_prop) %>%
      dplyr::slice_head(n = 10),
    n = 10
  )

  # Top genes for heatmap by |beta_prop_abn|
  top_genes <- res %>%
    dplyr::arrange(dplyr::desc(abs(beta_prop_abn))) %>%
    dplyr::slice_head(n = top_n_genes) %>%
    dplyr::pull(gene)

  # Prediction grid: vary prop_abn, hold total_nuc fixed
  prop_range <- range(prop_abn, na.rm = TRUE)
  prop_grid <- seq(prop_range[1], prop_range[2], length.out = 50)

  total_ref <- if (use_ref_total == "mean") mean_total else median(total_nuc, na.rm = TRUE)
  if (!is.finite(total_ref)) total_ref <- 1
  if (!is.finite(sd_total) || sd_total == 0) {
    total_ref_sc <- 0
  } else {
    total_ref_sc <- as.numeric((total_ref - mean_total) / sd_total)
  }

  message(
    cl, ": prop_abn range=", sprintf("%.3f", prop_range[1]), "-", sprintf("%.3f", prop_range[2]),
    " | total_nuc_ref=", round(total_ref), " (scaled=", sprintf("%.2f", total_ref_sc), ")",
    " | grid points=", length(prop_grid)
  )

  pred_df <- expand.grid(
    gene = top_genes,
    prop_abn_pred = prop_grid,
    stringsAsFactors = FALSE
  ) %>%
    dplyr::left_join(
      res %>%
        dplyr::filter(gene %in% top_genes) %>%
        dplyr::select(gene, beta0, beta_total_nuc_sd, beta_prop_abn),
      by = "gene"
    ) %>%
    dplyr::mutate(
      total_nuc_sc = total_ref_sc,
      log_mu = beta0 + beta_total_nuc_sd * total_nuc_sc + beta_prop_abn * prop_abn_pred,
      mu = exp(log_mu)
    )

  pred_df$gene <- factor(pred_df$gene, levels = top_genes)
  heatmap_pred_by_cell_line[[cl]] <- pred_df

  message(cl, ": heatmap genes=", length(top_genes), " | prop grid size=", length(prop_grid))
  message(sprintf("[%d/%d] %s: done (%.1fs)", i, n_cell_lines, cl,
                  as.numeric(difftime(Sys.time(), cl_t0, units = "secs"))))
}

trend_results_all <- dplyr::bind_rows(lapply(names(trend_results_by_cell_line), function(cl) {
  dplyr::mutate(trend_results_by_cell_line[[cl]], cell_line = cl)
}))

heatmap_pred_all <- dplyr::bind_rows(lapply(names(heatmap_pred_by_cell_line), function(cl) {
  dplyr::mutate(heatmap_pred_by_cell_line[[cl]], cell_line = cl)
}))

message(sprintf("Trend regression (myotubes) finished in %.1f minutes",
                as.numeric(difftime(Sys.time(), t0_all, units = "mins"))))

# ---------------- SAVE OUTPUTS ----------------
out_rds <- file.path(OUTPUT_DIR, "myotube_trend_regression_results.rds")
out_rdata <- file.path(OUTPUT_DIR, "myotube_trend_regression_results.RData")
out_csv <- file.path(OUTPUT_DIR, "myotube_trend_regression_results_all_cell_lines.csv")
out_heatmap_csv <- file.path(OUTPUT_DIR, "myotube_trend_regression_heatmap_predictions_all_cell_lines.csv")

saveRDS(list(
  trend_results_by_cell_line = trend_results_by_cell_line,
  heatmap_pred_by_cell_line = heatmap_pred_by_cell_line,
  trend_results_all = trend_results_all,
  heatmap_pred_all = heatmap_pred_all,
  min_expr_tubes = min_expr_tubes,
  eps = eps,
  top_n_genes = top_n_genes,
  use_ref_total = use_ref_total,
  prop_pseudo = prop_pseudo,
  input_path = INPUT_PATH
), out_rds, compress = FALSE)

save(trend_results_by_cell_line,
     heatmap_pred_by_cell_line,
     trend_results_all,
     heatmap_pred_all,
     min_expr_tubes,
     eps,
     top_n_genes,
     use_ref_total,
     prop_pseudo,
     INPUT_PATH,
     file = out_rdata)

if (nrow(trend_results_all) > 0) {
  trend_results_all <- trend_results_all[, c("cell_line", setdiff(colnames(trend_results_all), "cell_line"))]
  write.csv(trend_results_all, out_csv, row.names = FALSE)
}

if (nrow(heatmap_pred_all) > 0) {
  heatmap_pred_all <- heatmap_pred_all[, c("cell_line", setdiff(colnames(heatmap_pred_all), "cell_line"))]
  write.csv(heatmap_pred_all, out_heatmap_csv, row.names = FALSE)
}

message("Saved:")
message(" - ", out_rds)
message(" - ", out_rdata)
if (file.exists(out_csv)) message(" - ", out_csv)
if (file.exists(out_heatmap_csv)) message(" - ", out_heatmap_csv)

# ---------------- LOAD OUTPUT EXAMPLE ----------------
# out_rds <- file.path(OUTPUT_DIR, "myotube_trend_regression_results.rds")
# x <- readRDS(out_rds)
# trend_results_by_cell_line <- x$trend_results_by_cell_line
# heatmap_pred_by_cell_line <- x$heatmap_pred_by_cell_line
# trend_results_all <- x$trend_results_all
# heatmap_pred_all <- x$heatmap_pred_all
