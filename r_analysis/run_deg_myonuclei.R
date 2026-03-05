#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(edgeR)
  library(Matrix)
  library(tibble)
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
INPUT_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/processed_myonuclei.rds"
OUTPUT_DIR <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/deg"

min_expr_cells <- 100
alpha_padj <- 0.05
lfc_thr <- 0.25
eps <- 1e-6
area_threshold <- NULL  # e.g. 100
sigmoid_logits_filter <- c(0.2, 0.8)  # e.g. c(0.2, 0.8)

classification_col_candidates <- c("Predicted Class", "Predicted.Class", "predicted_class", "Classification", "classification")
area_col_candidates <- c("area_px2", "Area", "area", "area_um2", "nucleus_area", "cell_area")
sigmoid_col_candidates <- c("Sigmoid_Logits", "Sigmoid Logits", "sigmoid_logits", "sigmoid.logits")

# Saved so plotting notebooks can reuse expected defaults
y_max_clip <- 30
n_labels <- 15

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Supports:
#   --KEY value
#   --key=value
#   key=value
# Usage example:
# Rscript run_deg_myonuclei.R \
#   --INPUT_PATH /path/processed_myonuclei.rds \
#   --OUTPUT_DIR /path/deg \
#   --lfc_thr 0.25 \
#   --eps 1e-6 \
#   --sigmoid_logits_filter 0.2 0.8
overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
sigmoid_filter_set_null <- FALSE
if ("sigmoid_logits_filter" %in% names(overrides)) {
  raw_sigmoid <- tolower(trimws(overrides[["sigmoid_logits_filter"]]))
  if (raw_sigmoid %in% c("null", "none")) {
    sigmoid_filter_set_null <- TRUE
  }
}
apply_cli_overrides(overrides)
if (sigmoid_filter_set_null) {
  sigmoid_logits_filter <- NULL
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------- LOAD INPUT ----------------
myonuclei <- readRDS(INPUT_PATH)
adata <- myonuclei
if (is.null(adata)) stop("`myonuclei` is NULL.")

pick_col <- function(df, candidates, label, required = TRUE) {
  hit <- candidates[candidates %in% colnames(df)][1]
  if (is.na(hit)) {
    if (required) {
      stop("Missing required column: ", label, ". Available: ", paste(colnames(df), collapse = ", "))
    }
    return(NA_character_)
  }
  hit
}

cd0 <- as.data.frame(colData(adata))
cell_line_col <- pick_col(cd0, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), "Cell Line")
class_col <- pick_col(cd0, classification_col_candidates, "Predicted Class", required = FALSE)
area_col <- pick_col(cd0, area_col_candidates, "Area", required = FALSE)
sigmoid_col <- pick_col(cd0, sigmoid_col_candidates, "Sigmoid Logits", required = !is.null(sigmoid_logits_filter))

if (is.na(class_col)) {
  stop("Missing required column: Predicted Class (or compatible alias).")
}
if (!is.null(area_threshold)) {
  if (length(area_threshold) != 1 || !is.finite(area_threshold)) {
    stop("area_threshold must be NULL or a single finite numeric value.")
  }
  if (is.na(area_col)) {
    stop("area_threshold is set but no compatible area column was found.")
  }
}

sigmoid_low <- NA_real_
sigmoid_high <- NA_real_
if (!is.null(sigmoid_logits_filter)) {
  if (length(sigmoid_logits_filter) != 2 || any(!is.finite(sigmoid_logits_filter))) {
    stop("sigmoid_logits_filter must be a numeric vector of length 2, e.g. c(0.2, 0.8).")
  }
  sigmoid_low <- min(sigmoid_logits_filter)
  sigmoid_high <- max(sigmoid_logits_filter)
}

results_by_cell_line <- list()
sig_genes_by_cell_line <- list()
sig_genes_up_abnormal <- list()
sig_genes_down_abnormal <- list()

cell_lines <- sort(unique(na.omit(as.character(colData(adata)[[cell_line_col]]))))
n_cell_lines <- length(cell_lines)

t0_all <- Sys.time()
message(sprintf("NB-GLM DE (nuclei; Predicted Class 0 vs 1) starting at %s; cell lines=%d",
                format(t0_all, "%Y-%m-%d %H:%M:%S"), n_cell_lines))

for (i in seq_along(cell_lines)) {
  cl <- cell_lines[i]
  cl_t0 <- Sys.time()
  message(sprintf("[%d/%d] %s: starting", i, n_cell_lines, cl))

  a <- adata[, as.character(colData(adata)[[cell_line_col]]) == cl]
  message(sprintf("[%d/%d] %s: %d nuclei before filtering", i, n_cell_lines, cl, ncol(a)))
  if (ncol(a) == 0) next

  if (!is.null(area_threshold)) {
    area_vals <- suppressWarnings(as.numeric(as.character(colData(a)[[area_col]])))
    keep_area <- !is.na(area_vals) & area_vals >= area_threshold
    a <- a[, keep_area]
    message(sprintf("[%d/%d] %s: %d nuclei after area filter (%s >= %.3f)",
                    i, n_cell_lines, cl, ncol(a), area_col, area_threshold))
    if (ncol(a) == 0) {
      message(cl, ": no nuclei pass area_threshold.")
      next
    }
  }

  if (!is.null(sigmoid_logits_filter)) {
    logits_vals <- suppressWarnings(as.numeric(as.character(colData(a)[[sigmoid_col]])))
    keep_logits <- !is.na(logits_vals) & (logits_vals < sigmoid_low | logits_vals > sigmoid_high)
    a <- a[, keep_logits]
    message(sprintf("[%d/%d] %s: %d nuclei after sigmoid filter (%s outside [%.3f, %.3f])",
                    i, n_cell_lines, cl, ncol(a), sigmoid_col, sigmoid_low, sigmoid_high))
    if (ncol(a) == 0) {
      message(cl, ": no nuclei pass sigmoid_logits_filter.")
      next
    }
  }

  class_raw <- trimws(as.character(colData(a)[[class_col]]))
  class_norm <- tolower(class_raw)
  class_pred <- suppressWarnings(as.integer(class_raw))

  idx_na_class <- is.na(class_pred)
  if (any(idx_na_class)) {
    class_pred[idx_na_class] <- ifelse(class_norm[idx_na_class] %in% c("normal", "0"), 0L,
                                       ifelse(class_norm[idx_na_class] %in% c("abnormal", "1"), 1L, NA_integer_))
  }

  keep_grp <- class_pred %in% c(0L, 1L)
  a <- a[, keep_grp]
  class_pred <- class_pred[keep_grp]
  message(sprintf("[%d/%d] %s: %d nuclei after Predicted Class filter (0/1)", i, n_cell_lines, cl, ncol(a)))
  if (ncol(a) == 0) {
    message(cl, ": no nuclei with Predicted Class in {0,1}.")
    next
  }

  group <- as.integer(class_pred == 1L)  # 1 = abnormal (Predicted Class 1), 0 = normal (Predicted Class 0)
  mask_abn <- group == 1
  mask_norm <- group == 0
  if (sum(mask_abn) == 0 || sum(mask_norm) == 0) {
    message(cl, ": one group has zero nuclei.")
    next
  }

  cnt <- SummarizedExperiment::assay(a, "counts")

  nz_abn <- Matrix::rowSums(cnt[, mask_abn, drop = FALSE] > 0)
  nz_norm <- Matrix::rowSums(cnt[, mask_norm, drop = FALSE] > 0)
  keep_genes <- (nz_abn >= min_expr_cells) | (nz_norm >= min_expr_cells)
  if (!any(keep_genes)) {
    message(cl, ": no genes pass min_expr_cells.")
    next
  }

  a <- a[keep_genes, ]
  cnt <- SummarizedExperiment::assay(a, "counts")
  message(sprintf("[%d/%d] %s: %d genes after min_expr_cells", i, n_cell_lines, cl, nrow(cnt)))

  total_counts <- Matrix::colSums(cnt)
  nonzero_totals <- total_counts[total_counts > 0]
  if (length(nonzero_totals) == 0) {
    message(cl, ": all nuclei zero counts.")
    next
  }

  median_total <- median(nonzero_totals)
  size_factors <- total_counts / median_total
  sf_pos <- size_factors[size_factors > 0]
  if (length(sf_pos) == 0) {
    message(cl, ": no positive size factors.")
    next
  }
  size_factors[size_factors <= 0] <- min(sf_pos) * 1e-3

  design <- model.matrix(~ group)
  y <- DGEList(counts = cnt)
  y$samples$lib.size <- as.numeric(size_factors)
  y$samples$norm.factors <- rep(1, ncol(cnt))

  step_t0 <- Sys.time()
  message(sprintf("[%d/%d] %s: estimateDisp...", i, n_cell_lines, cl))
  y <- estimateDisp(y, design, trend.method = "none", grid.length = 11, grid.range = c(-6, 6))
  message(sprintf("[%d/%d] %s: estimateDisp done (%.1fs)", i, n_cell_lines, cl,
                  as.numeric(difftime(Sys.time(), step_t0, units = "secs"))))

  step_t0 <- Sys.time()
  fit <- glmFit(y, design)
  lrt <- glmLRT(fit, coef = "group")
  message(sprintf("[%d/%d] %s: glmFit+glmLRT done (%.1fs)", i, n_cell_lines, cl,
                  as.numeric(difftime(Sys.time(), step_t0, units = "secs"))))

  pvals <- lrt$table$PValue
  pvals_adj <- p.adjust(pvals, method = "BH")

  sf_sum_abn <- sum(size_factors[mask_abn])
  sf_sum_norm <- sum(size_factors[mask_norm])

  sum_abn <- Matrix::rowSums(cnt[, mask_abn, drop = FALSE])
  sum_norm <- Matrix::rowSums(cnt[, mask_norm, drop = FALSE])

  mean_abn <- as.numeric(sum_abn / sf_sum_abn)
  mean_norm <- as.numeric(sum_norm / sf_sum_norm)
  log2fc_xenium_eps <- log2((mean_abn + eps) / (mean_norm + eps))

  df <- tibble(
    names = rownames(cnt),
    beta0 = fit$coefficients[, 1],
    beta1 = fit$coefficients[, 2],
    log2fc_xenium_eps = log2fc_xenium_eps,
    mean_abn_sf_norm = mean_abn,
    mean_norm_sf_norm = mean_norm,
    pvals = pvals,
    pvals_adj = pvals_adj
  )

  sig <- (df$pvals_adj < alpha_padj) & (abs(df$log2fc_xenium_eps) >= lfc_thr)
  sig_up <- sig & (df$log2fc_xenium_eps >= lfc_thr)   # up in abnormal (Predicted Class 1)
  sig_dn <- sig & (df$log2fc_xenium_eps <= -lfc_thr)  # up in normal (Predicted Class 0)

  message(sprintf("[%d/%d] %s: Up in Abnormal=%d, Up in Normal=%d, Total Sig=%d (%.1fs)",
                  i, n_cell_lines, cl, sum(sig_up), sum(sig_dn), sum(sig),
                  as.numeric(difftime(Sys.time(), cl_t0, units = "secs"))))

  results_by_cell_line[[cl]] <- df
  sig_genes_by_cell_line[[cl]] <- df$names[sig]
  sig_genes_up_abnormal[[cl]] <- df$names[sig_up]
  sig_genes_down_abnormal[[cl]] <- df$names[sig_dn]
}

message(sprintf("NB-GLM DE (nuclei; Predicted Class 0 vs 1) finished in %.1f minutes",
                as.numeric(difftime(Sys.time(), t0_all, units = "mins"))))

# ---------------- SAVE OUTPUTS ----------------
out_rds <- file.path(OUTPUT_DIR, "myonuclei_deg_results.rds")
out_rdata <- file.path(OUTPUT_DIR, "myonuclei_deg_results.RData")
out_csv <- file.path(OUTPUT_DIR, "myonuclei_deg_results_all_cell_lines.csv")

saveRDS(list(
  results_by_cell_line = results_by_cell_line,
  sig_genes_by_cell_line = sig_genes_by_cell_line,
  sig_genes_up_abnormal = sig_genes_up_abnormal,
  sig_genes_down_abnormal = sig_genes_down_abnormal,
  min_expr_cells = min_expr_cells,
  alpha_padj = alpha_padj,
  lfc_thr = lfc_thr,
  eps = eps,
  area_threshold = area_threshold,
  sigmoid_logits_filter = sigmoid_logits_filter,
  y_max_clip = y_max_clip,
  n_labels = n_labels
), out_rds, compress = FALSE)

save(results_by_cell_line,
     sig_genes_by_cell_line,
     sig_genes_up_abnormal,
     sig_genes_down_abnormal,
     min_expr_cells,
     alpha_padj,
     lfc_thr,
     eps,
     area_threshold,
     sigmoid_logits_filter,
     y_max_clip,
     n_labels,
     file = out_rdata)

if (length(results_by_cell_line) > 0) {
  combined <- do.call(rbind, lapply(names(results_by_cell_line), function(cl) {
    x <- as.data.frame(results_by_cell_line[[cl]])
    x$cell_line <- cl
    x
  }))
  combined <- combined[, c("cell_line", setdiff(colnames(combined), "cell_line"))]
  write.csv(combined, out_csv, row.names = FALSE)
}

message("Saved:")
message(" - ", out_rds)
message(" - ", out_rdata)
if (file.exists(out_csv)) message(" - ", out_csv)
