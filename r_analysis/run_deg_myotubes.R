#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(edgeR)
  library(Matrix)
  library(tibble)
})

# ---------------- USER INPUTS ----------------
INPUT_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/r_dataset/processed_myotube_filtered.rds"
OUTPUT_DIR <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/r_dataset/deg"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------- LOAD INPUT ----------------
myotube_filtered <- readRDS(INPUT_PATH)

# Use filtered myotubes if present, else fallback to combined
adata <- if (exists("myotube_filtered")) myotube_filtered else myotube_combined
if (is.null(adata)) stop("No myotube object found (myotube_filtered or myotube_combined).")

# Parameters (same as nuclei)
min_expr_cells <- 100
alpha_padj <- 0.05
lfc_thr <- 0.25
eps <- 1e-6

# Saved so plotting notebooks can reuse expected defaults
y_max_clip <- 30
n_labels <- 15

pick_col <- function(df, candidates, label) {
  hit <- candidates[candidates %in% colnames(df)][1]
  if (is.na(hit)) {
    stop("Missing required column: ", label, ". Available: ", paste(colnames(df), collapse = ", "))
  }
  hit
}

# Ensure morphology_class exists (1=only normal, 2=only abnormal, 3=mixed, -1=none)
cd0 <- as.data.frame(colData(adata))
if (!"morphology_class" %in% colnames(cd0)) {
  if (all(c("n_normal_nuclei", "n_abnormal_nuclei") %in% colnames(cd0))) {
    morph <- rep(-1L, nrow(cd0))
    morph[cd0$n_normal_nuclei > 0 & cd0$n_abnormal_nuclei == 0] <- 1L
    morph[cd0$n_normal_nuclei == 0 & cd0$n_abnormal_nuclei > 0] <- 2L
    morph[cd0$n_normal_nuclei > 0 & cd0$n_abnormal_nuclei > 0] <- 3L
    colData(adata)$morphology_class <- morph
    cd0 <- as.data.frame(colData(adata))
  } else {
    stop("morphology_class missing and cannot be derived (n_normal_nuclei / n_abnormal_nuclei not found).")
  }
}

cell_line_col <- pick_col(cd0, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), "Cell Line")

results_by_cell_line <- list()
sig_genes_by_cell_line <- list()
sig_genes_up_abnormal <- list()
sig_genes_down_abnormal <- list()

cell_lines <- sort(unique(na.omit(as.character(colData(adata)[[cell_line_col]]))))
n_cell_lines <- length(cell_lines)

t0_all <- Sys.time()
message(sprintf("NB-GLM DE (myotubes) starting at %s; cell lines=%d",
                format(t0_all, "%Y-%m-%d %H:%M:%S"), n_cell_lines))

for (i in seq_along(cell_lines)) {
  cl <- cell_lines[i]
  cl_t0 <- Sys.time()
  message(sprintf("[%d/%d] %s: starting", i, n_cell_lines, cl))

  a <- adata[, as.character(colData(adata)[[cell_line_col]]) == cl]
  message(sprintf("[%d/%d] %s: %d myotubes before filtering", i, n_cell_lines, cl, ncol(a)))
  if (ncol(a) == 0) next

  morph <- suppressWarnings(as.integer(as.character(colData(a)$morphology_class)))
  keep_grp <- morph %in% c(1L, 2L)  # only-normal vs only-abnormal
  a <- a[, keep_grp]
  morph <- morph[keep_grp]
  message(sprintf("[%d/%d] %s: %d myotubes after normal/abnormal filter", i, n_cell_lines, cl, ncol(a)))
  if (ncol(a) == 0) {
    message(cl, ": no only-normal/only-abnormal myotubes.")
    next
  }

  group <- as.integer(morph == 2L) # 1 = abnormal-only, 0 = normal-only
  mask_abn <- group == 1
  mask_norm <- group == 0
  if (sum(mask_abn) == 0 || sum(mask_norm) == 0) {
    message(cl, ": one group has zero myotubes.")
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
    message(cl, ": all myotubes zero counts.")
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
  y <- estimateDisp(y, design)
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
  sig_up <- sig & (df$log2fc_xenium_eps >= lfc_thr)   # up in abnormal-only
  sig_dn <- sig & (df$log2fc_xenium_eps <= -lfc_thr)  # up in normal-only

  message(sprintf("[%d/%d] %s: Up in Abnormal=%d, Up in Normal=%d, Total Sig=%d (%.1fs)",
                  i, n_cell_lines, cl, sum(sig_up), sum(sig_dn), sum(sig),
                  as.numeric(difftime(Sys.time(), cl_t0, units = "secs"))))

  results_by_cell_line[[cl]] <- df
  sig_genes_by_cell_line[[cl]] <- df$names[sig]
  sig_genes_up_abnormal[[cl]] <- df$names[sig_up]
  sig_genes_down_abnormal[[cl]] <- df$names[sig_dn]
}

message(sprintf("NB-GLM DE (myotubes) finished in %.1f minutes",
                as.numeric(difftime(Sys.time(), t0_all, units = "mins"))))

# ---------------- SAVE OUTPUTS ----------------
out_rds <- file.path(OUTPUT_DIR, "myotube_deg_results.rds")
out_rdata <- file.path(OUTPUT_DIR, "myotube_deg_results.RData")
out_csv <- file.path(OUTPUT_DIR, "myotube_deg_results_all_cell_lines.csv")

saveRDS(list(
  results_by_cell_line = results_by_cell_line,
  sig_genes_by_cell_line = sig_genes_by_cell_line,
  sig_genes_up_abnormal = sig_genes_up_abnormal,
  sig_genes_down_abnormal = sig_genes_down_abnormal,
  min_expr_cells = min_expr_cells,
  alpha_padj = alpha_padj,
  lfc_thr = lfc_thr,
  eps = eps,
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
