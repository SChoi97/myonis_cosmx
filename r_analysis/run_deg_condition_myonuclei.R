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
pseudobulk <- FALSE
slide_covariate <- TRUE

control_lines <- c("NCRM1", "NCRM5", "L302P-Corr")
mutant_lines <- c("1174", "1175", "L302P")

area_col_candidates <- c("area_px2", "Area", "area", "area_um2", "nucleus_area", "cell_area")
sigmoid_col_candidates <- c("Sigmoid_Logits", "Sigmoid Logits", "sigmoid_logits", "sigmoid.logits")

# Saved so plotting notebooks can reuse expected defaults
y_max_clip <- 30
n_labels <- 15

format_ranked_genes_table <- function(df, direction = c("top", "bottom"), top_n = 20) {
  direction <- match.arg(direction)
  ord <- if (direction == "top") {
    order(-df$rank_score, df$pvals, df$names)
  } else {
    order(df$rank_score, df$pvals, df$names)
  }

  out <- as.data.frame(head(df[ord, c("names", "log2fc_xenium_eps", "pvals", "pvals_adj", "rank_score"),
                             drop = FALSE], top_n))
  rownames(out) <- NULL
  out
}

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Supports:
#   --KEY value
#   --key=value
#   key=value
# Usage example:
# Rscript run_deg_condition_myonuclei.R \
#   --INPUT_PATH /path/processed_myonuclei.rds \
#   --OUTPUT_DIR /path/deg \
#   --lfc_thr 0.25 \
#   --eps 1e-6 \
#   --sigmoid_logits_filter 0.2 0.8 \
#   --pseudobulk TRUE \
#   --slide_covariate TRUE \
#   --control_lines NCRM1 NCRM5 L302P-Corr \
#   --mutant_lines 1174 1175 L302P
overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
sigmoid_filter_set_null <- FALSE
area_threshold_raw <- NULL
if ("sigmoid_logits_filter" %in% names(overrides)) {
  raw_sigmoid <- tolower(trimws(overrides[["sigmoid_logits_filter"]]))
  if (raw_sigmoid %in% c("null", "none")) {
    sigmoid_filter_set_null <- TRUE
  }
}
if ("area_threshold" %in% names(overrides)) {
  area_threshold_raw <- trimws(overrides[["area_threshold"]])
}
apply_cli_overrides(overrides)
if (sigmoid_filter_set_null) {
  sigmoid_logits_filter <- NULL
}
if (!is.null(area_threshold_raw)) {
  if (tolower(area_threshold_raw) %in% c("null", "none")) {
    area_threshold <- NULL
  } else {
    area_threshold <- suppressWarnings(as.numeric(trimws(strsplit(area_threshold_raw, ",", fixed = TRUE)[[1]])))
  }
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------- LOAD INPUT ----------------
stop_if_missing_file(INPUT_PATH, "INPUT_PATH")
adata <- readRDS(INPUT_PATH)
if (is.null(adata)) stop("`myonuclei` is NULL.")

cd0 <- as.data.frame(colData(adata))
cell_line_col <- pick_col(cd0, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), "Cell Line")
slide_col <- pick_col(cd0, c("Slide Name", "Slide.Name", "slide_name", "slide"), "Slide Name", required = FALSE)
area_col <- pick_col(cd0, area_col_candidates, "Area", required = FALSE)
sigmoid_col <- pick_col(cd0, sigmoid_col_candidates, "Sigmoid Logits", required = !is.null(sigmoid_logits_filter))

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

control_norm <- normalize_label(control_lines)
mutant_norm <- normalize_label(mutant_lines)
overlap_lines <- intersect(control_norm, mutant_norm)
if (length(overlap_lines) > 0) {
  stop("control_lines and mutant_lines overlap after normalization: ", paste(overlap_lines, collapse = ", "))
}

cell_line_raw_all <- as.character(cd0[[cell_line_col]])
cell_line_norm_all <- normalize_label(cell_line_raw_all)
condition <- rep(NA_character_, length(cell_line_norm_all))
condition[cell_line_norm_all %in% control_norm] <- "control"
condition[cell_line_norm_all %in% mutant_norm] <- "mutant"

present_lines_norm <- unique(stats::na.omit(cell_line_norm_all))
missing_control <- control_lines[!(control_norm %in% present_lines_norm)]
missing_mutant <- mutant_lines[!(mutant_norm %in% present_lines_norm)]
if (length(missing_control) > 0) {
  warning("Control lines not found in data: ", paste(missing_control, collapse = ", "))
}
if (length(missing_mutant) > 0) {
  warning("Mutant lines not found in data: ", paste(missing_mutant, collapse = ", "))
}

keep_condition <- !is.na(condition)
adata <- adata[, keep_condition]
condition <- condition[keep_condition]
cell_line_used <- trimws(cell_line_raw_all[keep_condition])

if (ncol(adata) == 0) {
  stop("No nuclei matched the requested control/mutant cell line sets.")
}

t0_all <- Sys.time()
message(sprintf("NB-GLM DE (myonuclei; control vs mutant) starting at %s",
                format(t0_all, "%Y-%m-%d %H:%M:%S")))
message("Included cell lines: ", paste(sort(unique(cell_line_used)), collapse = ", "))
message("Counts by cell line and condition:")
print(table(condition = factor(condition, levels = c("control", "mutant")),
            cell_line = factor(cell_line_used, levels = sort(unique(cell_line_used)))))

message(sprintf("Selected %d nuclei from requested cell lines before additional filters", ncol(adata)))

if (!is.null(area_threshold)) {
  area_vals <- suppressWarnings(as.numeric(as.character(colData(adata)[[area_col]])))
  keep_area <- !is.na(area_vals) & area_vals >= area_threshold
  adata <- adata[, keep_area]
  condition <- condition[keep_area]
  cell_line_used <- cell_line_used[keep_area]
  message(sprintf("%d nuclei after area filter (%s >= %.3f)", ncol(adata), area_col, area_threshold))
  if (ncol(adata) == 0) {
    stop("No nuclei pass area_threshold.")
  }
}

if (!is.null(sigmoid_logits_filter)) {
  logits_vals <- suppressWarnings(as.numeric(as.character(colData(adata)[[sigmoid_col]])))
  keep_logits <- !is.na(logits_vals) & (logits_vals < sigmoid_low | logits_vals > sigmoid_high)
  adata <- adata[, keep_logits]
  condition <- condition[keep_logits]
  cell_line_used <- cell_line_used[keep_logits]
  message(sprintf("%d nuclei after sigmoid filter (%s outside [%.3f, %.3f])",
                  ncol(adata), sigmoid_col, sigmoid_low, sigmoid_high))
  if (ncol(adata) == 0) {
    stop("No nuclei pass sigmoid_logits_filter.")
  }
}

group <- as.integer(condition == "mutant")
mask_mut <- group == 1
mask_ctrl <- group == 0
if (sum(mask_mut) == 0 || sum(mask_ctrl) == 0) {
  stop("One condition has zero nuclei after filtering.")
}

cnt_cells <- SummarizedExperiment::assay(adata, "counts")

nz_mut <- Matrix::rowSums(cnt_cells[, mask_mut, drop = FALSE] > 0)
nz_ctrl <- Matrix::rowSums(cnt_cells[, mask_ctrl, drop = FALSE] > 0)
keep_genes <- (nz_mut >= min_expr_cells) | (nz_ctrl >= min_expr_cells)
if (!any(keep_genes)) {
  stop("No genes pass min_expr_cells.")
}

adata <- adata[keep_genes, ]
cnt_cells <- SummarizedExperiment::assay(adata, "counts")
message(sprintf("%d genes after min_expr_cells", nrow(cnt_cells)))

cell_line_condition_counts <- table(
  condition = factor(condition, levels = c("control", "mutant")),
  cell_line = factor(cell_line_used, levels = sort(unique(cell_line_used)))
)

pseudobulk_sample_info <- NULL
if (isTRUE(pseudobulk)) {
  pb_sample <- factor(cell_line_used, levels = sort(unique(cell_line_used)))
  agg_mat <- Matrix::sparse.model.matrix(~ 0 + pb_sample)
  cnt <- cnt_cells %*% agg_mat
  if (!inherits(cnt, "dgCMatrix")) {
    cnt <- as(cnt, "dgCMatrix")
  }
  colnames(cnt) <- levels(pb_sample)

  sample_cell_line <- colnames(cnt)
  sample_cell_line_norm <- normalize_label(sample_cell_line)
  missing_control_pb <- control_lines[!(control_norm %in% sample_cell_line_norm)]
  missing_mutant_pb <- mutant_lines[!(mutant_norm %in% sample_cell_line_norm)]
  if (length(missing_control_pb) > 0 || length(missing_mutant_pb) > 0) {
    stop(
      "pseudobulk=TRUE requires one pseudobulk sample per requested cell line. Missing after filtering: ",
      paste(c(missing_control_pb, missing_mutant_pb), collapse = ", ")
    )
  }

  sample_condition <- ifelse(normalize_label(sample_cell_line) %in% mutant_norm, "mutant", "control")
  group <- as.integer(sample_condition == "mutant")
  mask_mut <- group == 1
  mask_ctrl <- group == 0
  sample_counts <- table(factor(sample_condition, levels = c("control", "mutant")))
  expected_sample_counts <- c(length(control_lines), length(mutant_lines))
  if (any(as.integer(sample_counts) != expected_sample_counts)) {
    stop(
      "pseudobulk=TRUE expected ",
      length(control_lines), " control and ", length(mutant_lines), " mutant pseudobulk samples, got ",
      sample_counts[["control"]], " control and ", sample_counts[["mutant"]], " mutant."
    )
  }

  n_cells_by_sample <- as.integer(table(pb_sample)[sample_cell_line])
  names(n_cells_by_sample) <- sample_cell_line
  pseudobulk_sample_info <- data.frame(
    sample_id = sample_cell_line,
    cell_line = sample_cell_line,
    condition = sample_condition,
    n_cells = n_cells_by_sample,
    stringsAsFactors = FALSE
  )

  if (!is.na(slide_col)) {
    slide_chr <- as.character(colData(adata)[[slide_col]])
    slide_chr[is.na(slide_chr) | trimws(slide_chr) == ""] <- "UNKNOWN"
    slide_levels_per_sample <- tapply(slide_chr, pb_sample, function(x) paste(sort(unique(x)), collapse = "; "))
    n_slides_per_sample <- tapply(slide_chr, pb_sample, function(x) length(unique(x)))
    pseudobulk_sample_info$slides <- unname(slide_levels_per_sample[sample_cell_line])
    pseudobulk_sample_info$n_slides <- as.integer(unname(n_slides_per_sample[sample_cell_line]))
  }

  message("pseudobulk=TRUE: aggregating nuclei by cell line; fitting without slide covariate.")
  print(pseudobulk_sample_info)
} else {
  cnt <- cnt_cells
}

total_counts <- Matrix::colSums(cnt)
nonzero_totals <- total_counts[total_counts > 0]
if (length(nonzero_totals) == 0) {
  stop("All nuclei have zero counts.")
}

median_total <- median(nonzero_totals)
size_factors <- total_counts / median_total
sf_pos <- size_factors[size_factors > 0]
if (length(sf_pos) == 0) {
  stop("No positive size factors.")
}
size_factors[size_factors <= 0] <- min(sf_pos) * 1e-3

if (isTRUE(pseudobulk)) {
  design <- model.matrix(~ group)
} else if (isTRUE(slide_covariate)) {
  if (!is.na(slide_col)) {
    slide_chr <- as.character(colData(adata)[[slide_col]])
    slide_chr[is.na(slide_chr) | trimws(slide_chr) == ""] <- "UNKNOWN"
    slide <- factor(slide_chr)
    if (nlevels(slide) > 1) {
      design <- model.matrix(~ group + slide)
    } else {
      design <- model.matrix(~ group)
      message("slide column has one level; fitting without slide term.")
    }
  } else {
    design <- model.matrix(~ group)
    message("slide_covariate=TRUE but no slide column was found; fitting without slide term.")
  }
} else {
  design <- model.matrix(~ group)
}

y <- DGEList(counts = cnt)
y$samples$lib.size <- as.numeric(size_factors)
y$samples$norm.factors <- rep(1, ncol(cnt))

step_t0 <- Sys.time()
message("estimateDisp...")
y <- estimateDisp(y, design, trend.method = "none", grid.length = 11, grid.range = c(-6, 6))
message(sprintf("estimateDisp done (%.1fs)",
                as.numeric(difftime(Sys.time(), step_t0, units = "secs"))))

step_t0 <- Sys.time()
fit <- glmFit(y, design)
idx_group <- which(colnames(design) == "group")
if (length(idx_group) != 1) {
  stop("Could not find group in design: ", paste(colnames(design), collapse = ", "))
}
lrt <- glmLRT(fit, coef = idx_group)
message(sprintf("glmFit+glmLRT done (%.1fs)",
                as.numeric(difftime(Sys.time(), step_t0, units = "secs"))))

pvals <- lrt$table$PValue
pvals_adj <- p.adjust(pvals, method = "BH")

sf_sum_mut <- sum(size_factors[mask_mut])
sf_sum_ctrl <- sum(size_factors[mask_ctrl])

sum_mut <- Matrix::rowSums(cnt[, mask_mut, drop = FALSE])
sum_ctrl <- Matrix::rowSums(cnt[, mask_ctrl, drop = FALSE])

mean_mut <- as.numeric(sum_mut / sf_sum_mut)
mean_ctrl <- as.numeric(sum_ctrl / sf_sum_ctrl)
log2fc_xenium_eps <- log2((mean_mut + eps) / (mean_ctrl + eps))
rank_score <- -log10(pmax(as.numeric(pvals), .Machine$double.xmin)) * log2fc_xenium_eps

results <- tibble(
  names = rownames(cnt),
  beta0 = fit$coefficients[, 1],
  beta1 = fit$coefficients[, idx_group],
  beta_condition = fit$coefficients[, idx_group],
  log2fc_xenium_eps = log2fc_xenium_eps,
  mean_mutant_sf_norm = mean_mut,
  mean_control_sf_norm = mean_ctrl,
  pvals = pvals,
  pvals_adj = pvals_adj,
  rank_score = rank_score
)

sig <- (results$pvals_adj < alpha_padj) & (abs(results$log2fc_xenium_eps) >= lfc_thr)
sig_up <- sig & (results$log2fc_xenium_eps >= lfc_thr)   # up in mutant
sig_dn <- sig & (results$log2fc_xenium_eps <= -lfc_thr)  # up in control

sig_genes <- results$names[sig]
sig_genes_up_mutant <- results$names[sig_up]
sig_genes_down_mutant <- results$names[sig_dn]

condition_counts <- table(factor(condition, levels = c("control", "mutant")))
model_sample_counts <- table(factor(if (isTRUE(pseudobulk)) sample_condition else condition,
                                    levels = c("control", "mutant")))

message(sprintf("Up in Mutant=%d, Up in Control=%d, Total Sig=%d",
                sum(sig_up), sum(sig_dn), sum(sig)))

top_ranked_tbl <- format_ranked_genes_table(results, direction = "top", top_n = 20)
bottom_ranked_tbl <- format_ranked_genes_table(results, direction = "bottom", top_n = 20)

message("Top 20 genes by score = -log10(pval) * log2FC:")
print(top_ranked_tbl, row.names = FALSE)
message("Bottom 20 genes by score = -log10(pval) * log2FC:")
print(bottom_ranked_tbl, row.names = FALSE)

message(sprintf("NB-GLM DE (myonuclei; control vs mutant) finished in %.1f minutes",
                as.numeric(difftime(Sys.time(), t0_all, units = "mins"))))

# ---------------- SAVE OUTPUTS ----------------
out_suffix <- if (isTRUE(pseudobulk)) "_pseudobulk" else ""
out_rds <- file.path(OUTPUT_DIR, paste0("myonuclei_condition_deg_results", out_suffix, ".rds"))
out_rdata <- file.path(OUTPUT_DIR, paste0("myonuclei_condition_deg_results", out_suffix, ".RData"))
out_csv <- file.path(OUTPUT_DIR, paste0("myonuclei_condition_deg_results", out_suffix, ".csv"))

saveRDS(list(
  results = results,
  sig_genes = sig_genes,
  sig_genes_up_mutant = sig_genes_up_mutant,
  sig_genes_down_mutant = sig_genes_down_mutant,
  pseudobulk = pseudobulk,
  slide_covariate = slide_covariate,
  condition_counts = condition_counts,
  model_sample_counts = model_sample_counts,
  cell_line_condition_counts = cell_line_condition_counts,
  pseudobulk_sample_info = pseudobulk_sample_info,
  control_lines = control_lines,
  mutant_lines = mutant_lines,
  min_expr_cells = min_expr_cells,
  alpha_padj = alpha_padj,
  lfc_thr = lfc_thr,
  eps = eps,
  area_threshold = area_threshold,
  sigmoid_logits_filter = sigmoid_logits_filter,
  y_max_clip = y_max_clip,
  n_labels = n_labels,
  input_path = INPUT_PATH
), out_rds, compress = FALSE)

save(results,
     sig_genes,
     sig_genes_up_mutant,
     sig_genes_down_mutant,
     pseudobulk,
     slide_covariate,
     condition_counts,
     model_sample_counts,
     cell_line_condition_counts,
     pseudobulk_sample_info,
     control_lines,
     mutant_lines,
     min_expr_cells,
     alpha_padj,
     lfc_thr,
     eps,
     area_threshold,
     sigmoid_logits_filter,
     y_max_clip,
     n_labels,
     INPUT_PATH,
     file = out_rdata)

write.csv(as.data.frame(results), out_csv, row.names = FALSE)

message("Saved:")
message(" - ", out_rds)
message(" - ", out_rdata)
message(" - ", out_csv)
