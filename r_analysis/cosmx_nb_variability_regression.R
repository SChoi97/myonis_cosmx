#!/usr/bin/env Rscript

# Robust pilot test for morphology-associated gene-expression variability.
#
# For each cell line and gene:
#   1. Standardize the morphology predictor to one SD within cell line.
#   2. Fit the edgeR NB mean model: count ~ morphology_z + slide.
#   3. Calculate Pearson, NB deviance, and randomized-quantile
#      (Dunn-Smyth) residuals.
#   4. Fit Gamma(log) auxiliary models to winsorized squared residuals.
#
# The primary effect is exp(gamma): the residual-variance ratio per one-SD
# increase in morphology. The primary response is 99%-winsorized Pearson r^2;
# deviance and Dunn-Smyth results are reported as sensitivity analyses. HC3
# sandwich standard errors protect inference against remaining variance-model
# misspecification. Within-slide permutations provide a gene-specific empirical
# effect-size null instead of an arbitrary beta cutoff.
#
# Dunn-Smyth residuals also yield per-nucleus/myotube transcriptomic-instability
# scores (median absolute residual and trimmed mean squared residual). A targeted
# independent-object bootstrap refits both model stages. A targeted one-stage
# distributional NB validation jointly models mean and dispersion; it uses
# glmmTMB when available and a built-in maximum-likelihood fit otherwise.

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(edgeR)
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
source(file.path(script_dir, "utils", "cosmx_nb_glm_regression_utils.R"))

# ---------------- DEFAULTS (overridden by the .sh launcher) ----------------
myonuclei_rds_path <- ""
myotube_rds_path <- ""
output_dir <- ""
object_type <- "myonuclei"
morphology_feature <- "nuclear_aberration_mean_sigmoid"
cell_lines_to_include <- c("1174", "K32Del")

sample_n <- 10000L
sample_seed <- 20260802L
sample_balance_cell_lines <- TRUE
allow_missing_cell_lines <- TRUE
predictor_mode <- "auto"
global_rank_feature_patterns <- c("nuclear_aberration", "Sigmoid_Logits")
standardize_predictor_within_line <- TRUE
slide_covariate <- TRUE
min_obs_per_cell_line <- 100L
min_expr_obs <- 20L
alpha_padj <- 0.05
chunk_size <- 500L
console_top_n <- 10L
global_min_pct_positive <- 5
global_outlier_abs_threshold <- 2
winsor_upper_quantile <- 0.99
gamma_response_floor <- 1e-6
gamma_maxit <- 30L
gamma_tolerance <- 1e-7
dunn_smyth_seed <- 20260803L
instability_trim <- 0.10
permutation_reps <- 20L
permutation_quantile <- 0.95
permutation_seed <- 20260804L
permutation_cores <- 1L
bootstrap_reps <- 30L
bootstrap_top_n <- 20L
bootstrap_seed <- 20260805L
bootstrap_stratify_by_slide <- TRUE
run_one_stage_validation <- TRUE
one_stage_top_n <- 20L
run_extreme_octile_contrast <- TRUE
extreme_n_bins <- 8L
extreme_permutation_reps <- 20L
extreme_permutation_quantile <- 0.95
extreme_permutation_seed <- 20260806L
min_myonuclei_per_myotube <- 1L
filter_myonuclei <- TRUE
myonucleus_col <- "is_myonucleus"
derive_missing_circularity <- TRUE

overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
apply_cli_overrides(overrides)

as_flag <- function(x) {
  tolower(trimws(as.character(x)[1])) %in% c("1", "true", "t", "yes", "y")
}

object_type <- nbglm_canonical_object_type(object_type)
if (length(object_type) != 1L) stop("Provide exactly one object_type per run.")
morphology_feature <- as.character(morphology_feature)[1]
cell_lines_to_include <- nbglm_parse_list_arg(cell_lines_to_include)
global_rank_feature_patterns <- nbglm_parse_list_arg(global_rank_feature_patterns)
sample_balance_cell_lines <- as_flag(sample_balance_cell_lines)
allow_missing_cell_lines <- as_flag(allow_missing_cell_lines)
standardize_predictor_within_line <- as_flag(standardize_predictor_within_line)
slide_covariate <- as_flag(slide_covariate)
filter_myonuclei <- as_flag(filter_myonuclei)
derive_missing_circularity <- as_flag(derive_missing_circularity)
run_one_stage_validation <- as_flag(run_one_stage_validation)
bootstrap_stratify_by_slide <- as_flag(bootstrap_stratify_by_slide)
run_extreme_octile_contrast <- as_flag(run_extreme_octile_contrast)

sample_n <- as.integer(sample_n)
sample_seed <- as.integer(sample_seed)
min_obs_per_cell_line <- as.integer(min_obs_per_cell_line)
min_expr_obs <- as.integer(min_expr_obs)
chunk_size <- as.integer(chunk_size)
console_top_n <- as.integer(console_top_n)
gamma_maxit <- as.integer(gamma_maxit)
dunn_smyth_seed <- as.integer(dunn_smyth_seed)
permutation_reps <- as.integer(permutation_reps)
permutation_seed <- as.integer(permutation_seed)
permutation_cores <- as.integer(permutation_cores)
bootstrap_reps <- as.integer(bootstrap_reps)
bootstrap_top_n <- as.integer(bootstrap_top_n)
bootstrap_seed <- as.integer(bootstrap_seed)
one_stage_top_n <- as.integer(one_stage_top_n)
extreme_n_bins <- as.integer(extreme_n_bins)
extreme_permutation_reps <- as.integer(extreme_permutation_reps)
extreme_permutation_seed <- as.integer(extreme_permutation_seed)
min_myonuclei_per_myotube <- as.integer(min_myonuclei_per_myotube)
alpha_padj <- as.numeric(alpha_padj)
global_min_pct_positive <- as.numeric(global_min_pct_positive)
global_outlier_abs_threshold <- as.numeric(global_outlier_abs_threshold)
winsor_upper_quantile <- as.numeric(winsor_upper_quantile)
gamma_response_floor <- as.numeric(gamma_response_floor)
gamma_tolerance <- as.numeric(gamma_tolerance)
instability_trim <- as.numeric(instability_trim)
permutation_quantile <- as.numeric(permutation_quantile)
extreme_permutation_quantile <- as.numeric(extreme_permutation_quantile)

if (!nzchar(myonuclei_rds_path)) stop("myonuclei_rds_path cannot be empty.")
if (!nzchar(myotube_rds_path)) stop("myotube_rds_path cannot be empty.")
if (!nzchar(output_dir)) stop("output_dir cannot be empty.")
if (!nzchar(morphology_feature)) stop("morphology_feature cannot be empty.")
if (!length(cell_lines_to_include)) stop("cell_lines_to_include cannot be empty.")
if (is.na(sample_n) || sample_n < 1L) stop("sample_n must be >= 1.")
if (is.na(chunk_size) || chunk_size < 1L) stop("chunk_size must be >= 1.")
if (is.na(console_top_n) || console_top_n < 1L) stop("console_top_n must be >= 1.")
if (is.na(gamma_maxit) || gamma_maxit < 1L) stop("gamma_maxit must be >= 1.")
if (is.na(permutation_reps) || permutation_reps < 0L) stop("permutation_reps must be >= 0.")
if (is.na(permutation_cores) || permutation_cores < 1L) stop("permutation_cores must be >= 1.")
if (is.na(bootstrap_reps) || bootstrap_reps < 0L) stop("bootstrap_reps must be >= 0.")
if (is.na(bootstrap_top_n) || bootstrap_top_n < 0L) stop("bootstrap_top_n must be >= 0.")
if (is.na(one_stage_top_n) || one_stage_top_n < 0L) stop("one_stage_top_n must be >= 0.")
if (is.na(extreme_n_bins) || extreme_n_bins < 2L) stop("extreme_n_bins must be >= 2.")
if (is.na(extreme_permutation_reps) || extreme_permutation_reps < 0L) {
  stop("extreme_permutation_reps must be >= 0.")
}
if (!is.finite(global_min_pct_positive) || global_min_pct_positive < 0 ||
    global_min_pct_positive > 100) {
  stop("global_min_pct_positive must be between 0 and 100.")
}
if (!is.finite(global_outlier_abs_threshold) || global_outlier_abs_threshold <= 0) {
  stop("global_outlier_abs_threshold must be > 0.")
}
if (!is.finite(winsor_upper_quantile) || winsor_upper_quantile <= 0.5 ||
    winsor_upper_quantile >= 1) {
  stop("winsor_upper_quantile must be between 0.5 and 1.")
}
if (!is.finite(gamma_response_floor) || gamma_response_floor <= 0) {
  stop("gamma_response_floor must be > 0.")
}
if (!is.finite(gamma_tolerance) || gamma_tolerance <= 0) {
  stop("gamma_tolerance must be > 0.")
}
if (!is.finite(instability_trim) || instability_trim < 0 || instability_trim >= 0.5) {
  stop("instability_trim must be in [0, 0.5).")
}
if (!is.finite(permutation_quantile) || permutation_quantile <= 0.5 ||
    permutation_quantile >= 1) {
  stop("permutation_quantile must be between 0.5 and 1.")
}
if (!is.finite(extreme_permutation_quantile) ||
    extreme_permutation_quantile <= 0.5 || extreme_permutation_quantile >= 1) {
  stop("extreme_permutation_quantile must be between 0.5 and 1.")
}

varreg_sample_sce <- function(sce, cell_line_col, morphology_feature,
                              requested_lines, sample_n, sample_seed,
                              balance_cell_lines = TRUE,
                              allow_missing_cell_lines = TRUE) {
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  observed_lines <- sort(unique(stats::na.omit(as.character(cd[[cell_line_col]]))))
  requested_norm <- normalize_label(requested_lines)
  selected_lines <- observed_lines[normalize_label(observed_lines) %in% requested_norm]
  missing_lines <- requested_lines[!(requested_norm %in% normalize_label(selected_lines))]
  if (length(missing_lines)) {
    details <- paste0(
      "Requested cell line(s) not found: ", paste(missing_lines, collapse = ", "),
      ". Available lines: ", paste(observed_lines, collapse = ", "), "."
    )
    if (!isTRUE(allow_missing_cell_lines)) stop(details)
    warning(
      details,
      " Proceeding with requested line(s) that are present: ",
      if (length(selected_lines)) paste(selected_lines, collapse = ", ") else "<none>"
    )
  }
  if (!length(selected_lines)) stop("None of the requested cell lines are present in this object.")

  morphology <- suppressWarnings(as.numeric(as.character(cd[[morphology_feature]])))
  eligible <- as.character(cd[[cell_line_col]]) %in% selected_lines & is.finite(morphology)
  eligible_idx <- which(eligible)
  if (!length(eligible_idx)) stop("No eligible observations after cell-line and morphology filtering.")

  target_n <- min(as.integer(sample_n), length(eligible_idx))
  set.seed(as.integer(sample_seed))
  if (!isTRUE(balance_cell_lines) || length(selected_lines) == 1L) {
    sampled_idx <- if (target_n < length(eligible_idx)) {
      sample(eligible_idx, size = target_n, replace = FALSE)
    } else {
      eligible_idx
    }
  } else {
    # Begin with equal allocation; fill shortages from the remaining pool.
    selected_lines <- sort(selected_lines)
    base_n <- target_n %/% length(selected_lines)
    extra_n <- target_n %% length(selected_lines)
    targets <- rep(base_n, length(selected_lines))
    if (extra_n > 0L) targets[seq_len(extra_n)] <- targets[seq_len(extra_n)] + 1L

    sampled_parts <- vector("list", length(selected_lines))
    for (i in seq_along(selected_lines)) {
      line_idx <- eligible_idx[as.character(cd[[cell_line_col]][eligible_idx]) == selected_lines[i]]
      take_n <- min(targets[i], length(line_idx))
      sampled_parts[[i]] <- if (take_n < length(line_idx)) {
        sample(line_idx, size = take_n, replace = FALSE)
      } else {
        line_idx
      }
    }
    sampled_idx <- unlist(sampled_parts, use.names = FALSE)
    remaining_n <- target_n - length(sampled_idx)
    if (remaining_n > 0L) {
      remaining_pool <- setdiff(eligible_idx, sampled_idx)
      sampled_idx <- c(sampled_idx, sample(remaining_pool, size = remaining_n, replace = FALSE))
    }
  }

  sce[, sort(sampled_idx), drop = FALSE]
}

varreg_zscore <- function(x, label = "predictor") {
  x <- as.numeric(x)
  center <- mean(x)
  scale <- stats::sd(x)
  if (!is.finite(scale) || scale <= 0) {
    stop(label, " cannot be standardized because its SD is zero or non-finite.")
  }
  list(x = (x - center) / scale, center = center, scale = scale)
}

varreg_winsorize_rows <- function(x, upper_quantile = 0.99, floor = 1e-6) {
  caps <- apply(
    x,
    1L,
    stats::quantile,
    probs = upper_quantile,
    na.rm = TRUE,
    names = FALSE,
    type = 8
  )
  caps <- pmax(as.numeric(caps), floor)
  out <- sweep(x, 1L, caps, FUN = function(value, cap) pmin(value, cap))
  out[!is.finite(out)] <- floor
  out <- pmax(out, floor)
  list(response = out, cap = caps)
}

# Vectorized Gamma(log) IRLS. With a Gamma variance function and log link, the
# working weights are constant, so all genes sharing a design can be fitted in
# chunks. HC3 uses the Gamma score residual y/mu - 1, not raw response errors.
varreg_gamma_log_hc3 <- function(response, design, predictor_idx,
                                 response_floor = 1e-6,
                                 maxit = 30L, tolerance = 1e-7,
                                 compute_se = TRUE) {
  response <- pmax(as.matrix(response), response_floor)
  xtx_inv <- chol2inv(chol(crossprod(design)))
  coefficients <- matrix(0, nrow(response), ncol(design))
  intercept_idx <- which(colnames(design) == "(Intercept)")
  if (length(intercept_idx) == 1L) {
    coefficients[, intercept_idx] <- log(pmax(rowMeans(response), response_floor))
  }

  converged <- rep(FALSE, nrow(response))
  last_delta <- rep(Inf, nrow(response))
  iterations <- 0L
  for (iteration in seq_len(as.integer(maxit))) {
    eta <- coefficients %*% t(design)
    eta <- pmin(pmax(eta, -30), 30)
    mu <- exp(eta)
    working_response <- eta + response / mu - 1
    updated <- (working_response %*% design) %*% xtx_inv
    last_delta <- apply(abs(updated - coefficients), 1L, max)
    coefficients <- updated
    iterations <- iteration
    converged <- is.finite(last_delta) & last_delta <= tolerance
    if (all(converged)) break
  }

  beta <- coefficients[, predictor_idx]
  if (!isTRUE(compute_se)) {
    return(list(
      beta = beta,
      converged = converged,
      iterations = iterations,
      last_delta = last_delta
    ))
  }

  eta <- pmin(pmax(coefficients %*% t(design), -30), 30)
  mu <- exp(eta)
  score_residual <- response / mu - 1
  hat_values <- rowSums((design %*% xtx_inv) * design)
  hc3_denom <- pmax(1 - hat_values, 1e-6)
  predictor_influence <- drop(design %*% xtx_inv[, predictor_idx])
  hc3_weights <- (predictor_influence / hc3_denom)^2
  hc3_var <- drop((score_residual^2) %*% hc3_weights)
  hc3_se <- sqrt(pmax(hc3_var, 0))
  statistic <- beta / hc3_se
  statistic[!is.finite(statistic)] <- NA_real_
  residual_df <- nrow(design) - ncol(design)
  pvals <- 2 * stats::pt(abs(statistic), df = residual_df, lower.tail = FALSE)

  list(
    beta = beta,
    hc3_se = hc3_se,
    statistic = statistic,
    pvals = pvals,
    fitted = mu,
    converged = converged,
    iterations = iterations,
    last_delta = last_delta,
    residual_df = residual_df,
    max_design_leverage = max(hat_values)
  )
}

varreg_nb_abs_deviance_residual <- function(counts, mu, dispersion) {
  counts <- as.matrix(counts)
  mu <- pmax(as.matrix(mu), .Machine$double.eps)
  dispersion <- pmax(as.numeric(dispersion), 0)
  out <- matrix(NA_real_, nrow(counts), ncol(counts))

  nb_rows <- dispersion > 1e-10
  if (any(nb_rows)) {
    yy <- counts[nb_rows, , drop = FALSE]
    mm <- mu[nb_rows, , drop = FALSE]
    size <- pmin(1 / dispersion[nb_rows], 1e12)
    term1 <- matrix(0, nrow(yy), ncol(yy))
    positive <- yy > 0
    term1[positive] <- yy[positive] * log(yy[positive] / mm[positive])
    y_plus_size <- sweep(yy, 1L, size, "+")
    mu_plus_size <- sweep(mm, 1L, size, "+")
    term2 <- y_plus_size * log(y_plus_size / mu_plus_size)
    out[nb_rows, ] <- sqrt(pmax(2 * (term1 - term2), 0))
  }
  if (any(!nb_rows)) {
    yy <- counts[!nb_rows, , drop = FALSE]
    mm <- mu[!nb_rows, , drop = FALSE]
    term <- matrix(0, nrow(yy), ncol(yy))
    positive <- yy > 0
    term[positive] <- yy[positive] * log(yy[positive] / mm[positive])
    out[!nb_rows, ] <- sqrt(pmax(2 * (term - yy + mm), 0))
  }
  out
}

varreg_nb_dunn_smyth_residual <- function(counts, mu, dispersion,
                                          probability_floor = 1e-8) {
  counts <- as.matrix(counts)
  mu <- pmax(as.matrix(mu), .Machine$double.eps)
  size <- pmin(1 / pmax(as.numeric(dispersion), 1e-12), 1e12)
  y_vec <- as.vector(counts)
  mu_vec <- as.vector(mu)
  size_vec <- rep(size, times = ncol(counts))
  lower <- stats::pnbinom(y_vec - 1, mu = mu_vec, size = size_vec)
  upper <- stats::pnbinom(y_vec, mu = mu_vec, size = size_vec)
  probability <- lower + stats::runif(length(y_vec)) * pmax(upper - lower, 0)
  probability <- pmin(pmax(probability, probability_floor), 1 - probability_floor)
  matrix(stats::qnorm(probability), nrow = nrow(counts), ncol = ncol(counts))
}

varreg_make_permuted_designs <- function(cd, predictor, slide_col,
                                         slide_covariate, reps, seed) {
  if (reps < 1L) return(list())
  strata <- rep("ALL", length(predictor))
  if (isTRUE(slide_covariate) && !is.na(slide_col) && slide_col %in% colnames(cd)) {
    strata <- as.character(cd[[slide_col]])
    strata[is.na(strata) | !nzchar(trimws(strata))] <- "UNKNOWN"
  }
  strata_indices <- split(seq_along(predictor), strata)
  set.seed(as.integer(seed))
  lapply(seq_len(as.integer(reps)), function(i) {
    permuted <- predictor
    for (idx in strata_indices) permuted[idx] <- sample(predictor[idx], replace = FALSE)
    info <- nbglm_build_design(
      col_data = cd,
      predictor = permuted,
      slide_col = slide_col,
      slide_covariate = slide_covariate
    )
    predictor_idx <- which(colnames(info$design) == info$predictor_coef)
    if (length(predictor_idx) != 1L || qr(info$design)$rank < ncol(info$design)) {
      stop("A within-slide permutation produced an invalid design matrix.")
    }
    list(design = info$design, predictor_idx = predictor_idx)
  })
}

varreg_print_gene_table <- function(df, title, top_n = 10L) {
  cat("\n", title, "\n", sep = "")
  if (!nrow(df)) {
    cat("  <none>\n")
    return(invisible(NULL))
  }

  finite_adj <- unique(df$pvals_adj[is.finite(df$pvals_adj)])
  if (length(finite_adj) <= 1L) {
    df <- df[order(df$pvals, -abs(df$gamma_log_variability_per_sd)), , drop = FALSE]
  } else {
    df <- df[
      order(df$pvals_adj, df$pvals, -abs(df$gamma_log_variability_per_sd)),
      ,
      drop = FALSE
    ]
  }
  display_cols <- c(
    "names",
    "variance_ratio_per_sd",
    "variance_percent_change_per_sd",
    "variability_direction",
    "pvals",
    "pvals_adj",
    "permutation_effect_pass",
    "dunn_smyth_variance_ratio_per_sd",
    "abs_deviance_ratio_per_sd",
    "mean_model_log2_rate_ratio_per_sd",
    "mean_model_pvals_adj",
    "pct_positive"
  )
  display <- utils::head(df[, display_cols, drop = FALSE], as.integer(top_n))
  numeric_cols <- vapply(display, is.numeric, logical(1))
  display[numeric_cols] <- lapply(display[numeric_cols], function(x) signif(x, digits = 4L))
  print(display, row.names = FALSE)
  invisible(NULL)
}

varreg_print_sensitivity_gene_table <- function(df, title, ratio_col,
                                                 p_col, padj_col,
                                                 top_n = 10L) {
  cat("\n", title, "\n", sep = "")
  if (!nrow(df)) {
    cat("  <none>\n")
    return(invisible(NULL))
  }
  df <- df[
    order(df[[padj_col]], df[[p_col]], -abs(log(df[[ratio_col]]))),
    ,
    drop = FALSE
  ]
  display_cols <- unique(c(
    "names", ratio_col, p_col, padj_col,
    "variance_ratio_per_sd", "mean_model_pvals_adj", "pct_positive"
  ))
  display <- utils::head(df[, display_cols, drop = FALSE], as.integer(top_n))
  numeric_cols <- vapply(display, is.numeric, logical(1))
  display[numeric_cols] <- lapply(display[numeric_cols], function(x) signif(x, digits = 4L))
  print(display, row.names = FALSE)
  invisible(NULL)
}

varreg_print_line_results <- function(results, alpha_padj, top_n = 10L) {
  if (is.null(results) || !nrow(results)) return(invisible(NULL))

  cell_line <- unique(results$cell_line)
  significant <- results$significant %in% TRUE
  increased <- significant & results$gamma_log_variability_per_sd > 0
  decreased <- significant & results$gamma_log_variability_per_sd < 0
  mean_significant <- !is.na(results$mean_model_pvals_adj) &
    results$mean_model_pvals_adj < alpha_padj
  variability_only <- significant & !mean_significant &
    !is.na(results$mean_model_pvals_adj)

  cat("\n========== VARIABILITY RESULTS: ", cell_line, " ==========\n", sep = "")
  cat("Genes tested:                         ", nrow(results), "\n", sep = "")
  cat("Significant variability genes:       ", sum(significant), "\n", sep = "")
  cat("  Increased with morphology:         ", sum(increased), "\n", sep = "")
  cat("  Decreased with morphology:         ", sum(decreased), "\n", sep = "")
  cat("Variability hits without mean hit:   ", sum(variability_only), "\n", sep = "")
  cat("Variability hits also with mean hit: ", sum(significant & mean_significant), "\n", sep = "")
  cat(
    "Primary model: Gamma(log) on winsorized Pearson r^2; morphology is z-scored within line.\n",
    "Interpretation: variance_ratio_per_sd > 1 means more residual variance per one-SD ",
    "increase in morphology.\n",
    sep = ""
  )
  cat(
    "A significant call requires BH p < ", alpha_padj,
    " and an effect beyond the within-slide permutation cutoff.\n",
    sep = ""
  )
  finite_cutoff <- results$permutation_abs_log_variance_ratio_cutoff[
    is.finite(results$permutation_abs_log_variance_ratio_cutoff)
  ]
  if (length(finite_cutoff)) {
    median_cutoff <- stats::median(finite_cutoff)
    cat(
      "Median gene-specific permutation cutoff: |log variance ratio| > ",
      signif(median_cutoff, 4L), " (ratios outside approximately ",
      signif(exp(-median_cutoff), 4L), "-", signif(exp(median_cutoff), 4L), ").\n",
      sep = ""
    )
  }

  varreg_print_gene_table(
    results[increased, , drop = FALSE],
    paste0("Top significant INCREASED-variability genes (up to ", top_n, ")"),
    top_n
  )
  varreg_print_gene_table(
    results[decreased, , drop = FALSE],
    paste0("Top significant DECREASED-variability genes (up to ", top_n, ")"),
    top_n
  )
  varreg_print_gene_table(
    results[variability_only, , drop = FALSE],
    paste0("Top variability hits WITHOUT a significant mean-expression trend (up to ", top_n, ")"),
    top_n
  )
  if (!any(significant)) {
    varreg_print_gene_table(
      results[!is.na(results$pvals_adj), , drop = FALSE],
      paste0("No genes passed FDR plus permutation effect cutoff; strongest evidence (up to ", top_n, ")"),
      top_n
    )
  }

  deviance_significant <- results$abs_deviance_significant_fdr %in% TRUE
  cat(
    "\nAbsolute-deviance sensitivity: ", sum(deviance_significant),
    " genes with BH p < ", alpha_padj, ".\n",
    sep = ""
  )
  varreg_print_sensitivity_gene_table(
    if (any(deviance_significant)) {
      results[deviance_significant, , drop = FALSE]
    } else {
      results[is.finite(results$abs_deviance_pvals_adj), , drop = FALSE]
    },
    if (any(deviance_significant)) {
      paste0("Top absolute-deviance sensitivity hits (up to ", top_n, ")")
    } else {
      paste0("No absolute-deviance FDR hits; strongest evidence (up to ", top_n, ")")
    },
    ratio_col = "abs_deviance_ratio_per_sd",
    p_col = "abs_deviance_pvals",
    padj_col = "abs_deviance_pvals_adj",
    top_n = top_n
  )

  dunn_significant <- results$dunn_smyth_significant_fdr %in% TRUE
  cat(
    "\nDunn-Smyth sensitivity: ", sum(dunn_significant),
    " genes with BH p < ", alpha_padj, ".\n",
    sep = ""
  )
  varreg_print_sensitivity_gene_table(
    if (any(dunn_significant)) {
      results[dunn_significant, , drop = FALSE]
    } else {
      results[is.finite(results$dunn_smyth_pvals_adj), , drop = FALSE]
    },
    if (any(dunn_significant)) {
      paste0("Top Dunn-Smyth sensitivity hits (up to ", top_n, ")")
    } else {
      paste0("No Dunn-Smyth FDR hits; strongest evidence (up to ", top_n, ")")
    },
    ratio_col = "dunn_smyth_variance_ratio_per_sd",
    p_col = "dunn_smyth_pvals",
    padj_col = "dunn_smyth_pvals_adj",
    top_n = top_n
  )
  cat("===================================================\n")
  invisible(NULL)
}

varreg_fit_global_metrics <- function(scores, design, predictor_idx,
                                      object_type, morphology_feature,
                                      cell_line, alpha_padj = 0.05,
                                      gamma_response_floor = 1e-6,
                                      gamma_maxit = 30L,
                                      gamma_tolerance = 1e-7) {
  metric_cols <- c(
    "median_abs_dunn_smyth_residual",
    "trimmed_mean_squared_dunn_smyth_residual",
    "mean_squared_pearson_residual",
    "fraction_abs_pearson_outliers",
    "normalized_residual_entropy"
  )
  metric_labels <- c(
    "Transcriptomic instability (median |Dunn-Smyth residual|)",
    "Transcriptomic instability (trimmed mean Dunn-Smyth r^2)",
    "Overall residual variance",
    "Fraction of gene-expression outliers",
    "Residual entropy"
  )
  metric_interpretation <- c(
    "Ratio > 1: the typical absolute unexpected deviation is larger",
    "Ratio > 1: robust transcriptome-wide residual variance is larger",
    "Ratio > 1: greater overall NB-standardized expression variability",
    "Ratio > 1: a larger fraction of genes are expression outliers",
    paste0(
      "Ratio > 1: deviations are spread more diffusely across genes; ",
      "ratio < 1: deviations are concentrated in fewer genes"
    )
  )

  response <- t(as.matrix(scores[, metric_cols, drop = FALSE]))
  fit <- varreg_gamma_log_hc3(
    response = response,
    design = design,
    predictor_idx = predictor_idx,
    response_floor = gamma_response_floor,
    maxit = gamma_maxit,
    tolerance = gamma_tolerance
  )
  beta <- fit$beta
  hc3_se <- fit$hc3_se
  robust_t <- fit$statistic
  pvals <- fit$pvals
  pvals_adj <- stats::p.adjust(pvals, method = "BH")

  predictor_rank <- rank(scores$predictor, ties.method = "average")
  low_idx <- predictor_rank <= stats::quantile(predictor_rank, 0.2, names = FALSE)
  high_idx <- predictor_rank >= stats::quantile(predictor_rank, 0.8, names = FALSE)
  response_by_object <- t(response)
  low_mean <- colMeans(response_by_object[low_idx, , drop = FALSE])
  high_mean <- colMeans(response_by_object[high_idx, , drop = FALSE])
  raw_difference <- high_mean - low_mean
  percent_change <- ifelse(
    is.finite(low_mean) & abs(low_mean) > .Machine$double.eps,
    100 * raw_difference / abs(low_mean),
    NA_real_
  )

  data.frame(
    object_type = object_type,
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    metric = metric_cols,
    metric_label = metric_labels,
    model = "Gamma(log) with HC3 standard errors",
    beta_log_ratio_per_sd = beta,
    effect_ratio_per_sd = exp(beta),
    percent_change_per_sd = 100 * (exp(beta) - 1),
    beta_per_predictor = beta,
    direction = ifelse(beta > 0, "increased", "decreased"),
    hc3_se = hc3_se,
    robust_t = robust_t,
    pvals = pvals,
    pvals_adj = pvals_adj,
    significant = !is.na(pvals_adj) & pvals_adj < alpha_padj,
    lowest_morphology_quintile_mean = low_mean,
    highest_morphology_quintile_mean = high_mean,
    high_minus_low = raw_difference,
    percent_change_low_to_high = percent_change,
    n_lowest_quintile = sum(low_idx),
    n_highest_quintile = sum(high_idx),
    n_obs = nrow(scores),
    residual_df = fit$residual_df,
    gamma_converged = fit$converged,
    interpretation = metric_interpretation,
    stringsAsFactors = FALSE
  )
}

varreg_make_global_bins <- function(scores, n_bins = 5L) {
  metric_cols <- c(
    "median_abs_dunn_smyth_residual",
    "trimmed_mean_squared_dunn_smyth_residual",
    "mean_squared_pearson_residual",
    "fraction_abs_pearson_outliers",
    "normalized_residual_entropy"
  )
  predictor_rank <- rank(scores$predictor, ties.method = "average")
  morphology_bin <- pmin(
    as.integer(n_bins),
    pmax(1L, ceiling(as.integer(n_bins) * predictor_rank / length(predictor_rank)))
  )

  rows <- lapply(sort(unique(morphology_bin)), function(bin) {
    keep <- morphology_bin == bin
    metric_means <- vapply(scores[keep, metric_cols, drop = FALSE], mean, numeric(1))
    metric_medians <- vapply(scores[keep, metric_cols, drop = FALSE], stats::median, numeric(1))
    data.frame(
      object_type = unique(scores$object_type),
      morphology_feature = unique(scores$morphology_feature),
      cell_line = unique(scores$cell_line),
      morphology_bin = paste0("Q", bin),
      n_obs = sum(keep),
      predictor_min = min(scores$predictor[keep]),
      predictor_mean = mean(scores$predictor[keep]),
      predictor_max = max(scores$predictor[keep]),
      morphology_raw_mean = mean(scores$morphology_raw[keep]),
      morphology_raw_median = stats::median(scores$morphology_raw[keep]),
      median_abs_dunn_smyth_residual_mean = metric_means[[1]],
      trimmed_mean_squared_dunn_smyth_residual_mean = metric_means[[2]],
      mean_squared_pearson_residual_mean = metric_means[[3]],
      fraction_abs_pearson_outliers_mean = metric_means[[4]],
      normalized_residual_entropy_mean = metric_means[[5]],
      median_abs_dunn_smyth_residual_median = metric_medians[[1]],
      trimmed_mean_squared_dunn_smyth_residual_median = metric_medians[[2]],
      mean_squared_pearson_residual_median = metric_medians[[3]],
      fraction_abs_pearson_outliers_median = metric_medians[[4]],
      normalized_residual_entropy_median = metric_medians[[5]],
      stringsAsFactors = FALSE
    )
  })
  nbglm_bind_rows(rows)
}

varreg_print_global_results <- function(results, bins, n_genes,
                                        min_pct_positive,
                                        outlier_abs_threshold) {
  if (is.null(results) || !nrow(results)) return(invisible(NULL))
  cell_line <- unique(results$cell_line)
  cat("\n========== TRANSCRIPTOME-WIDE STOCHASTICITY: ", cell_line, " ==========\n", sep = "")
  cat("Genes used in per-nucleus scores: ", n_genes, "\n", sep = "")
  cat("Gene detection threshold:         >=", min_pct_positive, "% of nuclei\n", sep = "")
  cat("Outlier definition:               |Pearson residual| >= ", outlier_abs_threshold, "\n", sep = "")
  cat("Primary instability residual:     randomized-quantile (Dunn-Smyth)\n")
  cat("All effects are Gamma(log) ratios per one-SD morphology and adjust for slide.\n\n")

  display_cols <- c(
    "metric_label",
    "effect_ratio_per_sd",
    "percent_change_per_sd",
    "pvals",
    "pvals_adj",
    "lowest_morphology_quintile_mean",
    "highest_morphology_quintile_mean",
    "percent_change_low_to_high",
    "significant"
  )
  display <- results[, display_cols, drop = FALSE]
  numeric_cols <- vapply(display, is.numeric, logical(1))
  display[numeric_cols] <- lapply(display[numeric_cols], function(x) signif(x, digits = 4L))
  print(display, row.names = FALSE)

  cat("\nInterpretation by analysis:\n")
  for (i in seq_len(nrow(results))) {
    cat("- ", results$metric_label[i], ": ", results$interpretation[i], ".\n", sep = "")
  }

  cat("\nUnadjusted descriptive means across nuclear-aberration quintiles:\n")
  bin_display_cols <- c(
    "morphology_bin",
    "n_obs",
    "morphology_raw_mean",
    "median_abs_dunn_smyth_residual_mean",
    "trimmed_mean_squared_dunn_smyth_residual_mean",
    "mean_squared_pearson_residual_mean",
    "fraction_abs_pearson_outliers_mean",
    "normalized_residual_entropy_mean"
  )
  bin_display <- bins[, bin_display_cols, drop = FALSE]
  numeric_bin_cols <- vapply(bin_display, is.numeric, logical(1))
  bin_display[numeric_bin_cols] <- lapply(
    bin_display[numeric_bin_cols], function(x) signif(x, digits = 4L)
  )
  print(bin_display, row.names = FALSE)
  cat("================================================================\n")
  invisible(NULL)
}

varreg_case_bootstrap <- function(counts, size_factors, cd, predictor,
                                  slide_col, slide_covariate,
                                  selected_genes, observed_gamma,
                                  reps = 30L, seed = 1L,
                                  stratify_by_slide = TRUE,
                                  winsor_upper_quantile = 0.99,
                                  gamma_response_floor = 1e-6,
                                  gamma_maxit = 30L,
                                  gamma_tolerance = 1e-7) {
  if (reps < 1L || !length(selected_genes)) {
    return(list(replicates = data.frame(), summary = data.frame()))
  }
  if (!requireNamespace("MASS", quietly = TRUE)) {
    warning("MASS is unavailable; skipping refitted case bootstrap.")
    return(list(replicates = data.frame(), summary = data.frame()))
  }

  slide <- rep("ALL", length(predictor))
  use_slide <- isTRUE(slide_covariate) && !is.na(slide_col) &&
    slide_col %in% colnames(cd)
  if (use_slide) {
    slide <- as.character(cd[[slide_col]])
    slide[is.na(slide) | !nzchar(trimws(slide))] <- "UNKNOWN"
    use_slide <- length(unique(slide)) > 1L
  }
  strata <- if (isTRUE(stratify_by_slide) && use_slide) slide else rep("ALL", length(slide))
  strata_indices <- split(seq_along(predictor), strata)
  selected_genes <- intersect(selected_genes, rownames(counts))
  output <- vector("list", as.integer(reps) * length(selected_genes))
  output_idx <- 0L
  set.seed(as.integer(seed))

  for (bootstrap_rep in seq_len(as.integer(reps))) {
    sampled_idx <- unlist(
      lapply(strata_indices, function(idx) sample(idx, length(idx), replace = TRUE)),
      use.names = FALSE
    )
    bootstrap_data <- data.frame(
      predictor = predictor[sampled_idx],
      log_size_factor = log(size_factors[sampled_idx]),
      slide = factor(slide[sampled_idx]),
      stringsAsFactors = FALSE
    )
    bootstrap_design <- if (use_slide && nlevels(bootstrap_data$slide) > 1L) {
      stats::model.matrix(~ predictor + slide, data = bootstrap_data)
    } else {
      stats::model.matrix(~ predictor, data = bootstrap_data)
    }
    predictor_idx <- which(colnames(bootstrap_design) == "predictor")

    for (gene in selected_genes) {
      output_idx <- output_idx + 1L
      count <- as.numeric(counts[gene, sampled_idx])
      bootstrap_data$count <- count
      formula <- if (use_slide && nlevels(bootstrap_data$slide) > 1L) {
        count ~ predictor + slide + offset(log_size_factor)
      } else {
        count ~ predictor + offset(log_size_factor)
      }
      nb_fit <- tryCatch(
        suppressWarnings(MASS::glm.nb(
          formula,
          data = bootstrap_data,
          control = stats::glm.control(maxit = 50L)
        )),
        error = function(e) e
      )
      if (inherits(nb_fit, "error") || !is.finite(nb_fit$theta) || nb_fit$theta <= 0) {
        output[[output_idx]] <- data.frame(
          names = gene,
          bootstrap_rep = bootstrap_rep,
          gamma_log_variability_per_sd = NA_real_,
          variance_ratio_per_sd = NA_real_,
          status = if (inherits(nb_fit, "error")) conditionMessage(nb_fit) else "invalid_theta",
          stringsAsFactors = FALSE
        )
        next
      }

      mu <- matrix(stats::fitted(nb_fit), nrow = 1L)
      nb_variance <- mu + (1 / nb_fit$theta) * mu^2
      pearson_r2 <- matrix((count - mu)^2 / pmax(nb_variance, .Machine$double.eps), nrow = 1L)
      response <- varreg_winsorize_rows(
        pearson_r2,
        upper_quantile = winsor_upper_quantile,
        floor = gamma_response_floor
      )$response
      gamma_fit <- tryCatch(
        varreg_gamma_log_hc3(
          response = response,
          design = bootstrap_design,
          predictor_idx = predictor_idx,
          response_floor = gamma_response_floor,
          maxit = gamma_maxit,
          tolerance = gamma_tolerance,
          compute_se = FALSE
        ),
        error = function(e) e
      )
      if (inherits(gamma_fit, "error")) {
        gamma <- NA_real_
        status <- conditionMessage(gamma_fit)
      } else {
        gamma <- gamma_fit$beta[[1]]
        status <- if (gamma_fit$converged[[1]]) "ok" else "gamma_not_converged"
      }
      output[[output_idx]] <- data.frame(
        names = gene,
        bootstrap_rep = bootstrap_rep,
        gamma_log_variability_per_sd = gamma,
        variance_ratio_per_sd = exp(gamma),
        status = status,
        stringsAsFactors = FALSE
      )
    }
  }

  replicates <- nbglm_bind_rows(output)
  summaries <- lapply(selected_genes, function(gene) {
    values <- replicates$gamma_log_variability_per_sd[replicates$names == gene]
    values <- values[is.finite(values)]
    observed <- observed_gamma[[gene]]
    if (!length(values)) {
      return(data.frame(
        names = gene,
        observed_gamma_log_variability_per_sd = observed,
        observed_variance_ratio_per_sd = exp(observed),
        bootstrap_reps_successful = 0L,
        bootstrap_gamma_median = NA_real_,
        bootstrap_gamma_ci_low = NA_real_,
        bootstrap_gamma_ci_high = NA_real_,
        bootstrap_variance_ratio_ci_low = NA_real_,
        bootstrap_variance_ratio_ci_high = NA_real_,
        bootstrap_sign_stability = NA_real_,
        stringsAsFactors = FALSE
      ))
    }
    ci <- stats::quantile(values, c(0.025, 0.975), names = FALSE, type = 8)
    data.frame(
      names = gene,
      observed_gamma_log_variability_per_sd = observed,
      observed_variance_ratio_per_sd = exp(observed),
      bootstrap_reps_successful = length(values),
      bootstrap_gamma_median = stats::median(values),
      bootstrap_gamma_ci_low = ci[[1]],
      bootstrap_gamma_ci_high = ci[[2]],
      bootstrap_variance_ratio_ci_low = exp(ci[[1]]),
      bootstrap_variance_ratio_ci_high = exp(ci[[2]]),
      bootstrap_sign_stability = mean(sign(values) == sign(observed)),
      stringsAsFactors = FALSE
    )
  })
  list(replicates = replicates, summary = nbglm_bind_rows(summaries))
}

varreg_custom_distributional_nb_gene <- function(count, design, predictor_idx,
                                                  log_size_factor) {
  count <- as.numeric(count)
  log_size_factor <- as.numeric(log_size_factor)
  poisson_start <- tryCatch(
    suppressWarnings(stats::glm.fit(
      x = design,
      y = count,
      family = stats::poisson(),
      offset = log_size_factor,
      control = stats::glm.control(maxit = 50L)
    )),
    error = function(e) e
  )
  if (inherits(poisson_start, "error")) {
    return(list(status = paste0("Poisson start failed: ", conditionMessage(poisson_start))))
  }
  beta_start <- poisson_start$coefficients
  beta_start[!is.finite(beta_start)] <- 0
  mu_start <- pmax(poisson_start$fitted.values, 1e-8)
  alpha_moment <- stats::median(
    pmax(((count - mu_start)^2 - mu_start) / pmax(mu_start^2, 1e-8), 0),
    na.rm = TRUE
  )
  if (!is.finite(alpha_moment) || alpha_moment <= 1e-6) alpha_moment <- 0.05
  gamma_start <- rep(0, ncol(design))
  intercept_idx <- which(colnames(design) == "(Intercept)")
  if (length(intercept_idx) == 1L) gamma_start[intercept_idx] <- log(alpha_moment)

  negative_log_likelihood <- function(parameters) {
    p <- ncol(design)
    beta <- parameters[seq_len(p)]
    gamma <- parameters[p + seq_len(p)]
    log_mu <- pmin(pmax(log_size_factor + drop(design %*% beta), -30), 30)
    log_alpha <- pmin(pmax(drop(design %*% gamma), -20), 20)
    mu <- exp(log_mu)
    size <- exp(-log_alpha)
    value <- -sum(stats::dnbinom(count, mu = mu, size = size, log = TRUE))
    if (is.finite(value)) value else .Machine$double.xmax / 100
  }

  fit <- tryCatch(
    stats::optim(
      par = c(beta_start, gamma_start),
      fn = negative_log_likelihood,
      method = "BFGS",
      hessian = TRUE,
      control = list(maxit = 500L, reltol = 1e-8)
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(status = paste0("joint NB optimization failed: ", conditionMessage(fit))))
  }
  p <- ncol(design)
  beta <- fit$par[seq_len(p)]
  gamma <- fit$par[p + seq_len(p)]
  covariance <- tryCatch(solve(fit$hessian), error = function(e) NULL)
  if (is.null(covariance) || any(!is.finite(covariance))) {
    beta_se <- gamma_se <- NA_real_
  } else {
    beta_se <- sqrt(pmax(covariance[predictor_idx, predictor_idx], 0))
    gamma_position <- p + predictor_idx
    gamma_se <- sqrt(pmax(covariance[gamma_position, gamma_position], 0))
  }
  beta_p <- if (is.finite(beta_se) && beta_se > 0) {
    2 * stats::pnorm(abs(beta[predictor_idx] / beta_se), lower.tail = FALSE)
  } else {
    NA_real_
  }
  gamma_p <- if (is.finite(gamma_se) && gamma_se > 0) {
    2 * stats::pnorm(abs(gamma[predictor_idx] / gamma_se), lower.tail = FALSE)
  } else {
    NA_real_
  }
  list(
    status = if (fit$convergence == 0L) "ok" else paste0("optim_convergence_", fit$convergence),
    beta = beta[predictor_idx],
    beta_se = beta_se,
    beta_p = beta_p,
    gamma = gamma[predictor_idx],
    gamma_se = gamma_se,
    gamma_p = gamma_p,
    convergence = fit$convergence,
    objective = fit$value
  )
}

varreg_one_stage_validation <- function(counts, size_factors, cd, predictor,
                                        slide_col, slide_covariate,
                                        selected_genes, alpha_padj = 0.05) {
  if (!length(selected_genes)) return(data.frame())
  use_glmmtmb <- requireNamespace("glmmTMB", quietly = TRUE)
  if (!use_glmmtmb) {
    message(
      "glmmTMB is unavailable; using the built-in one-stage distributional NB ",
      "maximum-likelihood validator."
    )
  }

  slide <- rep("ALL", length(predictor))
  use_slide <- isTRUE(slide_covariate) && !is.na(slide_col) &&
    slide_col %in% colnames(cd)
  if (use_slide) {
    slide <- as.character(cd[[slide_col]])
    slide[is.na(slide) | !nzchar(trimws(slide))] <- "UNKNOWN"
    use_slide <- length(unique(slide)) > 1L
  }
  model_data <- data.frame(
    predictor = predictor,
    log_size_factor = log(size_factors),
    slide = factor(slide),
    stringsAsFactors = FALSE
  )
  conditional_formula <- if (use_slide) {
    count ~ predictor + slide + offset(log_size_factor)
  } else {
    count ~ predictor + offset(log_size_factor)
  }
  dispersion_formula <- if (use_slide) ~ predictor + slide else ~ predictor
  design_info <- nbglm_build_design(
    col_data = cd,
    predictor = predictor,
    slide_col = slide_col,
    slide_covariate = slide_covariate
  )
  joint_design <- design_info$design
  joint_predictor_idx <- which(colnames(joint_design) == design_info$predictor_coef)

  rows <- lapply(intersect(selected_genes, rownames(counts)), function(gene) {
    model_data$count <- as.numeric(counts[gene, ])
    if (!use_glmmtmb) {
      fit <- varreg_custom_distributional_nb_gene(
        count = model_data$count,
        design = joint_design,
        predictor_idx = joint_predictor_idx,
        log_size_factor = model_data$log_size_factor
      )
      return(data.frame(
        names = gene,
        status = fit$status,
        backend = "built-in distributional NB maximum likelihood",
        conditional_mean_log_rate_ratio_per_sd = if (is.null(fit$beta)) NA_real_ else fit$beta,
        conditional_mean_rate_ratio_per_sd = if (is.null(fit$beta)) NA_real_ else exp(fit$beta),
        conditional_mean_pvals = if (is.null(fit$beta_p)) NA_real_ else fit$beta_p,
        dispersion_log_phi_ratio_per_sd = if (is.null(fit$gamma)) NA_real_ else -fit$gamma,
        overdispersion_log_ratio_per_sd = if (is.null(fit$gamma)) NA_real_ else fit$gamma,
        overdispersion_log_ratio_se = if (is.null(fit$gamma_se)) NA_real_ else fit$gamma_se,
        overdispersion_ratio_per_sd = if (is.null(fit$gamma)) NA_real_ else exp(fit$gamma),
        dispersion_pvals = if (is.null(fit$gamma_p)) NA_real_ else fit$gamma_p,
        stringsAsFactors = FALSE
      ))
    }
    fit <- tryCatch(
      suppressWarnings(glmmTMB::glmmTMB(
        formula = conditional_formula,
        dispformula = dispersion_formula,
        family = glmmTMB::nbinom2,
        data = model_data
      )),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      return(data.frame(
        names = gene,
        status = conditionMessage(fit),
        backend = "glmmTMB nbinom2",
        conditional_mean_log_rate_ratio_per_sd = NA_real_,
        conditional_mean_rate_ratio_per_sd = NA_real_,
        conditional_mean_pvals = NA_real_,
        dispersion_log_phi_ratio_per_sd = NA_real_,
        overdispersion_log_ratio_per_sd = NA_real_,
        overdispersion_log_ratio_se = NA_real_,
        overdispersion_ratio_per_sd = NA_real_,
        dispersion_pvals = NA_real_,
        stringsAsFactors = FALSE
      ))
    }
    coefficients <- summary(fit)$coefficients
    conditional <- coefficients$cond
    dispersion <- coefficients$disp
    if (!("predictor" %in% rownames(conditional)) || !("predictor" %in% rownames(dispersion))) {
      return(data.frame(
        names = gene,
        status = "predictor coefficient missing",
        backend = "glmmTMB nbinom2",
        conditional_mean_log_rate_ratio_per_sd = NA_real_,
        conditional_mean_rate_ratio_per_sd = NA_real_,
        conditional_mean_pvals = NA_real_,
        dispersion_log_phi_ratio_per_sd = NA_real_,
        overdispersion_log_ratio_per_sd = NA_real_,
        overdispersion_log_ratio_se = NA_real_,
        overdispersion_ratio_per_sd = NA_real_,
        dispersion_pvals = NA_real_,
        stringsAsFactors = FALSE
      ))
    }
    # glmmTMB nbinom2 uses Var(Y)=mu+mu^2/phi. Thus exp(-gamma_disp)
    # is the fold-change in NB overdispersion 1/phi per one-SD morphology.
    data.frame(
      names = gene,
      status = "ok",
      backend = "glmmTMB nbinom2",
      conditional_mean_log_rate_ratio_per_sd = conditional["predictor", "Estimate"],
      conditional_mean_rate_ratio_per_sd = exp(conditional["predictor", "Estimate"]),
      conditional_mean_pvals = conditional["predictor", "Pr(>|z|)"],
      dispersion_log_phi_ratio_per_sd = dispersion["predictor", "Estimate"],
      overdispersion_log_ratio_per_sd = -dispersion["predictor", "Estimate"],
      overdispersion_log_ratio_se = dispersion["predictor", "Std. Error"],
      overdispersion_ratio_per_sd = exp(-dispersion["predictor", "Estimate"]),
      dispersion_pvals = dispersion["predictor", "Pr(>|z|)"],
      stringsAsFactors = FALSE
    )
  })
  results <- nbglm_bind_rows(rows)
  if (nrow(results) && "dispersion_pvals" %in% colnames(results)) {
    results$dispersion_pvals_adj <- stats::p.adjust(results$dispersion_pvals, method = "BH")
    results$significant_dispersion_validation <-
      !is.na(results$dispersion_pvals_adj) & results$dispersion_pvals_adj < alpha_padj
    results$p_adjust_scope <- paste0(
      "targeted validation set only (n=", sum(is.finite(results$dispersion_pvals)), ")"
    )
  }
  results
}

varreg_print_validation_results <- function(bootstrap_summary, one_stage_results,
                                            top_n = 10L) {
  cat("\n========== TARGETED UNCERTAINTY VALIDATION ==========\n")
  if (nrow(bootstrap_summary)) {
    cat("Independent-object bootstrap (NB mean and Gamma stage both refitted):\n")
    display_cols <- intersect(c(
      "names",
      "observed_variance_ratio_per_sd",
      "bootstrap_variance_ratio_ci_low",
      "bootstrap_variance_ratio_ci_high",
      "bootstrap_sign_stability",
      "bootstrap_reps_successful"
    ), colnames(bootstrap_summary))
    display <- utils::head(bootstrap_summary[, display_cols, drop = FALSE], top_n)
    numeric_cols <- vapply(display, is.numeric, logical(1))
    display[numeric_cols] <- lapply(display[numeric_cols], signif, digits = 4L)
    print(display, row.names = FALSE)
  } else {
    cat("Bootstrap: <not run or unavailable>\n")
  }
  if (nrow(one_stage_results)) {
    cat("\nOne-stage NB mean/dispersion validation:\n")
    display_cols <- intersect(c(
      "names",
      "status",
      "backend",
      "conditional_mean_rate_ratio_per_sd",
      "overdispersion_ratio_per_sd",
      "dispersion_pvals_adj",
      "significant_dispersion_validation"
    ), colnames(one_stage_results))
    display <- utils::head(one_stage_results[, display_cols, drop = FALSE], top_n)
    numeric_cols <- vapply(display, is.numeric, logical(1))
    display[numeric_cols] <- lapply(display[numeric_cols], signif, digits = 4L)
    print(display, row.names = FALSE)
  } else {
    cat("One-stage NB model: <not run>\n")
  }
  cat("=====================================================\n")
  invisible(NULL)
}

varreg_fit_extreme_octile_contrast <- function(
    counts, size_factors, cd, predictor, morphology_raw,
    object_type, morphology_feature, cell_line, slide_col,
    slide_covariate = TRUE, min_expr_obs = 20L,
    alpha_padj = 0.05, chunk_size = 500L,
    n_bins = 8L, winsor_upper_quantile = 0.99,
    gamma_response_floor = 1e-6, gamma_maxit = 30L,
    gamma_tolerance = 1e-7, dunn_smyth_seed = 1L,
    permutation_reps = 20L, permutation_quantile = 0.95,
    permutation_seed = 1L, permutation_cores = 1L) {
  n_obs_all <- ncol(counts)
  object_ids <- colnames(counts)
  if (is.null(object_ids)) object_ids <- paste0("object_", seq_len(n_obs_all))

  # Stable rank-based bins give near-equal group sizes. Ties are resolved by
  # object order only to keep the requested bottom/top 12.5% sample sizes equal.
  predictor_rank <- rank(predictor, ties.method = "first")
  octile <- pmin(
    as.integer(n_bins),
    pmax(1L, ceiling(as.integer(n_bins) * predictor_rank / n_obs_all))
  )
  extreme_keep <- octile %in% c(1L, as.integer(n_bins))
  high_group <- as.numeric(octile[extreme_keep] == as.integer(n_bins))
  morphology_extreme <- morphology_raw[extreme_keep]
  cd_extreme <- cd[extreme_keep, , drop = FALSE]
  counts_extreme_all <- counts[, extreme_keep, drop = FALSE]
  size_factors_extreme <- size_factors[extreme_keep]

  low_columns <- high_group == 0
  high_columns <- high_group == 1
  n_expr_low_all <- Matrix::rowSums(counts_extreme_all[, low_columns, drop = FALSE] > 0)
  n_expr_high_all <- Matrix::rowSums(counts_extreme_all[, high_columns, drop = FALSE] > 0)
  keep_genes <- n_expr_low_all >= min_expr_obs & n_expr_high_all >= min_expr_obs
  counts_extreme <- counts_extreme_all[keep_genes, , drop = FALSE]
  if (nrow(counts_extreme) < 1L) {
    stop(
      "No genes had at least ", min_expr_obs,
      " detected observations in both Q1 and Q", n_bins, "."
    )
  }

  design_info <- nbglm_build_design(
    col_data = cd_extreme,
    predictor = high_group,
    slide_col = slide_col,
    slide_covariate = slide_covariate
  )
  design <- design_info$design
  predictor_idx <- which(colnames(design) == design_info$predictor_coef)
  if (length(predictor_idx) != 1L || qr(design)$rank < ncol(design)) {
    stop("Invalid Q1-versus-Q", n_bins, " design matrix.")
  }

  permuted_designs <- varreg_make_permuted_designs(
    cd = cd_extreme,
    predictor = high_group,
    slide_col = slide_col,
    slide_covariate = slide_covariate,
    reps = permutation_reps,
    seed = permutation_seed
  )

  # Refit the NB mean model specifically in the two extreme groups. This allows
  # a separate mean for Q1 and Q8 before residual variability is compared.
  y <- edgeR::DGEList(counts = counts_extreme)
  y$samples$lib.size <- as.numeric(size_factors_extreme)
  y$samples$norm.factors <- rep(1, ncol(counts_extreme))
  y <- nbglm_estimate_disp(y, design, object_type = object_type)
  mean_fit <- edgeR::glmFit(y, design)
  mean_lrt <- edgeR::glmLRT(mean_fit, coef = predictor_idx)
  fitted_mean <- mean_fit$fitted.values
  dispersion <- as.numeric(mean_fit$dispersion)
  if (length(dispersion) == 1L) dispersion <- rep(dispersion, nrow(counts_extreme))

  n_genes <- nrow(counts_extreme)
  primary_gamma <- primary_se <- primary_p <- rep(NA_real_, n_genes)
  primary_converged <- rep(FALSE, n_genes)
  deviance_gamma <- deviance_p <- rep(NA_real_, n_genes)
  deviance_converged <- rep(FALSE, n_genes)
  dunn_gamma <- dunn_p <- rep(NA_real_, n_genes)
  dunn_converged <- rep(FALSE, n_genes)
  permutation_beta <- matrix(
    NA_real_, nrow = n_genes, ncol = length(permuted_designs),
    dimnames = list(rownames(counts_extreme), paste0("perm_", seq_along(permuted_designs)))
  )

  set.seed(as.integer(dunn_smyth_seed))
  for (chunk_start in seq.int(1L, n_genes, by = chunk_size)) {
    chunk_idx <- seq.int(chunk_start, min(chunk_start + chunk_size - 1L, n_genes))
    message(
      "[", object_type, "/", morphology_feature, "/", cell_line,
      "] Q1-vs-Q", n_bins, " variability genes ", min(chunk_idx), "-",
      max(chunk_idx), "/", n_genes
    )
    count_chunk <- as.matrix(counts_extreme[chunk_idx, , drop = FALSE])
    mu_chunk <- fitted_mean[chunk_idx, , drop = FALSE]
    phi_chunk <- dispersion[chunk_idx]
    nb_variance <- mu_chunk + phi_chunk * mu_chunk^2
    pearson_r2 <- (count_chunk - mu_chunk)^2 /
      pmax(nb_variance, .Machine$double.eps)
    primary_response <- varreg_winsorize_rows(
      pearson_r2,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )$response
    primary_fit <- varreg_gamma_log_hc3(
      primary_response, design, predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit, tolerance = gamma_tolerance
    )
    primary_gamma[chunk_idx] <- primary_fit$beta
    primary_se[chunk_idx] <- primary_fit$hc3_se
    primary_p[chunk_idx] <- primary_fit$pvals
    primary_converged[chunk_idx] <- primary_fit$converged

    abs_deviance <- varreg_nb_abs_deviance_residual(count_chunk, mu_chunk, phi_chunk)
    deviance_response <- varreg_winsorize_rows(
      abs_deviance,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )$response
    deviance_fit <- varreg_gamma_log_hc3(
      deviance_response, design, predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit, tolerance = gamma_tolerance
    )
    deviance_gamma[chunk_idx] <- deviance_fit$beta
    deviance_p[chunk_idx] <- deviance_fit$pvals
    deviance_converged[chunk_idx] <- deviance_fit$converged

    dunn_residual <- varreg_nb_dunn_smyth_residual(count_chunk, mu_chunk, phi_chunk)
    dunn_response <- varreg_winsorize_rows(
      dunn_residual^2,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )$response
    dunn_fit <- varreg_gamma_log_hc3(
      dunn_response, design, predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit, tolerance = gamma_tolerance
    )
    dunn_gamma[chunk_idx] <- dunn_fit$beta
    dunn_p[chunk_idx] <- dunn_fit$pvals
    dunn_converged[chunk_idx] <- dunn_fit$converged

    if (length(permuted_designs)) {
      fit_one_permutation <- function(permutation_idx) {
        varreg_gamma_log_hc3(
          primary_response,
          permuted_designs[[permutation_idx]]$design,
          permuted_designs[[permutation_idx]]$predictor_idx,
          response_floor = gamma_response_floor,
          maxit = gamma_maxit,
          tolerance = gamma_tolerance,
          compute_se = FALSE
        )$beta
      }
      permutation_fits <- if (permutation_cores > 1L && .Platform$OS.type == "unix") {
        parallel::mclapply(
          seq_along(permuted_designs), fit_one_permutation,
          mc.cores = min(permutation_cores, length(permuted_designs)),
          mc.preschedule = TRUE, mc.set.seed = FALSE
        )
      } else {
        lapply(seq_along(permuted_designs), fit_one_permutation)
      }
      for (permutation_idx in seq_along(permutation_fits)) {
        permutation_beta[chunk_idx, permutation_idx] <- permutation_fits[[permutation_idx]]
      }
    }
  }

  primary_p_adj <- stats::p.adjust(primary_p, method = "BH")
  deviance_p_adj <- stats::p.adjust(deviance_p, method = "BH")
  dunn_p_adj <- stats::p.adjust(dunn_p, method = "BH")
  if (ncol(permutation_beta) > 0L) {
    permutation_cutoff <- apply(
      abs(permutation_beta), 1L, stats::quantile,
      probs = permutation_quantile, na.rm = TRUE, names = FALSE, type = 8
    )
    permutation_effect_pass <- abs(primary_gamma) > permutation_cutoff
    permutation_empirical_p <- (
      1 + rowSums(abs(permutation_beta) >= abs(primary_gamma), na.rm = TRUE)
    ) / (1 + rowSums(is.finite(permutation_beta)))
  } else {
    permutation_cutoff <- rep(NA_real_, n_genes)
    permutation_effect_pass <- rep(TRUE, n_genes)
    permutation_empirical_p <- rep(NA_real_, n_genes)
  }
  significant <- is.finite(primary_p_adj) & primary_p_adj < alpha_padj &
    permutation_effect_pass & primary_converged

  mean_beta <- mean_fit$coefficients[, predictor_idx]
  expression_summary <- nbglm_expression_summary(counts_extreme, size_factors_extreme)
  results <- data.frame(
    names = rownames(counts_extreme),
    object_type = object_type,
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    comparison = paste0("Q", n_bins, "_vs_Q1"),
    n_quantile_bins = n_bins,
    n_low_group = sum(high_group == 0),
    n_high_group = sum(high_group == 1),
    low_group_morphology_min = min(morphology_extreme[high_group == 0], na.rm = TRUE),
    low_group_morphology_mean = mean(morphology_extreme[high_group == 0], na.rm = TRUE),
    low_group_morphology_max = max(morphology_extreme[high_group == 0], na.rm = TRUE),
    high_group_morphology_min = min(morphology_extreme[high_group == 1], na.rm = TRUE),
    high_group_morphology_mean = mean(morphology_extreme[high_group == 1], na.rm = TRUE),
    high_group_morphology_max = max(morphology_extreme[high_group == 1], na.rm = TRUE),
    variance_ratio_high_vs_low = exp(primary_gamma),
    variance_percent_change_high_vs_low = 100 * (exp(primary_gamma) - 1),
    gamma_log_variance_ratio_high_vs_low = primary_gamma,
    gamma_hc3_se = primary_se,
    pvals = primary_p,
    pvals_adj = primary_p_adj,
    permutation_reps = ncol(permutation_beta),
    permutation_quantile = permutation_quantile,
    permutation_abs_log_ratio_cutoff = permutation_cutoff,
    permutation_effect_pass = permutation_effect_pass,
    permutation_empirical_p = permutation_empirical_p,
    significant = significant,
    variability_direction = ifelse(
      primary_gamma > 0,
      paste0("increased_in_Q", n_bins),
      paste0("decreased_in_Q", n_bins)
    ),
    abs_deviance_ratio_high_vs_low = exp(deviance_gamma),
    abs_deviance_pvals = deviance_p,
    abs_deviance_pvals_adj = deviance_p_adj,
    abs_deviance_significant_fdr = is.finite(deviance_p_adj) &
      deviance_p_adj < alpha_padj & deviance_converged,
    dunn_smyth_variance_ratio_high_vs_low = exp(dunn_gamma),
    dunn_smyth_pvals = dunn_p,
    dunn_smyth_pvals_adj = dunn_p_adj,
    dunn_smyth_significant_fdr = is.finite(dunn_p_adj) &
      dunn_p_adj < alpha_padj & dunn_converged,
    sensitivity_effect_direction_concordant =
      sign(primary_gamma) == sign(deviance_gamma) &
      sign(primary_gamma) == sign(dunn_gamma),
    mean_model_rate_ratio_high_vs_low = exp(mean_beta),
    mean_model_log2_rate_ratio_high_vs_low = mean_beta / log(2),
    mean_model_pvals = mean_lrt$table$PValue,
    mean_model_pvals_adj = stats::p.adjust(mean_lrt$table$PValue, method = "BH"),
    min_expr_obs_required_per_group = min_expr_obs,
    n_expr_obs_low_group = as.numeric(n_expr_low_all[keep_genes]),
    n_expr_obs_high_group = as.numeric(n_expr_high_all[keep_genes]),
    n_expr_obs_extreme_groups = as.numeric(Matrix::rowSums(counts_extreme > 0)),
    used_slide_covariate = design_info$used_slide,
    stringsAsFactors = FALSE
  )
  results <- merge(results, expression_summary, by = "names", sort = FALSE)
  results <- results[match(rownames(counts_extreme), results$names), , drop = FALSE]

  membership <- data.frame(
    object_id = object_ids,
    object_type = object_type,
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    morphology_raw = morphology_raw,
    predictor_z_within_line = predictor,
    quantile_bin = octile,
    quantile_label = paste0("Q", octile),
    included_in_extreme_contrast = extreme_keep,
    extreme_group = ifelse(
      octile == 1L, "Q1_low",
      ifelse(octile == as.integer(n_bins), paste0("Q", n_bins, "_high"), "middle_not_used")
    ),
    stringsAsFactors = FALSE
  )
  list(results = results, membership = membership)
}

varreg_print_extreme_octile_results <- function(results, alpha_padj = 0.05,
                                                 top_n = 10L) {
  if (is.null(results) || !nrow(results)) return(invisible(NULL))
  significant <- results$significant %in% TRUE
  increased <- significant & results$variance_ratio_high_vs_low > 1
  decreased <- significant & results$variance_ratio_high_vs_low < 1
  cat(
    "\n========== EXTREME OCTILE VARIABILITY: ", unique(results$cell_line), " ==========\n",
    sep = ""
  )
  cat(
    "Comparison: Q1 lowest burden (n=", unique(results$n_low_group),
    ") versus Q8 highest burden (n=", unique(results$n_high_group), ").\n",
    sep = ""
  )
  cat(
    "Raw morphology Q1 range/mean: ",
    signif(unique(results$low_group_morphology_min), 4L), "-",
    signif(unique(results$low_group_morphology_max), 4L), " / ",
    signif(unique(results$low_group_morphology_mean), 4L), "\n",
    sep = ""
  )
  cat(
    "Raw morphology Q8 range/mean: ",
    signif(unique(results$high_group_morphology_min), 4L), "-",
    signif(unique(results$high_group_morphology_max), 4L), " / ",
    signif(unique(results$high_group_morphology_mean), 4L), "\n",
    sep = ""
  )
  cat("Genes tested:                       ", nrow(results), "\n", sep = "")
  cat("Significant Q8-vs-Q1 variability:   ", sum(significant), "\n", sep = "")
  cat("  More variable in Q8:              ", sum(increased), "\n", sep = "")
  cat("  Less variable in Q8:              ", sum(decreased), "\n", sep = "")
  cat(
    "Dunn-Smyth sensitivity FDR hits:  ",
    sum(results$dunn_smyth_significant_fdr, na.rm = TRUE), "\n",
    sep = ""
  )
  cat(
    "Absolute-deviance sensitivity hits:",
    sum(results$abs_deviance_significant_fdr, na.rm = TRUE), "\n",
    sep = ""
  )
  cat(
    "Primary hits concordant in all residual methods: ",
    sum(significant & results$sensitivity_effect_direction_concordant, na.rm = TRUE),
    "\n",
    sep = ""
  )
  cat(
    "Significance requires BH p < ", alpha_padj,
    " plus the within-slide permutation effect cutoff.\n",
    sep = ""
  )

  print_subset <- function(keep, title) {
    cat("\n", title, "\n", sep = "")
    subset <- results[keep, , drop = FALSE]
    if (!nrow(subset)) {
      cat("  <none>\n")
      return(invisible(NULL))
    }
    subset <- subset[order(subset$pvals_adj, subset$pvals), , drop = FALSE]
    columns <- c(
      "names", "variance_ratio_high_vs_low", "pvals", "pvals_adj",
      "permutation_effect_pass", "dunn_smyth_variance_ratio_high_vs_low",
      "dunn_smyth_pvals_adj", "abs_deviance_ratio_high_vs_low",
      "mean_model_rate_ratio_high_vs_low", "mean_model_pvals_adj", "pct_positive"
    )
    display <- utils::head(subset[, columns, drop = FALSE], top_n)
    numeric_cols <- vapply(display, is.numeric, logical(1))
    display[numeric_cols] <- lapply(display[numeric_cols], signif, digits = 4L)
    print(display, row.names = FALSE)
    invisible(NULL)
  }
  print_subset(increased, paste0("Top genes MORE variable in Q8 (up to ", top_n, ")"))
  print_subset(decreased, paste0("Top genes LESS variable in Q8 (up to ", top_n, ")"))
  if (!any(significant)) {
    print_subset(
      is.finite(results$pvals_adj),
      paste0("No primary hits; strongest Q8-vs-Q1 evidence (up to ", top_n, ")")
    )
  }
  cat("=============================================================\n")
  invisible(NULL)
}

varreg_fit_line <- function(sce, object_type, morphology_feature,
                            predictor_mode, global_rank_feature_patterns,
                            global_x = NULL, global_info = NULL,
                            standardize_predictor_within_line = TRUE,
                            slide_covariate = TRUE,
                            min_obs_per_cell_line = 100L,
                            min_expr_obs = 20L,
                            alpha_padj = 0.05,
                            chunk_size = 500L,
                            global_min_pct_positive = 5,
                            global_outlier_abs_threshold = 2,
                            winsor_upper_quantile = 0.99,
                            gamma_response_floor = 1e-6,
                            gamma_maxit = 30L,
                            gamma_tolerance = 1e-7,
                            dunn_smyth_seed = 20260803L,
                            instability_trim = 0.10,
                            permutation_reps = 20L,
                            permutation_quantile = 0.95,
                            permutation_seed = 20260804L,
                            permutation_cores = 1L,
                            bootstrap_reps = 30L,
                            bootstrap_top_n = 20L,
                            bootstrap_seed = 20260805L,
                            bootstrap_stratify_by_slide = TRUE,
                            run_one_stage_validation = TRUE,
                            one_stage_top_n = 20L,
                            run_extreme_octile_contrast = TRUE,
                            extreme_n_bins = 8L,
                            extreme_permutation_reps = 20L,
                            extreme_permutation_quantile = 0.95,
                            extreme_permutation_seed = 20260806L) {
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  cols <- nbglm_resolve_metadata_cols(sce)
  cell_line <- unique(as.character(cd[[cols$cell_line_col]]))
  if (length(cell_line) != 1L) stop("varreg_fit_line expects exactly one cell line.")

  make_skip <- function(reason, details = "") {
    list(
      results = NULL,
      summary = NULL,
      skip = nbglm_skip_row(
        object_type, "gene_variability", morphology_feature, cell_line,
        reason = reason, details = details
      )
    )
  }
  if (ncol(sce) < min_obs_per_cell_line) {
    return(make_skip(
      "too_few_observations",
      paste0("n_obs=", ncol(sce), "; min_obs_per_cell_line=", min_obs_per_cell_line)
    ))
  }

  mode_info <- nbglm_resolve_predictor_mode(
    morphology_feature,
    predictor_mode = predictor_mode,
    global_rank_feature_patterns = global_rank_feature_patterns
  )
  pred <- tryCatch(
    nbglm_prepare_model_predictor(
      cd[[morphology_feature]],
      mode_info = mode_info,
      global_x = global_x,
      global_info = global_info
    ),
    error = function(e) e
  )
  if (inherits(pred, "error")) {
    return(make_skip("invalid_predictor", conditionMessage(pred)))
  }

  counts_all <- nbglm_get_counts(sce, paste(object_type, cell_line))
  sf <- nbglm_make_size_factors(counts_all)
  if (is.null(sf$size_factors)) return(make_skip("invalid_size_factors"))

  keep_obs <- is.finite(sf$size_factors) & sf$size_factors > 0 & is.finite(pred$x)
  sce <- sce[, keep_obs, drop = FALSE]
  counts_all <- counts_all[, keep_obs, drop = FALSE]
  size_factors <- sf$size_factors[keep_obs]
  pred$x <- pred$x[keep_obs]
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  if (ncol(sce) < min_obs_per_cell_line) {
    return(make_skip(
      "too_few_observations_after_size_factor_filter",
      paste0("n_obs=", ncol(sce), "; min_obs_per_cell_line=", min_obs_per_cell_line)
    ))
  }

  predictor_before_within_line_zscore <- pred$x
  predictor_z_center <- 0
  predictor_z_scale <- 1
  if (isTRUE(standardize_predictor_within_line)) {
    standardized <- tryCatch(
      varreg_zscore(pred$x, label = paste0(cell_line, " morphology predictor")),
      error = function(e) e
    )
    if (inherits(standardized, "error")) {
      return(make_skip("invalid_within_line_standardization", conditionMessage(standardized)))
    }
    pred$x <- standardized$x
    predictor_z_center <- standardized$center
    predictor_z_scale <- standardized$scale
  }

  keep_genes <- Matrix::rowSums(counts_all > 0) >= min_expr_obs
  if (!any(keep_genes)) {
    return(make_skip("no_features_pass_min_expr_obs", paste0("min_expr_obs=", min_expr_obs)))
  }
  counts <- counts_all[keep_genes, , drop = FALSE]

  design_info <- nbglm_build_design(
    col_data = cd,
    predictor = pred$x,
    slide_col = cols$slide_col,
    slide_covariate = slide_covariate
  )
  design <- design_info$design
  predictor_idx <- which(colnames(design) == design_info$predictor_coef)
  if (length(predictor_idx) != 1L) stop("Could not identify the morphology coefficient.")
  if (qr(design)$rank < ncol(design)) stop("The model design matrix is not full rank.")

  permuted_designs <- varreg_make_permuted_designs(
    cd = cd,
    predictor = pred$x,
    slide_col = cols$slide_col,
    slide_covariate = slide_covariate,
    reps = permutation_reps,
    seed = permutation_seed
  )

  # Stage 1: negative-binomial mean model.
  y <- edgeR::DGEList(counts = counts)
  y$samples$lib.size <- as.numeric(size_factors)
  y$samples$norm.factors <- rep(1, ncol(counts))
  y <- nbglm_estimate_disp(y, design, object_type = object_type)
  mean_fit <- edgeR::glmFit(y, design)
  mean_lrt <- edgeR::glmLRT(mean_fit, coef = predictor_idx)

  fitted_mean <- mean_fit$fitted.values
  dispersion <- as.numeric(mean_fit$dispersion)
  if (length(dispersion) == 1L) dispersion <- rep(dispersion, nrow(counts))
  if (length(dispersion) != nrow(counts)) {
    stop("Unexpected edgeR dispersion length: ", length(dispersion))
  }

  # Components shared by the legacy linear diagnostic and Gamma-HC3 models.
  xtx_inv <- chol2inv(chol(crossprod(design)))
  hat_values <- rowSums((design %*% xtx_inv) * design)
  hc3_denom <- pmax(1 - hat_values, 1e-6)
  predictor_influence <- drop(design %*% xtx_inv[, predictor_idx])
  hc3_weights <- (predictor_influence / hc3_denom)^2
  residual_df <- nrow(design) - ncol(design)
  if (residual_df < 1L) stop("No residual degrees of freedom in auxiliary regression.")

  legacy_beta_variability <- rep(NA_real_, nrow(counts))
  legacy_hc3_se <- rep(NA_real_, nrow(counts))
  mean_residual_sq <- rep(NA_real_, nrow(counts))
  primary_winsor_cap <- rep(NA_real_, nrow(counts))

  gamma_log_variability <- rep(NA_real_, nrow(counts))
  gamma_hc3_se <- rep(NA_real_, nrow(counts))
  gamma_statistic <- rep(NA_real_, nrow(counts))
  gamma_pvals <- rep(NA_real_, nrow(counts))
  gamma_converged <- rep(FALSE, nrow(counts))

  abs_deviance_gamma <- rep(NA_real_, nrow(counts))
  abs_deviance_hc3_se <- rep(NA_real_, nrow(counts))
  abs_deviance_pvals <- rep(NA_real_, nrow(counts))
  abs_deviance_converged <- rep(FALSE, nrow(counts))

  dunn_smyth_gamma <- rep(NA_real_, nrow(counts))
  dunn_smyth_hc3_se <- rep(NA_real_, nrow(counts))
  dunn_smyth_pvals <- rep(NA_real_, nrow(counts))
  dunn_smyth_converged <- rep(FALSE, nrow(counts))

  permutation_beta <- matrix(
    NA_real_,
    nrow = nrow(counts),
    ncol = length(permuted_designs),
    dimnames = list(rownames(counts), paste0("perm_", seq_along(permuted_designs)))
  )

  gene_detection_pct <- as.numeric(Matrix::rowMeans(counts > 0) * 100)
  global_gene_mask <- gene_detection_pct >= global_min_pct_positive
  n_global_genes <- sum(global_gene_mask)
  if (n_global_genes < 2L) {
    stop(
      "Fewer than 2 genes passed global_min_pct_positive=",
      global_min_pct_positive,
      "."
    )
  }
  global_sum_residual_sq <- numeric(ncol(counts))
  global_sum_residual_sq_log <- numeric(ncol(counts))
  global_outlier_count <- integer(ncol(counts))
  global_outlier_sq_threshold <- global_outlier_abs_threshold^2
  global_gene_indices <- which(global_gene_mask)
  global_dunn_smyth_residual <- matrix(
    NA_real_,
    nrow = n_global_genes,
    ncol = ncol(counts),
    dimnames = list(rownames(counts)[global_gene_indices], colnames(counts))
  )

  set.seed(as.integer(dunn_smyth_seed))
  # Chunking limits peak memory while keeping the Gamma IRLS fits vectorized.
  for (chunk_start in seq.int(1L, nrow(counts), by = chunk_size)) {
    chunk_idx <- seq.int(chunk_start, min(chunk_start + chunk_size - 1L, nrow(counts)))
    message(
      "[", object_type, "/", morphology_feature, "/", cell_line,
      "] robust variability genes ", min(chunk_idx), "-", max(chunk_idx),
      "/", nrow(counts)
    )
    count_chunk <- as.matrix(counts[chunk_idx, , drop = FALSE])
    mu_chunk <- fitted_mean[chunk_idx, , drop = FALSE]
    phi_chunk <- dispersion[chunk_idx]
    nb_variance <- mu_chunk + phi_chunk * mu_chunk^2
    pearson_residual_sq <-
      (count_chunk - mu_chunk)^2 / pmax(nb_variance, .Machine$double.eps)

    # Legacy linear Pearson-r^2 analysis retained only for comparison.
    coef_chunk <- (pearson_residual_sq %*% design) %*% xtx_inv
    aux_residual <- pearson_residual_sq - coef_chunk %*% t(design)
    legacy_beta_variability[chunk_idx] <- coef_chunk[, predictor_idx]
    hc3_var <- drop((aux_residual^2) %*% hc3_weights)
    legacy_hc3_se[chunk_idx] <- sqrt(pmax(hc3_var, 0))
    mean_residual_sq[chunk_idx] <- rowMeans(pearson_residual_sq)

    # Primary analysis: Gamma(log) on upper-winsorized Pearson r^2.
    primary_response <- varreg_winsorize_rows(
      pearson_residual_sq,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )
    primary_winsor_cap[chunk_idx] <- primary_response$cap
    primary_fit <- varreg_gamma_log_hc3(
      response = primary_response$response,
      design = design,
      predictor_idx = predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit,
      tolerance = gamma_tolerance
    )
    gamma_log_variability[chunk_idx] <- primary_fit$beta
    gamma_hc3_se[chunk_idx] <- primary_fit$hc3_se
    gamma_statistic[chunk_idx] <- primary_fit$statistic
    gamma_pvals[chunk_idx] <- primary_fit$pvals
    gamma_converged[chunk_idx] <- primary_fit$converged

    # Sensitivity 1: absolute NB deviance residuals under Gamma(log).
    abs_deviance <- varreg_nb_abs_deviance_residual(
      count_chunk, mu_chunk, phi_chunk
    )
    deviance_response <- varreg_winsorize_rows(
      abs_deviance,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )$response
    deviance_fit <- varreg_gamma_log_hc3(
      response = deviance_response,
      design = design,
      predictor_idx = predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit,
      tolerance = gamma_tolerance
    )
    abs_deviance_gamma[chunk_idx] <- deviance_fit$beta
    abs_deviance_hc3_se[chunk_idx] <- deviance_fit$hc3_se
    abs_deviance_pvals[chunk_idx] <- deviance_fit$pvals
    abs_deviance_converged[chunk_idx] <- deviance_fit$converged

    # Sensitivity 2: randomized-quantile residual variance under Gamma(log).
    dunn_smyth_residual <- varreg_nb_dunn_smyth_residual(
      count_chunk, mu_chunk, phi_chunk
    )
    dunn_smyth_response <- varreg_winsorize_rows(
      dunn_smyth_residual^2,
      upper_quantile = winsor_upper_quantile,
      floor = gamma_response_floor
    )$response
    dunn_smyth_fit <- varreg_gamma_log_hc3(
      response = dunn_smyth_response,
      design = design,
      predictor_idx = predictor_idx,
      response_floor = gamma_response_floor,
      maxit = gamma_maxit,
      tolerance = gamma_tolerance
    )
    dunn_smyth_gamma[chunk_idx] <- dunn_smyth_fit$beta
    dunn_smyth_hc3_se[chunk_idx] <- dunn_smyth_fit$hc3_se
    dunn_smyth_pvals[chunk_idx] <- dunn_smyth_fit$pvals
    dunn_smyth_converged[chunk_idx] <- dunn_smyth_fit$converged

    # Empirical technical-null effects from morphology shuffled within slide.
    if (length(permuted_designs)) {
      fit_one_permutation <- function(permutation_idx) {
        permutation_fit <- varreg_gamma_log_hc3(
          response = primary_response$response,
          design = permuted_designs[[permutation_idx]]$design,
          predictor_idx = permuted_designs[[permutation_idx]]$predictor_idx,
          response_floor = gamma_response_floor,
          maxit = gamma_maxit,
          tolerance = gamma_tolerance,
          compute_se = FALSE
        )
        permutation_fit$beta
      }
      permutation_fits <- if (permutation_cores > 1L && .Platform$OS.type == "unix") {
        parallel::mclapply(
          seq_along(permuted_designs),
          fit_one_permutation,
          mc.cores = min(permutation_cores, length(permuted_designs)),
          mc.preschedule = TRUE,
          mc.set.seed = FALSE
        )
      } else {
        lapply(seq_along(permuted_designs), fit_one_permutation)
      }
      for (permutation_idx in seq_along(permutation_fits)) {
        permutation_beta[chunk_idx, permutation_idx] <- permutation_fits[[permutation_idx]]
      }
    }

    metric_gene_keep <- global_gene_mask[chunk_idx]
    if (any(metric_gene_keep)) {
      metric_residual_sq <- pearson_residual_sq[metric_gene_keep, , drop = FALSE]
      global_sum_residual_sq <- global_sum_residual_sq + colSums(metric_residual_sq)
      global_outlier_count <- global_outlier_count +
        colSums(metric_residual_sq >= global_outlier_sq_threshold)
      residual_sq_log_term <- metric_residual_sq
      positive <- residual_sq_log_term > 0
      residual_sq_log_term[positive] <-
        residual_sq_log_term[positive] * log(residual_sq_log_term[positive])
      residual_sq_log_term[!positive] <- 0
      global_sum_residual_sq_log <- global_sum_residual_sq_log +
        colSums(residual_sq_log_term)

      global_rows <- match(chunk_idx[metric_gene_keep], global_gene_indices)
      global_dunn_smyth_residual[global_rows, ] <-
        dunn_smyth_residual[metric_gene_keep, , drop = FALSE]
    }
  }

  normalized_entropy <- rep(NA_real_, ncol(counts))
  positive_sum <- is.finite(global_sum_residual_sq) & global_sum_residual_sq > 0
  normalized_entropy[positive_sum] <- (
    log(global_sum_residual_sq[positive_sum]) -
      global_sum_residual_sq_log[positive_sum] /
        global_sum_residual_sq[positive_sum]
  ) / log(n_global_genes)
  normalized_entropy[positive_sum] <- pmin(
    pmax(normalized_entropy[positive_sum], 0),
    1
  )

  message(
    "[", object_type, "/", morphology_feature, "/", cell_line,
    "] calculating Dunn-Smyth transcriptomic-instability scores"
  )
  median_abs_dunn_smyth <- apply(
    abs(global_dunn_smyth_residual), 2L, stats::median, na.rm = TRUE
  )
  trimmed_mean_squared_dunn_smyth <- apply(
    global_dunn_smyth_residual^2,
    2L,
    mean,
    trim = instability_trim,
    na.rm = TRUE
  )

  object_ids <- colnames(sce)
  if (is.null(object_ids)) object_ids <- paste0("object_", seq_len(ncol(sce)))
  global_scores <- data.frame(
    object_id = object_ids,
    object_type = object_type,
    cell_line = cell_line,
    morphology_feature = morphology_feature,
    morphology_raw = suppressWarnings(as.numeric(as.character(cd[[morphology_feature]]))),
    predictor = pred$x,
    predictor_z_within_line = pred$x,
    median_abs_dunn_smyth_residual = median_abs_dunn_smyth,
    trimmed_mean_squared_dunn_smyth_residual = trimmed_mean_squared_dunn_smyth,
    mean_squared_pearson_residual = global_sum_residual_sq / n_global_genes,
    n_abs_pearson_outliers = global_outlier_count,
    fraction_abs_pearson_outliers = global_outlier_count / n_global_genes,
    normalized_residual_entropy = normalized_entropy,
    n_genes_in_score = n_global_genes,
    global_min_pct_positive = global_min_pct_positive,
    outlier_abs_pearson_threshold = global_outlier_abs_threshold,
    dunn_smyth_seed = dunn_smyth_seed,
    instability_trim = instability_trim,
    stringsAsFactors = FALSE
  )
  if (!is.na(cols$slide_col)) global_scores$slide <- as.character(cd[[cols$slide_col]])

  global_results <- varreg_fit_global_metrics(
    scores = global_scores,
    design = design,
    predictor_idx = predictor_idx,
    object_type = object_type,
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    alpha_padj = alpha_padj,
    gamma_response_floor = gamma_response_floor,
    gamma_maxit = gamma_maxit,
    gamma_tolerance = gamma_tolerance
  )
  global_bins <- varreg_make_global_bins(global_scores)

  pvals <- gamma_pvals
  pvals_adj <- stats::p.adjust(pvals, method = "BH")
  abs_deviance_pvals_adj <- stats::p.adjust(abs_deviance_pvals, method = "BH")
  dunn_smyth_pvals_adj <- stats::p.adjust(dunn_smyth_pvals, method = "BH")

  if (ncol(permutation_beta) > 0L) {
    permutation_abs_gamma_cutoff <- apply(
      abs(permutation_beta),
      1L,
      stats::quantile,
      probs = permutation_quantile,
      na.rm = TRUE,
      names = FALSE,
      type = 8
    )
    permutation_effect_pass <-
      abs(gamma_log_variability) > permutation_abs_gamma_cutoff
    permutation_empirical_p <- (
      1 + rowSums(
        abs(permutation_beta) >= abs(gamma_log_variability),
        na.rm = TRUE
      )
    ) / (1 + rowSums(is.finite(permutation_beta)))
  } else {
    permutation_abs_gamma_cutoff <- rep(NA_real_, nrow(counts))
    permutation_effect_pass <- rep(TRUE, nrow(counts))
    permutation_empirical_p <- rep(NA_real_, nrow(counts))
  }

  significant <- !is.na(pvals_adj) & pvals_adj < alpha_padj &
    permutation_effect_pass & gamma_converged
  legacy_robust_t <- legacy_beta_variability / legacy_hc3_se
  legacy_robust_t[!is.finite(legacy_robust_t)] <- NA_real_
  legacy_pvals <- 2 * stats::pt(
    abs(legacy_robust_t), df = residual_df, lower.tail = FALSE
  )

  mean_beta <- mean_fit$coefficients[, predictor_idx]
  mean_pvals <- mean_lrt$table$PValue
  expr_summary <- nbglm_expression_summary(counts, size_factors)
  results <- data.frame(
    object_type = object_type,
    target_type = "gene_variability",
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    names = rownames(counts),
    primary_variability_model = "Gamma(log), HC3, winsorized Pearson r^2",
    gamma_log_variability_per_sd = gamma_log_variability,
    variance_ratio_per_sd = exp(gamma_log_variability),
    variance_percent_change_per_sd = 100 * (exp(gamma_log_variability) - 1),
    beta_variability = gamma_log_variability,
    variability_direction = ifelse(gamma_log_variability > 0, "increased", "decreased"),
    gamma_hc3_se = gamma_hc3_se,
    gamma_hc3_t = gamma_statistic,
    hc3_se = gamma_hc3_se,
    robust_t = gamma_statistic,
    pvals = pvals,
    pvals_adj = pvals_adj,
    permutation_reps = ncol(permutation_beta),
    permutation_quantile = permutation_quantile,
    permutation_abs_log_variance_ratio_cutoff = permutation_abs_gamma_cutoff,
    permutation_effect_pass = permutation_effect_pass,
    permutation_empirical_p = permutation_empirical_p,
    significant = significant,
    abs_deviance_gamma_log_ratio_per_sd = abs_deviance_gamma,
    abs_deviance_ratio_per_sd = exp(abs_deviance_gamma),
    abs_deviance_hc3_se = abs_deviance_hc3_se,
    abs_deviance_pvals = abs_deviance_pvals,
    abs_deviance_pvals_adj = abs_deviance_pvals_adj,
    abs_deviance_gamma_converged = abs_deviance_converged,
    abs_deviance_significant_fdr = !is.na(abs_deviance_pvals_adj) &
      abs_deviance_pvals_adj < alpha_padj & abs_deviance_converged,
    dunn_smyth_gamma_log_variance_ratio_per_sd = dunn_smyth_gamma,
    dunn_smyth_variance_ratio_per_sd = exp(dunn_smyth_gamma),
    dunn_smyth_hc3_se = dunn_smyth_hc3_se,
    dunn_smyth_pvals = dunn_smyth_pvals,
    dunn_smyth_pvals_adj = dunn_smyth_pvals_adj,
    dunn_smyth_gamma_converged = dunn_smyth_converged,
    dunn_smyth_significant_fdr = !is.na(dunn_smyth_pvals_adj) &
      dunn_smyth_pvals_adj < alpha_padj & dunn_smyth_converged,
    sensitivity_effect_direction_concordant =
      sign(gamma_log_variability) == sign(abs_deviance_gamma) &
      sign(gamma_log_variability) == sign(dunn_smyth_gamma),
    gamma_converged = gamma_converged,
    winsor_upper_quantile = winsor_upper_quantile,
    pearson_r2_winsor_cap = primary_winsor_cap,
    legacy_linear_beta_pearson_r2 = legacy_beta_variability,
    legacy_linear_hc3_se = legacy_hc3_se,
    legacy_linear_hc3_t = legacy_robust_t,
    legacy_linear_pvals = legacy_pvals,
    legacy_linear_pvals_adj = stats::p.adjust(legacy_pvals, method = "BH"),
    mean_residual_sq = mean_residual_sq,
    nb_dispersion = dispersion,
    mean_model_beta_predictor = mean_beta,
    mean_model_log2_rate_ratio_per_sd = mean_beta / log(2),
    mean_model_log2_rate_ratio_per_predictor = mean_beta / log(2),
    mean_model_lr_stat = mean_lrt$table$LR,
    mean_model_pvals = mean_pvals,
    mean_model_pvals_adj = stats::p.adjust(mean_pvals, method = "BH"),
    n_obs = ncol(counts),
    n_expr_obs = as.numeric(Matrix::rowSums(counts > 0)),
    predictor_mode = pred$mode,
    predictor_mode_requested = pred$predictor_mode_requested,
    predictor_mode_resolved = pred$predictor_mode_resolved,
    predictor_transform_scope = pred$predictor_transform_scope,
    predictor_standardized_within_line = standardize_predictor_within_line,
    predictor_pre_z_mean = mean(predictor_before_within_line_zscore),
    predictor_pre_z_sd = stats::sd(predictor_before_within_line_zscore),
    predictor_z_center = predictor_z_center,
    predictor_z_scale = predictor_z_scale,
    predictor_raw_mean = pred$raw_mean,
    predictor_raw_sd = pred$raw_sd,
    predictor_raw_min = pred$raw_min,
    predictor_raw_max = pred$raw_max,
    used_slide_covariate = design_info$used_slide,
    design_cols = paste(design_info$design_cols, collapse = ";"),
    residual_df = residual_df,
    max_design_leverage = max(hat_values),
    stringsAsFactors = FALSE
  )
  results <- merge(results, expr_summary, by = "names", sort = FALSE)
  results <- results[match(rownames(counts), results$names), , drop = FALSE]

  extreme_octile <- list(results = data.frame(), membership = data.frame())
  if (isTRUE(run_extreme_octile_contrast)) {
    extreme_octile <- tryCatch(
      varreg_fit_extreme_octile_contrast(
        counts = counts,
        size_factors = size_factors,
        cd = cd,
        predictor = pred$x,
        morphology_raw = suppressWarnings(as.numeric(as.character(cd[[morphology_feature]]))),
        object_type = object_type,
        morphology_feature = morphology_feature,
        cell_line = cell_line,
        slide_col = cols$slide_col,
        slide_covariate = slide_covariate,
        min_expr_obs = min_expr_obs,
        alpha_padj = alpha_padj,
        chunk_size = chunk_size,
        n_bins = extreme_n_bins,
        winsor_upper_quantile = winsor_upper_quantile,
        gamma_response_floor = gamma_response_floor,
        gamma_maxit = gamma_maxit,
        gamma_tolerance = gamma_tolerance,
        dunn_smyth_seed = dunn_smyth_seed + 1000L,
        permutation_reps = extreme_permutation_reps,
        permutation_quantile = extreme_permutation_quantile,
        permutation_seed = extreme_permutation_seed,
        permutation_cores = permutation_cores
      ),
      error = function(e) {
        warning(
          "Q1-versus-Q", extreme_n_bins, " variability contrast failed for ",
          cell_line, ": ", conditionMessage(e)
        )
        list(results = data.frame(), membership = data.frame())
      }
    )
  }

  ranked_gene_idx <- order(results$pvals_adj, results$pvals, na.last = NA)
  validation_n <- max(as.integer(bootstrap_top_n), as.integer(one_stage_top_n), 0L)
  validation_genes <- if (validation_n > 0L) {
    results$names[utils::head(ranked_gene_idx, validation_n)]
  } else {
    character(0)
  }
  observed_gamma <- stats::setNames(
    results$gamma_log_variability_per_sd,
    results$names
  )
  bootstrap_genes <- utils::head(validation_genes, as.integer(bootstrap_top_n))
  bootstrap_results <- varreg_case_bootstrap(
    counts = counts,
    size_factors = size_factors,
    cd = cd,
    predictor = pred$x,
    slide_col = cols$slide_col,
    slide_covariate = slide_covariate,
    selected_genes = bootstrap_genes,
    observed_gamma = observed_gamma,
    reps = bootstrap_reps,
    seed = bootstrap_seed,
    stratify_by_slide = bootstrap_stratify_by_slide,
    winsor_upper_quantile = winsor_upper_quantile,
    gamma_response_floor = gamma_response_floor,
    gamma_maxit = gamma_maxit,
    gamma_tolerance = gamma_tolerance
  )
  one_stage_results <- data.frame()
  if (isTRUE(run_one_stage_validation) && one_stage_top_n > 0L) {
    one_stage_results <- varreg_one_stage_validation(
      counts = counts,
      size_factors = size_factors,
      cd = cd,
      predictor = pred$x,
      slide_col = cols$slide_col,
      slide_covariate = slide_covariate,
      selected_genes = utils::head(validation_genes, as.integer(one_stage_top_n)),
      alpha_padj = alpha_padj
    )
  }

  summary <- data.frame(
    object_type = object_type,
    morphology_feature = morphology_feature,
    cell_line = cell_line,
    n_obs = ncol(counts),
    n_features_tested = nrow(counts),
    n_significant = sum(results$significant, na.rm = TRUE),
    n_increased_variability = sum(
      results$significant & results$gamma_log_variability_per_sd > 0,
      na.rm = TRUE
    ),
    n_decreased_variability = sum(
      results$significant & results$gamma_log_variability_per_sd < 0,
      na.rm = TRUE
    ),
    n_primary_gamma_nonconverged = sum(!results$gamma_converged, na.rm = TRUE),
    n_abs_deviance_sensitivity_fdr = sum(
      results$abs_deviance_significant_fdr, na.rm = TRUE
    ),
    n_dunn_smyth_sensitivity_fdr = sum(
      results$dunn_smyth_significant_fdr, na.rm = TRUE
    ),
    n_bootstrap_genes = nrow(bootstrap_results$summary),
    bootstrap_reps_requested = bootstrap_reps,
    n_one_stage_genes = nrow(one_stage_results),
    n_extreme_octile_genes_tested = nrow(extreme_octile$results),
    n_extreme_octile_significant = if (nrow(extreme_octile$results)) {
      sum(extreme_octile$results$significant, na.rm = TRUE)
    } else {
      0L
    },
    median_permutation_abs_log_ratio_cutoff = if (
      any(is.finite(results$permutation_abs_log_variance_ratio_cutoff))
    ) {
      stats::median(results$permutation_abs_log_variance_ratio_cutoff, na.rm = TRUE)
    } else {
      NA_real_
    },
    n_genes_in_global_stochasticity_scores = n_global_genes,
    n_significant_global_stochasticity_metrics = sum(global_results$significant, na.rm = TRUE),
    predictor_mode = pred$mode,
    used_slide_covariate = design_info$used_slide,
    max_design_leverage = max(hat_values),
    stringsAsFactors = FALSE
  )

  permutation_results <- data.frame(stringsAsFactors = FALSE)
  if (ncol(permutation_beta) > 0L) {
    permutation_results <- data.frame(
      object_type = object_type,
      morphology_feature = morphology_feature,
      cell_line = cell_line,
      names = rep(rownames(permutation_beta), times = ncol(permutation_beta)),
      permutation = rep(seq_len(ncol(permutation_beta)), each = nrow(permutation_beta)),
      gamma_log_variability_per_sd = as.vector(permutation_beta),
      variance_ratio_per_sd = exp(as.vector(permutation_beta)),
      stringsAsFactors = FALSE
    )
  }
  list(
    results = results,
    summary = summary,
    skip = NULL,
    global_results = global_results,
    global_scores = global_scores,
    global_bins = global_bins,
    n_global_genes = n_global_genes,
    permutation_results = permutation_results,
    bootstrap_replicates = bootstrap_results$replicates,
    bootstrap_summary = bootstrap_results$summary,
    one_stage_results = one_stage_results,
    extreme_octile_results = extreme_octile$results,
    extreme_octile_membership = extreme_octile$membership
  )
}

cat("========== COSMX NB VARIABILITY PILOT CONFIG ==========\n")
cat("myonuclei_rds_path:       ", myonuclei_rds_path, "\n", sep = "")
cat("myotube_rds_path:         ", myotube_rds_path, "\n", sep = "")
cat("output_dir:               ", output_dir, "\n", sep = "")
cat("object_type:              ", object_type, "\n", sep = "")
cat("morphology_feature:       ", morphology_feature, "\n", sep = "")
cat("cell_lines_to_include:    ", paste(cell_lines_to_include, collapse = ", "), "\n", sep = "")
cat("sample_n (total/object):  ", sample_n, "\n", sep = "")
cat("sample_seed:              ", sample_seed, "\n", sep = "")
cat("sample_balance_cell_lines:", sample_balance_cell_lines, "\n", sep = "")
cat("allow_missing_cell_lines: ", allow_missing_cell_lines, "\n", sep = "")
cat("predictor_mode:           ", predictor_mode, "\n", sep = "")
cat("standardize_predictor_within_line:", standardize_predictor_within_line, "\n", sep = "")
cat("slide_covariate:          ", slide_covariate, "\n", sep = "")
cat("min_obs_per_cell_line:    ", min_obs_per_cell_line, "\n", sep = "")
cat("min_expr_obs:             ", min_expr_obs, "\n", sep = "")
cat("alpha_padj:               ", alpha_padj, "\n", sep = "")
cat("chunk_size:               ", chunk_size, "\n", sep = "")
cat("console_top_n:            ", console_top_n, "\n", sep = "")
cat("global_min_pct_positive:  ", global_min_pct_positive, "\n", sep = "")
cat("global_outlier_abs_threshold:", global_outlier_abs_threshold, "\n", sep = "")
cat("winsor_upper_quantile:    ", winsor_upper_quantile, "\n", sep = "")
cat("dunn_smyth_seed:          ", dunn_smyth_seed, "\n", sep = "")
cat("instability_trim:         ", instability_trim, "\n", sep = "")
cat("permutation_reps:         ", permutation_reps, "\n", sep = "")
cat("permutation_quantile:     ", permutation_quantile, "\n", sep = "")
cat("permutation_cores:        ", permutation_cores, "\n", sep = "")
cat("bootstrap_reps/top_n:     ", bootstrap_reps, "/", bootstrap_top_n, "\n", sep = "")
cat("bootstrap unit:           independent objects within slide strata\n")
cat("run_one_stage_validation: ", run_one_stage_validation, "\n", sep = "")
cat("one_stage_top_n:          ", one_stage_top_n, "\n", sep = "")
cat("run_extreme_octile_contrast:", run_extreme_octile_contrast, "\n", sep = "")
cat("extreme_n_bins:           ", extreme_n_bins, " (compare Q1 vs Q", extreme_n_bins, ")\n", sep = "")
cat("extreme_permutation_reps: ", extreme_permutation_reps, "\n", sep = "")
cat("extreme_permutation_quantile:", extreme_permutation_quantile, "\n", sep = "")
cat("=======================================================\n\n")

t0 <- Sys.time()
sce <- nbglm_load_sce_for_object(
  object_type = object_type,
  myonuclei_rds_path = myonuclei_rds_path,
  myotube_rds_path = myotube_rds_path,
  min_myonuclei_per_myotube = min_myonuclei_per_myotube,
  filter_myonuclei = filter_myonuclei,
  myonucleus_col = myonucleus_col
)
if (identical(object_type, "myonuclei")) {
  sce <- nbglm_ensure_nuclear_circularity(
    sce,
    requested_features = morphology_feature,
    derive_missing_circularity = derive_missing_circularity
  )
}

cd <- as.data.frame(SummarizedExperiment::colData(sce))
if (!(morphology_feature %in% colnames(cd))) {
  stop(
    "Morphology feature not found: ", morphology_feature,
    ". Available columns: ", paste(colnames(cd), collapse = ", ")
  )
}
cols <- nbglm_resolve_metadata_cols(sce)
observed_cell_lines <- sort(unique(stats::na.omit(as.character(cd[[cols$cell_line_col]]))))
present_requested_lines <- observed_cell_lines[
  normalize_label(observed_cell_lines) %in% normalize_label(cell_lines_to_include)
]
missing_requested_lines <- cell_lines_to_include[
  !(normalize_label(cell_lines_to_include) %in% normalize_label(present_requested_lines))
]
message("Available cell lines in RDS: ", paste(observed_cell_lines, collapse = ", "))
message(
  "Requested cell lines present: ",
  if (length(present_requested_lines)) paste(present_requested_lines, collapse = ", ") else "<none>"
)
if (length(missing_requested_lines)) {
  message("Requested cell lines absent:  ", paste(missing_requested_lines, collapse = ", "))
}
sce <- varreg_sample_sce(
  sce = sce,
  cell_line_col = cols$cell_line_col,
  morphology_feature = morphology_feature,
  requested_lines = cell_lines_to_include,
  sample_n = sample_n,
  sample_seed = sample_seed,
  balance_cell_lines = sample_balance_cell_lines,
  allow_missing_cell_lines = allow_missing_cell_lines
)
cd <- as.data.frame(SummarizedExperiment::colData(sce))
sampled_lines <- sort(unique(as.character(cd[[cols$cell_line_col]])))
sample_counts <- table(cd[[cols$cell_line_col]])
message(
  "Sampled ", ncol(sce), " ", object_type, ": ",
  paste(names(sample_counts), as.integer(sample_counts), sep = "=", collapse = ", ")
)

mode_info <- nbglm_resolve_predictor_mode(
  morphology_feature,
  predictor_mode = predictor_mode,
  global_rank_feature_patterns = global_rank_feature_patterns
)
global_info <- NULL
if (identical(mode_info$resolved, "global_rank")) {
  global_info <- nbglm_make_global_rank_predictor(
    cd[[morphology_feature]], global_mask = rep(TRUE, nrow(cd))
  )
}

condition_name <- paste(
  nbglm_safe_slug(object_type),
  nbglm_safe_slug(morphology_feature),
  "gene_variability",
  sep = "-"
)
condition_out_dir <- file.path(output_dir, condition_name)
dir.create(condition_out_dir, recursive = TRUE, showWarnings = FALSE)

analysis_manifest <- data.frame(
  object_type = object_type,
  morphology_feature = morphology_feature,
  requested_cell_lines = paste(cell_lines_to_include, collapse = ";"),
  present_requested_cell_lines = paste(present_requested_lines, collapse = ";"),
  missing_requested_cell_lines = paste(missing_requested_lines, collapse = ";"),
  sampled_cell_lines = paste(sampled_lines, collapse = ";"),
  sample_n_requested_total = sample_n,
  sample_n_observed_total = ncol(sce),
  sample_counts_by_cell_line = paste(
    names(sample_counts), as.integer(sample_counts), sep = "=", collapse = ";"
  ),
  sample_seed = sample_seed,
  predictor_mode_resolved = mode_info$resolved,
  standardize_predictor_within_line = standardize_predictor_within_line,
  slide_covariate = slide_covariate,
  alpha_padj = alpha_padj,
  global_min_pct_positive = global_min_pct_positive,
  global_outlier_abs_threshold = global_outlier_abs_threshold,
  primary_variability_model = "Gamma(log), HC3, winsorized Pearson r^2",
  winsor_upper_quantile = winsor_upper_quantile,
  gamma_response_floor = gamma_response_floor,
  gamma_maxit = gamma_maxit,
  gamma_tolerance = gamma_tolerance,
  dunn_smyth_seed = dunn_smyth_seed,
  instability_trim = instability_trim,
  permutation_reps = permutation_reps,
  permutation_quantile = permutation_quantile,
  permutation_seed = permutation_seed,
  permutation_cores = permutation_cores,
  bootstrap_reps = bootstrap_reps,
  bootstrap_top_n = bootstrap_top_n,
  bootstrap_seed = bootstrap_seed,
  bootstrap_unit = "independent_objects_within_slide_strata",
  bootstrap_stratify_by_slide = bootstrap_stratify_by_slide,
  run_one_stage_validation = run_one_stage_validation,
  one_stage_top_n = one_stage_top_n,
  run_extreme_octile_contrast = run_extreme_octile_contrast,
  extreme_n_bins = extreme_n_bins,
  extreme_comparison = paste0("Q", extreme_n_bins, "_vs_Q1"),
  extreme_permutation_reps = extreme_permutation_reps,
  extreme_permutation_quantile = extreme_permutation_quantile,
  extreme_permutation_seed = extreme_permutation_seed,
  stringsAsFactors = FALSE
)
nbglm_write_csv(analysis_manifest, file.path(condition_out_dir, "analysis_manifest.csv"))

sample_manifest <- data.frame(
  object_id = colnames(sce),
  object_type = object_type,
  cell_line = as.character(cd[[cols$cell_line_col]]),
  morphology_feature = morphology_feature,
  morphology_value = suppressWarnings(as.numeric(as.character(cd[[morphology_feature]]))),
  sample_seed = sample_seed,
  stringsAsFactors = FALSE
)
if (!is.na(cols$slide_col)) sample_manifest$slide <- as.character(cd[[cols$slide_col]])
nbglm_write_csv(sample_manifest, file.path(condition_out_dir, "sampled_objects.csv"))

all_results <- list()
all_summaries <- list()
all_skips <- list()
all_global_results <- list()
all_global_scores <- list()
all_global_bins <- list()
all_permutation_results <- list()
all_bootstrap_replicates <- list()
all_bootstrap_summaries <- list()
all_one_stage_results <- list()
all_extreme_octile_results <- list()
all_extreme_octile_membership <- list()
for (cell_line in sampled_lines) {
  keep_line <- as.character(cd[[cols$cell_line_col]]) == cell_line
  line_sce <- sce[, keep_line, drop = FALSE]
  line_global_x <- if (!is.null(global_info)) global_info$x[keep_line] else NULL
  line_t0 <- Sys.time()
  message("[", object_type, "/", morphology_feature, "] fitting ", cell_line)
  fit <- tryCatch(
    varreg_fit_line(
      sce = line_sce,
      object_type = object_type,
      morphology_feature = morphology_feature,
      predictor_mode = predictor_mode,
      global_rank_feature_patterns = global_rank_feature_patterns,
      global_x = line_global_x,
      global_info = global_info,
      standardize_predictor_within_line = standardize_predictor_within_line,
      slide_covariate = slide_covariate,
      min_obs_per_cell_line = min_obs_per_cell_line,
      min_expr_obs = min_expr_obs,
      alpha_padj = alpha_padj,
      chunk_size = chunk_size,
      global_min_pct_positive = global_min_pct_positive,
      global_outlier_abs_threshold = global_outlier_abs_threshold,
      winsor_upper_quantile = winsor_upper_quantile,
      gamma_response_floor = gamma_response_floor,
      gamma_maxit = gamma_maxit,
      gamma_tolerance = gamma_tolerance,
      dunn_smyth_seed = dunn_smyth_seed,
      instability_trim = instability_trim,
      permutation_reps = permutation_reps,
      permutation_quantile = permutation_quantile,
      permutation_seed = permutation_seed,
      permutation_cores = permutation_cores,
      bootstrap_reps = bootstrap_reps,
      bootstrap_top_n = bootstrap_top_n,
      bootstrap_seed = bootstrap_seed,
      bootstrap_stratify_by_slide = bootstrap_stratify_by_slide,
      run_one_stage_validation = run_one_stage_validation,
      one_stage_top_n = one_stage_top_n,
      run_extreme_octile_contrast = run_extreme_octile_contrast,
      extreme_n_bins = extreme_n_bins,
      extreme_permutation_reps = extreme_permutation_reps,
      extreme_permutation_quantile = extreme_permutation_quantile,
      extreme_permutation_seed = extreme_permutation_seed
    ),
    error = function(e) list(
      results = NULL,
      summary = NULL,
      skip = nbglm_skip_row(
        object_type, "gene_variability", morphology_feature, cell_line,
        reason = "model_fit_failed", details = conditionMessage(e)
      )
    )
  )
  if (!is.null(fit$results)) all_results[[length(all_results) + 1L]] <- fit$results
  if (!is.null(fit$summary)) {
    fit$summary$elapsed_sec <- as.numeric(difftime(Sys.time(), line_t0, units = "secs"))
    all_summaries[[length(all_summaries) + 1L]] <- fit$summary
  }
  if (!is.null(fit$skip)) {
    all_skips[[length(all_skips) + 1L]] <- fit$skip
    message(
      "[", object_type, "/", morphology_feature, "/", cell_line,
      "] MODEL SKIPPED OR FAILED: ", fit$skip$reason,
      if (nzchar(fit$skip$details)) paste0(" — ", fit$skip$details) else ""
    )
  }
  varreg_print_line_results(fit$results, alpha_padj = alpha_padj, top_n = console_top_n)
  if (!is.null(fit$global_results)) {
    all_global_results[[length(all_global_results) + 1L]] <- fit$global_results
    all_global_scores[[length(all_global_scores) + 1L]] <- fit$global_scores
    all_global_bins[[length(all_global_bins) + 1L]] <- fit$global_bins
    varreg_print_global_results(
      results = fit$global_results,
      bins = fit$global_bins,
      n_genes = fit$n_global_genes,
      min_pct_positive = global_min_pct_positive,
      outlier_abs_threshold = global_outlier_abs_threshold
    )
  }
  if (!is.null(fit$permutation_results) && nrow(fit$permutation_results)) {
    all_permutation_results[[length(all_permutation_results) + 1L]] <- fit$permutation_results
  }
  if (!is.null(fit$bootstrap_replicates) && nrow(fit$bootstrap_replicates)) {
    fit$bootstrap_replicates$object_type <- object_type
    fit$bootstrap_replicates$morphology_feature <- morphology_feature
    fit$bootstrap_replicates$cell_line <- cell_line
    all_bootstrap_replicates[[length(all_bootstrap_replicates) + 1L]] <-
      fit$bootstrap_replicates
  }
  if (!is.null(fit$bootstrap_summary) && nrow(fit$bootstrap_summary)) {
    fit$bootstrap_summary$object_type <- object_type
    fit$bootstrap_summary$morphology_feature <- morphology_feature
    fit$bootstrap_summary$cell_line <- cell_line
    all_bootstrap_summaries[[length(all_bootstrap_summaries) + 1L]] <-
      fit$bootstrap_summary
  }
  if (!is.null(fit$one_stage_results) && nrow(fit$one_stage_results)) {
    fit$one_stage_results$object_type <- object_type
    fit$one_stage_results$morphology_feature <- morphology_feature
    fit$one_stage_results$cell_line <- cell_line
    all_one_stage_results[[length(all_one_stage_results) + 1L]] <- fit$one_stage_results
  }
  if (!is.null(fit$extreme_octile_results) && nrow(fit$extreme_octile_results)) {
    all_extreme_octile_results[[length(all_extreme_octile_results) + 1L]] <-
      fit$extreme_octile_results
  }
  if (!is.null(fit$extreme_octile_membership) && nrow(fit$extreme_octile_membership)) {
    all_extreme_octile_membership[[length(all_extreme_octile_membership) + 1L]] <-
      fit$extreme_octile_membership
  }
  varreg_print_extreme_octile_results(
    results = fit$extreme_octile_results,
    alpha_padj = alpha_padj,
    top_n = console_top_n
  )
  varreg_print_validation_results(
    bootstrap_summary = if (is.null(fit$bootstrap_summary)) data.frame() else fit$bootstrap_summary,
    one_stage_results = if (is.null(fit$one_stage_results)) data.frame() else fit$one_stage_results,
    top_n = console_top_n
  )
}

results <- nbglm_bind_rows(all_results)
if (nrow(results)) {
  results <- results[order(results$cell_line, results$pvals_adj, -abs(results$beta_variability)), , drop = FALSE]
}
summaries <- nbglm_bind_rows(all_summaries)
skips <- if (length(all_skips)) nbglm_bind_rows(all_skips) else nbglm_empty_skip_df()
global_results <- nbglm_bind_rows(all_global_results)
global_scores <- nbglm_bind_rows(all_global_scores)
global_bins <- nbglm_bind_rows(all_global_bins)
permutation_results <- nbglm_bind_rows(all_permutation_results)
bootstrap_replicates <- nbglm_bind_rows(all_bootstrap_replicates)
bootstrap_summaries <- nbglm_bind_rows(all_bootstrap_summaries)
one_stage_results <- nbglm_bind_rows(all_one_stage_results)
extreme_octile_results <- nbglm_bind_rows(all_extreme_octile_results)
extreme_octile_membership <- nbglm_bind_rows(all_extreme_octile_membership)

nbglm_write_csv(results, file.path(condition_out_dir, "variability_regression_results_all_cell_lines.csv"))
nbglm_write_csv(summaries, file.path(condition_out_dir, "model_summary_by_cell_line.csv"))
nbglm_write_csv(skips, file.path(condition_out_dir, "skipped_models.csv"))
if (nrow(global_results)) {
  nbglm_write_csv(
    global_results,
    file.path(condition_out_dir, "global_stochasticity_regression_results.csv")
  )
  nbglm_write_csv(
    global_scores,
    file.path(condition_out_dir, "per_object_stochasticity_scores.csv")
  )
  nbglm_write_csv(
    global_bins,
    file.path(condition_out_dir, "global_stochasticity_by_morphology_quintile.csv")
  )
}
if (nrow(permutation_results)) {
  nbglm_write_csv(
    permutation_results,
    file.path(condition_out_dir, "permutation_null_log_variance_ratios.csv")
  )
}
if (nrow(bootstrap_replicates)) {
  nbglm_write_csv(
    bootstrap_replicates,
    file.path(condition_out_dir, "refitted_independent_case_bootstrap_replicates.csv")
  )
}
if (nrow(bootstrap_summaries)) {
  nbglm_write_csv(
    bootstrap_summaries,
    file.path(condition_out_dir, "refitted_independent_case_bootstrap_summary.csv")
  )
}
if (nrow(one_stage_results)) {
  nbglm_write_csv(
    one_stage_results,
    file.path(condition_out_dir, "one_stage_nb_mean_dispersion_validation.csv")
  )
}
if (nrow(extreme_octile_results)) {
  nbglm_write_csv(
    extreme_octile_results,
    file.path(condition_out_dir, "extreme_octile_Q8_vs_Q1_variability_results.csv")
  )
  extreme_significant <- extreme_octile_results[
    extreme_octile_results$significant %in% TRUE, , drop = FALSE
  ]
  nbglm_write_csv(
    extreme_significant,
    file.path(condition_out_dir, "extreme_octile_Q8_vs_Q1_significant_genes.csv")
  )
  nbglm_write_csv(
    extreme_significant[extreme_significant$variance_ratio_high_vs_low > 1, , drop = FALSE],
    file.path(condition_out_dir, "extreme_octile_Q8_vs_Q1_increased_variability_genes.csv")
  )
  nbglm_write_csv(
    extreme_significant[extreme_significant$variance_ratio_high_vs_low < 1, , drop = FALSE],
    file.path(condition_out_dir, "extreme_octile_Q8_vs_Q1_decreased_variability_genes.csv")
  )
}
if (nrow(extreme_octile_membership)) {
  nbglm_write_csv(
    extreme_octile_membership,
    file.path(condition_out_dir, "extreme_octile_object_membership.csv")
  )
}
if (nrow(results)) {
  significant_results <- results[results$significant %in% TRUE, , drop = FALSE]
  increased_results <- significant_results[
    significant_results$beta_variability > 0, , drop = FALSE
  ]
  decreased_results <- significant_results[
    significant_results$beta_variability < 0, , drop = FALSE
  ]
  variability_only_results <- significant_results[
    !is.na(significant_results$mean_model_pvals_adj) &
      significant_results$mean_model_pvals_adj >= alpha_padj,
    ,
    drop = FALSE
  ]
  abs_deviance_sensitivity_results <- results[
    results$abs_deviance_significant_fdr %in% TRUE, , drop = FALSE
  ]
  dunn_smyth_sensitivity_results <- results[
    results$dunn_smyth_significant_fdr %in% TRUE, , drop = FALSE
  ]
  nbglm_write_csv(
    significant_results,
    file.path(condition_out_dir, "significant_variability_genes.csv")
  )
  nbglm_write_csv(
    increased_results,
    file.path(condition_out_dir, "significant_increased_variability_genes.csv")
  )
  nbglm_write_csv(
    decreased_results,
    file.path(condition_out_dir, "significant_decreased_variability_genes.csv")
  )
  nbglm_write_csv(
    variability_only_results,
    file.path(
      condition_out_dir,
      "significant_variability_without_significant_mean_change.csv"
    )
  )
  nbglm_write_csv(
    abs_deviance_sensitivity_results,
    file.path(condition_out_dir, "significant_abs_deviance_sensitivity_genes.csv")
  )
  nbglm_write_csv(
    dunn_smyth_sensitivity_results,
    file.path(condition_out_dir, "significant_dunn_smyth_sensitivity_genes.csv")
  )
}
predictor_transform <- nbglm_predictor_info_row(
  object_type, "gene_variability", morphology_feature,
  mode_info = mode_info, global_info = global_info
)
predictor_transform$within_line_standardization <- standardize_predictor_within_line
predictor_transform$final_effect_scale <- if (standardize_predictor_within_line) {
  "per one within-cell-line SD of the prepared predictor"
} else {
  "per one unit of the prepared predictor"
}
nbglm_write_csv(
  predictor_transform,
  file.path(condition_out_dir, "predictor_transform.csv")
)

message("Saved variability pilot outputs: ", condition_out_dir)
message("Main results: variability_regression_results_all_cell_lines.csv")
message("All significant hits: significant_variability_genes.csv")
message("Increased variability: significant_increased_variability_genes.csv")
message("Decreased variability: significant_decreased_variability_genes.csv")
message("Absolute-deviance sensitivity hits: significant_abs_deviance_sensitivity_genes.csv")
message("Dunn-Smyth sensitivity hits: significant_dunn_smyth_sensitivity_genes.csv")
message(
  "Variability hits without significant mean change: ",
  "significant_variability_without_significant_mean_change.csv"
)
message("Global stochasticity tests: global_stochasticity_regression_results.csv")
message("Per-object stochasticity scores: per_object_stochasticity_scores.csv")
message(
  "Global scores by morphology quintile: ",
  "global_stochasticity_by_morphology_quintile.csv"
)
message("Permutation null effects: permutation_null_log_variance_ratios.csv")
message("Refitted bootstrap summary: refitted_independent_case_bootstrap_summary.csv")
message("One-stage validation: one_stage_nb_mean_dispersion_validation.csv")
message("Extreme-octile results: extreme_octile_Q8_vs_Q1_variability_results.csv")
message("Extreme-octile significant hits: extreme_octile_Q8_vs_Q1_significant_genes.csv")
message("Extreme-octile membership: extreme_octile_object_membership.csv")
message(sprintf(
  "CosMx NB variability pilot finished in %.1f minutes",
  as.numeric(difftime(Sys.time(), t0, units = "mins"))
))
