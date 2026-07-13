nbglm_parse_list_arg <- function(x) {
  if (is.null(x) || length(x) == 0) return(character(0))
  values <- unlist(strsplit(as.character(x), ",", fixed = TRUE), use.names = FALSE)
  values <- trimws(values)
  values[nzchar(values)]
}

nbglm_parse_nullable_path <- function(x) {
  if (is.null(x) || length(x) == 0) return(NULL)
  x <- as.character(x[[1]])
  if (!nzchar(x) || tolower(trimws(x)) %in% c("null", "none", "na")) return(NULL)
  x
}

nbglm_first_nonempty <- function(...) {
  values <- list(...)
  for (value in values) {
    parsed <- nbglm_parse_list_arg(value)
    if (length(parsed) > 0) return(parsed)
  }
  character(0)
}

nbglm_safe_slug <- function(x) {
  x <- gsub("[^A-Za-z0-9._-]+", "_", as.character(x))
  x <- gsub("^_+|_+$", "", x)
  ifelse(nzchar(x), x, "unnamed")
}

nbglm_canonical_object_type <- function(x) {
  x <- tolower(trimws(as.character(x)))
  out <- ifelse(x %in% c("myotube", "myotubes"), "myotubes", x)
  out <- ifelse(out %in% c("myonucleus", "myonuclei", "nuclei"), "myonuclei", out)
  bad <- setdiff(out, c("myonuclei", "myotubes"))
  if (length(bad) > 0) {
    stop("Unsupported object_types: ", paste(unique(bad), collapse = ", "))
  }
  unique(out)
}

nbglm_canonical_target_type <- function(x) {
  x <- tolower(trimws(as.character(x)))
  out <- ifelse(x %in% c("gene", "genes"), "genes", x)
  out <- ifelse(out %in% c("pathway", "pathways", "decoupler_pathway",
                           "decoupler_pathways", "decoupler_mlm",
                           "decoupler_pathways_mlm"),
                "decoupler_pathways", out)
  out <- ifelse(out %in% c("tf", "tfs", "decoupler_tf", "decoupler_tfs",
                           "decoupler_ulm", "decoupler_tfs_ulm"),
                "decoupler_tfs", out)
  bad <- setdiff(out, c("genes", "decoupler_pathways", "decoupler_tfs"))
  if (length(bad) > 0) {
    stop("Unsupported target_types: ", paste(unique(bad), collapse = ", "))
  }
  unique(out)
}

nbglm_object_label <- function(object_type) {
  if (identical(object_type, "myonuclei")) return("myonuclei")
  "myotubes"
}

nbglm_decoupler_stem <- function(object_type) {
  if (identical(object_type, "myonuclei")) return("myonuclei")
  "myotubes"
}

nbglm_decoupler_spec <- function(object_type, target_type) {
  stem <- nbglm_decoupler_stem(object_type)
  if (identical(target_type, "decoupler_pathways")) {
    return(list(
      filename = paste0(stem, "_mlm_zscore.csv"),
      availability_col = "pathwaysmlm_zscore_available",
      reduction_prefixes = c("pathwaysmlm_pca_", "pathwaysmlm_umap_")
    ))
  }
  if (identical(target_type, "decoupler_tfs")) {
    return(list(
      filename = paste0(stem, "_ulm_zscore.csv"),
      availability_col = "tfsulm_zscore_available",
      reduction_prefixes = c("tfsulm_pca_", "tfsulm_umap_")
    ))
  }
  stop("No decoupleR spec for target_type=", target_type)
}

nbglm_empty_skip_df <- function() {
  data.frame(
    object_type = character(),
    target_type = character(),
    morphology_feature = character(),
    cell_line = character(),
    names = character(),
    reason = character(),
    details = character(),
    stringsAsFactors = FALSE
  )
}

nbglm_skip_row <- function(object_type, target_type, morphology_feature,
                           cell_line = NA_character_, names = NA_character_,
                           reason, details = "") {
  data.frame(
    object_type = object_type,
    target_type = target_type,
    morphology_feature = morphology_feature,
    cell_line = ifelse(is.na(cell_line), NA_character_, as.character(cell_line)),
    names = ifelse(is.na(names), NA_character_, as.character(names)),
    reason = reason,
    details = details,
    stringsAsFactors = FALSE
  )
}

nbglm_bind_rows <- function(rows) {
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(data.frame(stringsAsFactors = FALSE))
  out <- do.call(rbind, lapply(rows, as.data.frame))
  rownames(out) <- NULL
  out
}

nbglm_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(as.data.frame(x), file = path, row.names = FALSE, na = "")
}

nbglm_get_counts <- function(sce, label) {
  assay_names <- SummarizedExperiment::assayNames(sce)
  if (!("counts" %in% assay_names)) {
    stop("[", label, "] Missing 'counts' assay. Available assays: ",
         paste(assay_names, collapse = ", "))
  }
  cnt <- SummarizedExperiment::assay(sce, "counts")
  if (inherits(cnt, "dgCMatrix")) return(cnt)
  if (inherits(cnt, "Matrix")) return(methods::as(cnt, "dgCMatrix"))
  methods::as(Matrix::Matrix(cnt, sparse = TRUE), "dgCMatrix")
}

nbglm_make_size_factors <- function(counts) {
  total_counts <- Matrix::colSums(counts)
  nonzero_totals <- total_counts[total_counts > 0]
  if (length(nonzero_totals) == 0) {
    return(list(size_factors = NULL, total_counts = as.numeric(total_counts)))
  }

  median_total <- stats::median(nonzero_totals)
  if (!is.finite(median_total) || median_total <= 0) {
    return(list(size_factors = NULL, total_counts = as.numeric(total_counts)))
  }

  size_factors <- as.numeric(total_counts / median_total)
  sf_pos <- size_factors[is.finite(size_factors) & size_factors > 0]
  if (length(sf_pos) == 0) {
    return(list(size_factors = NULL, total_counts = as.numeric(total_counts)))
  }
  size_factors[!is.finite(size_factors) | size_factors <= 0] <- min(sf_pos) * 1e-3

  list(size_factors = size_factors, total_counts = as.numeric(total_counts))
}

nbglm_prepare_predictor <- function(values, mode = "zscore") {
  raw <- suppressWarnings(as.numeric(as.character(values)))
  mode <- tolower(trimws(as.character(mode)))
  finite <- is.finite(raw)
  if (sum(finite) < 2) stop("Fewer than 2 finite predictor values.")

  if (mode %in% c("zscore", "z", "standardized", "standardised")) {
    center <- mean(raw[finite])
    scale <- stats::sd(raw[finite])
    if (!is.finite(scale) || scale <= 0) stop("Predictor has no variation.")
    x <- (raw - center) / scale
    return(list(
      x = x,
      mode = "zscore",
      raw_mean = center,
      raw_sd = scale,
      raw_min = min(raw[finite]),
      raw_max = max(raw[finite])
    ))
  }

  if (mode %in% c("raw", "identity")) {
    scale <- stats::sd(raw[finite])
    if (!is.finite(scale) || scale <= 0) stop("Predictor has no variation.")
    return(list(
      x = raw,
      mode = "raw",
      raw_mean = mean(raw[finite]),
      raw_sd = scale,
      raw_min = min(raw[finite]),
      raw_max = max(raw[finite])
    ))
  }

  if (mode %in% c("rank", "global_rank")) {
    ranks <- rep(NA_real_, length(raw))
    r <- rank(raw[finite], ties.method = "average")
    denom <- max(r) - min(r)
    if (!is.finite(denom) || denom <= 0) stop("Rank predictor has no variation.")
    ranks[finite] <- (r - min(r)) / denom
    return(list(
      x = ranks,
      mode = "rank",
      raw_mean = mean(raw[finite]),
      raw_sd = stats::sd(raw[finite]),
      raw_min = min(raw[finite]),
      raw_max = max(raw[finite])
    ))
  }

  stop("Unsupported predictor_mode: ", mode)
}

nbglm_resolve_predictor_mode <- function(morphology_feature, predictor_mode = "auto",
                                         global_rank_feature_patterns = c("nuclear_aberration")) {
  requested <- tolower(trimws(as.character(predictor_mode)[1]))
  if (!nzchar(requested)) requested <- "auto"

  patterns <- nbglm_parse_list_arg(global_rank_feature_patterns)
  matches_global_rank <- length(patterns) > 0 && any(vapply(
    patterns,
    function(pattern) grepl(pattern, morphology_feature, ignore.case = TRUE),
    logical(1)
  ))

  if (requested %in% c("auto", "default", "feature_auto")) {
    if (matches_global_rank) {
      return(list(
        requested = requested,
        resolved = "global_rank",
        scope = "global_across_included_object_type",
        detail = paste0(
          "auto: feature matched global-rank pattern(s): ",
          paste(patterns[vapply(patterns, function(pattern) {
            grepl(pattern, morphology_feature, ignore.case = TRUE)
          }, logical(1))], collapse = ", ")
        )
      ))
    }
    return(list(
      requested = requested,
      resolved = "zscore",
      scope = "within_cell_line_model",
      detail = "auto: feature did not match any global-rank pattern"
    ))
  }

  if (requested %in% c("rank", "global_rank", "global-rank")) {
    return(list(
      requested = requested,
      resolved = "global_rank",
      scope = "global_across_included_object_type",
      detail = "forced by predictor_mode"
    ))
  }

  if (requested %in% c("zscore", "z", "standardized", "standardised")) {
    return(list(
      requested = requested,
      resolved = "zscore",
      scope = "within_cell_line_model",
      detail = "forced by predictor_mode"
    ))
  }

  if (requested %in% c("raw", "identity")) {
    return(list(
      requested = requested,
      resolved = "raw",
      scope = "within_cell_line_model",
      detail = "forced by predictor_mode"
    ))
  }

  stop("Unsupported predictor_mode: ", predictor_mode)
}

nbglm_make_global_rank_predictor <- function(values, global_mask) {
  raw <- suppressWarnings(as.numeric(as.character(values)))
  if (length(global_mask) != length(raw)) {
    stop("global_mask length does not match predictor length.")
  }

  finite <- is.finite(raw) & as.logical(global_mask)
  if (sum(finite) < 2) stop("Fewer than 2 finite values for global-rank predictor.")

  ranks <- rank(raw[finite], ties.method = "average")
  denom <- max(ranks) - min(ranks)
  if (!is.finite(denom) || denom <= 0) {
    stop("Global-rank predictor has no variation.")
  }

  x <- rep(NA_real_, length(raw))
  x[finite] <- (ranks - min(ranks)) / denom

  list(
    x = x,
    n_global = sum(finite),
    raw_mean_global = mean(raw[finite]),
    raw_sd_global = stats::sd(raw[finite]),
    raw_min_global = min(raw[finite]),
    raw_max_global = max(raw[finite])
  )
}

nbglm_prepare_model_predictor <- function(raw_values, mode_info,
                                          global_x = NULL,
                                          global_info = NULL) {
  raw <- suppressWarnings(as.numeric(as.character(raw_values)))

  if (identical(mode_info$resolved, "global_rank")) {
    if (is.null(global_x)) stop("global_x is required for global_rank predictor mode.")
    x <- suppressWarnings(as.numeric(global_x))
    finite <- is.finite(raw) & is.finite(x)
    if (sum(finite) < 2) stop("Fewer than 2 finite global-rank predictor values.")
    if (!is.finite(stats::sd(x[finite])) || stats::sd(x[finite]) <= 0) {
      stop("Global-rank predictor has no variation in this model subset.")
    }

    return(list(
      x = x,
      mode = "global_rank",
      predictor_mode_requested = mode_info$requested,
      predictor_mode_resolved = mode_info$resolved,
      predictor_transform_scope = mode_info$scope,
      predictor_transform_detail = mode_info$detail,
      raw_mean = mean(raw[finite]),
      raw_sd = stats::sd(raw[finite]),
      raw_min = min(raw[finite]),
      raw_max = max(raw[finite]),
      global_rank_n = if (!is.null(global_info$n_global)) global_info$n_global else NA_integer_,
      global_rank_raw_mean = if (!is.null(global_info$raw_mean_global)) global_info$raw_mean_global else NA_real_,
      global_rank_raw_sd = if (!is.null(global_info$raw_sd_global)) global_info$raw_sd_global else NA_real_,
      global_rank_raw_min = if (!is.null(global_info$raw_min_global)) global_info$raw_min_global else NA_real_,
      global_rank_raw_max = if (!is.null(global_info$raw_max_global)) global_info$raw_max_global else NA_real_
    ))
  }

  pred <- nbglm_prepare_predictor(raw, mode = mode_info$resolved)
  pred$predictor_mode_requested <- mode_info$requested
  pred$predictor_mode_resolved <- mode_info$resolved
  pred$predictor_transform_scope <- mode_info$scope
  pred$predictor_transform_detail <- mode_info$detail
  pred$global_rank_n <- NA_integer_
  pred$global_rank_raw_mean <- NA_real_
  pred$global_rank_raw_sd <- NA_real_
  pred$global_rank_raw_min <- NA_real_
  pred$global_rank_raw_max <- NA_real_
  pred
}

nbglm_predictor_info_row <- function(object_type, target_type, morphology_feature,
                                     mode_info, global_info = NULL) {
  data.frame(
    object_type = object_type,
    target_type = target_type,
    morphology_feature = morphology_feature,
    predictor_mode_requested = mode_info$requested,
    predictor_mode_resolved = mode_info$resolved,
    predictor_transform_scope = mode_info$scope,
    predictor_transform_detail = mode_info$detail,
    global_rank_n = if (!is.null(global_info$n_global)) global_info$n_global else NA_integer_,
    global_rank_raw_mean = if (!is.null(global_info$raw_mean_global)) global_info$raw_mean_global else NA_real_,
    global_rank_raw_sd = if (!is.null(global_info$raw_sd_global)) global_info$raw_sd_global else NA_real_,
    global_rank_raw_min = if (!is.null(global_info$raw_min_global)) global_info$raw_min_global else NA_real_,
    global_rank_raw_max = if (!is.null(global_info$raw_max_global)) global_info$raw_max_global else NA_real_,
    stringsAsFactors = FALSE
  )
}

nbglm_build_design <- function(col_data, predictor, slide_col = NA_character_,
                               slide_covariate = TRUE) {
  design_df <- data.frame(predictor = as.numeric(predictor))
  use_slide <- isTRUE(slide_covariate) && !is.na(slide_col) && slide_col %in% colnames(col_data)

  if (use_slide) {
    slide_chr <- as.character(col_data[[slide_col]])
    slide_chr[is.na(slide_chr) | trimws(slide_chr) == ""] <- "UNKNOWN"
    design_df$slide <- factor(slide_chr)
    if (nlevels(design_df$slide) > 1) {
      design <- stats::model.matrix(~ predictor + slide, data = design_df)
      return(list(design = design, predictor_coef = "predictor", design_cols = colnames(design),
                  used_slide = TRUE))
    }
  }

  design <- stats::model.matrix(~ predictor, data = design_df)
  list(design = design, predictor_coef = "predictor", design_cols = colnames(design),
       used_slide = FALSE)
}

nbglm_mutant_status <- function(cell_lines, control_lines, mutant_lines) {
  control_norm <- unique(stats::na.omit(normalize_label(control_lines)))
  mutant_norm <- unique(stats::na.omit(normalize_label(mutant_lines)))
  overlap <- intersect(control_norm, mutant_norm)
  if (length(overlap) > 0) {
    stop("Control and mutant line aliases overlap: ", paste(overlap, collapse = ", "))
  }

  line_norm <- normalize_label(cell_lines)
  status <- rep(NA_character_, length(line_norm))
  status[line_norm %in% control_norm] <- "control"
  status[line_norm %in% mutant_norm] <- "mutant"
  status
}

nbglm_build_mutant_status_interaction_design <- function(
  col_data,
  predictor,
  mutant_status,
  slide_col = NA_character_,
  slide_covariate = TRUE
) {
  design_df <- data.frame(
    predictor = as.numeric(predictor),
    mutant_status = factor(mutant_status, levels = c("control", "mutant"))
  )
  use_slide <- isTRUE(slide_covariate) && !is.na(slide_col) && slide_col %in% colnames(col_data)

  if (use_slide) {
    slide_chr <- as.character(col_data[[slide_col]])
    slide_chr[is.na(slide_chr) | trimws(slide_chr) == ""] <- "UNKNOWN"
    design_df$slide <- factor(slide_chr)
    if (nlevels(design_df$slide) > 1) {
      design <- stats::model.matrix(~ predictor * mutant_status + slide, data = design_df)
    } else {
      use_slide <- FALSE
      design <- stats::model.matrix(~ predictor * mutant_status, data = design_df)
    }
  } else {
    design <- stats::model.matrix(~ predictor * mutant_status, data = design_df)
  }

  interaction_candidates <- c(
    "predictor:mutant_statusmutant",
    "mutant_statusmutant:predictor"
  )
  interaction_coef <- interaction_candidates[interaction_candidates %in% colnames(design)][1]
  if (is.na(interaction_coef)) {
    stop("Could not find predictor:mutant_status interaction coefficient in design: ",
         paste(colnames(design), collapse = ", "))
  }

  list(
    design = design,
    predictor_coef = "predictor",
    mutant_status_coef = "mutant_statusmutant",
    interaction_coef = interaction_coef,
    design_cols = colnames(design),
    used_slide = use_slide
  )
}

nbglm_expression_summary <- function(counts, size_factors) {
  norm <- counts %*% Matrix::Diagonal(x = 1 / size_factors)
  rownames(norm) <- rownames(counts)
  colnames(norm) <- colnames(counts)

  log_norm <- norm
  log_norm@x <- log2(log_norm@x + 1)

  data.frame(
    names = rownames(counts),
    mean_size_factor_norm_expr = as.numeric(Matrix::rowMeans(norm)),
    mean_log2_size_factor_norm_expr_plus1 = as.numeric(Matrix::rowMeans(log_norm)),
    pct_positive = as.numeric(Matrix::rowMeans(counts > 0) * 100),
    stringsAsFactors = FALSE
  )
}

nbglm_estimate_disp <- function(y, design, object_type) {
  if (identical(object_type, "myonuclei")) {
    return(edgeR::estimateDisp(
      y,
      design,
      trend.method = "none",
      grid.length = 11,
      grid.range = c(-6, 6)
    ))
  }
  edgeR::estimateDisp(y, design)
}

nbglm_resolve_metadata_cols <- function(sce) {
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  cell_line_col <- pick_col(
    cd,
    c("Cell Line", "Cell.Line", "cell_line", "CellLine", "cell_line_key"),
    "Cell Line"
  )
  slide_col <- pick_col(
    cd,
    c("Slide Name", "Slide.Name", "slide_name", "slide"),
    "Slide Name",
    required = FALSE
  )
  list(cell_line_col = cell_line_col, slide_col = slide_col)
}

nbglm_cell_lines <- function(sce, cell_line_col, cell_lines_to_include = character(0)) {
  all_lines <- sort(unique(stats::na.omit(as.character(SummarizedExperiment::colData(sce)[[cell_line_col]]))))
  if (!length(cell_lines_to_include)) return(all_lines)
  keep_norm <- normalize_label(cell_lines_to_include)
  all_lines[normalize_label(all_lines) %in% keep_norm]
}

nbglm_run_gene_condition <- function(
  sce,
  object_type,
  morphology_feature,
  cell_lines_to_include = character(0),
  predictor_mode = "zscore",
  slide_covariate = TRUE,
  min_obs_per_cell_line = 100L,
  min_expr_obs = 20L,
  alpha_padj = 0.05,
  effect_thr = 0.15,
  global_rank_feature_patterns = c("nuclear_aberration")
) {
  target_type <- "genes"
  cd_all <- as.data.frame(SummarizedExperiment::colData(sce))
  if (!(morphology_feature %in% colnames(cd_all))) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        reason = "missing_morphology_feature",
        details = paste("Available columns:", paste(colnames(cd_all), collapse = ", "))
      )
    ))
  }

  cols <- nbglm_resolve_metadata_cols(sce)
  cell_lines <- nbglm_cell_lines(sce, cols$cell_line_col, cell_lines_to_include)
  if (!length(cell_lines)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(object_type, target_type, morphology_feature,
                               reason = "no_matching_cell_lines")
    ))
  }

  mode_info <- nbglm_resolve_predictor_mode(
    morphology_feature = morphology_feature,
    predictor_mode = predictor_mode,
    global_rank_feature_patterns = global_rank_feature_patterns
  )
  global_info <- NULL
  if (identical(mode_info$resolved, "global_rank")) {
    global_mask <- as.character(cd_all[[cols$cell_line_col]]) %in% cell_lines
    global_info <- tryCatch(
      nbglm_make_global_rank_predictor(cd_all[[morphology_feature]], global_mask = global_mask),
      error = function(e) e
    )
    if (inherits(global_info, "error")) {
      return(list(
        results = data.frame(stringsAsFactors = FALSE),
        summary = data.frame(stringsAsFactors = FALSE),
        skipped = nbglm_skip_row(
          object_type, target_type, morphology_feature,
          reason = "invalid_global_rank_predictor",
          details = conditionMessage(global_info)
        ),
        predictor_info = nbglm_predictor_info_row(object_type, target_type, morphology_feature, mode_info)
      ))
    }
  }
  predictor_info <- nbglm_predictor_info_row(
    object_type, target_type, morphology_feature, mode_info, global_info = global_info
  )

  all_results <- list()
  all_summary <- list()
  all_skips <- list()
  label <- nbglm_object_label(object_type)

  for (cl in cell_lines) {
    t0 <- Sys.time()
    message(sprintf("[%s/%s/%s] %s: starting genes", object_type, morphology_feature, target_type, cl))

    keep_cl <- as.character(SummarizedExperiment::colData(sce)[[cols$cell_line_col]]) == cl
    a0 <- sce[, keep_cl, drop = FALSE]
    cd <- as.data.frame(SummarizedExperiment::colData(a0))
    predictor_raw <- suppressWarnings(as.numeric(as.character(cd[[morphology_feature]])))
    global_x0 <- if (identical(mode_info$resolved, "global_rank")) global_info$x[keep_cl] else NULL
    keep_pred <- is.finite(predictor_raw)
    if (identical(mode_info$resolved, "global_rank")) {
      keep_pred <- keep_pred & is.finite(global_x0)
    }
    a0 <- a0[, keep_pred, drop = FALSE]
    if (identical(mode_info$resolved, "global_rank")) {
      global_x0 <- global_x0[keep_pred]
    }
    cd <- as.data.frame(SummarizedExperiment::colData(a0))

    if (ncol(a0) < min_obs_per_cell_line) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "too_few_observations",
        details = paste0("n_obs=", ncol(a0), "; min_obs_per_cell_line=", min_obs_per_cell_line)
      )
      next
    }

    counts_all <- nbglm_get_counts(a0, label = paste(object_type, cl))
    sf <- nbglm_make_size_factors(counts_all)
    if (is.null(sf$size_factors)) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "invalid_size_factors"
      )
      next
    }

    finite_sf <- is.finite(sf$size_factors) & sf$size_factors > 0
    a0 <- a0[, finite_sf, drop = FALSE]
    counts_all <- counts_all[, finite_sf, drop = FALSE]
    size_factors <- sf$size_factors[finite_sf]
    if (identical(mode_info$resolved, "global_rank")) {
      global_x0 <- global_x0[finite_sf]
    }
    cd <- as.data.frame(SummarizedExperiment::colData(a0))

    if (ncol(a0) < min_obs_per_cell_line) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "too_few_observations_after_size_factor_filter",
        details = paste0("n_obs=", ncol(a0), "; min_obs_per_cell_line=", min_obs_per_cell_line)
      )
      next
    }

    pred <- tryCatch(
      nbglm_prepare_model_predictor(
        cd[[morphology_feature]],
        mode_info = mode_info,
        global_x = global_x0,
        global_info = global_info
      ),
      error = function(e) e
    )
    if (inherits(pred, "error")) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "invalid_predictor",
        details = conditionMessage(pred)
      )
      next
    }

    keep_genes <- Matrix::rowSums(counts_all > 0) >= as.integer(min_expr_obs)
    if (!any(keep_genes)) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "no_features_pass_min_expr_obs",
        details = paste0("min_expr_obs=", min_expr_obs)
      )
      next
    }

    counts <- counts_all[keep_genes, , drop = FALSE]
    design_info <- nbglm_build_design(
      col_data = cd,
      predictor = pred$x,
      slide_col = cols$slide_col,
      slide_covariate = slide_covariate
    )
    coef_idx <- which(colnames(design_info$design) == design_info$predictor_coef)
    if (length(coef_idx) != 1) {
      stop("Could not find predictor coefficient in design: ",
           paste(colnames(design_info$design), collapse = ", "))
    }

    y <- edgeR::DGEList(counts = counts)
    y$samples$lib.size <- as.numeric(size_factors)
    y$samples$norm.factors <- rep(1, ncol(counts))

	    fit_out <- tryCatch({
	      y <- nbglm_estimate_disp(y, design_info$design, object_type = object_type)
	      fit <- edgeR::glmFit(y, design_info$design)
	      lrt <- edgeR::glmLRT(fit, coef = coef_idx)
	      qlf <- tryCatch({
	        ql_fit <- edgeR::glmQLFit(y, design_info$design, robust = TRUE)
	        edgeR::glmQLFTest(ql_fit, coef = coef_idx)
	      }, error = function(e) e)
	      list(fit = fit, lrt = lrt, qlf = qlf)
	    }, error = function(e) e)

    if (inherits(fit_out, "error")) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "model_fit_failed",
        details = conditionMessage(fit_out)
      )
      next
    }

	    beta <- fit_out$fit$coefficients[, coef_idx]
	    lr <- fit_out$lrt$table$LR
	    pvals <- fit_out$lrt$table$PValue
	    pvals_adj <- stats::p.adjust(pvals, method = "BH")
	    qlf_stat <- rep(NA_real_, length(beta))
	    qlf_pvals <- rep(NA_real_, length(beta))
	    qlf_pvals_adj <- rep(NA_real_, length(beta))
	    if (inherits(fit_out$qlf, "error")) {
	      warning(
	        "QL fit failed for ",
	        paste(object_type, target_type, morphology_feature, cl, sep = "/"),
	        ": ",
	        conditionMessage(fit_out$qlf)
	      )
	    } else {
	      qlf_stat <- fit_out$qlf$table$F
	      qlf_pvals <- fit_out$qlf$table$PValue
	      qlf_pvals_adj <- stats::p.adjust(qlf_pvals, method = "BH")
	    }
	    log2_rate_ratio <- beta / log(2)
	    signed_lrt_z <- sign(beta) * sqrt(pmax(lr, 0))
	    signed_qlf_stat <- sign(beta) * sqrt(pmax(qlf_stat, 0))
	    expr_summary <- nbglm_expression_summary(counts, size_factors)

    res <- data.frame(
      object_type = object_type,
      target_type = target_type,
      morphology_feature = morphology_feature,
      cell_line = cl,
      names = rownames(counts),
      beta0 = fit_out$fit$coefficients[, 1],
      beta_predictor = beta,
      log2_rate_ratio_per_predictor = log2_rate_ratio,
	      lr_stat = lr,
	      signed_lrt_z = signed_lrt_z,
	      pvals = pvals,
	      pvals_adj = pvals_adj,
	      qlf_stat = qlf_stat,
	      signed_qlf_stat = signed_qlf_stat,
	      qlf_pvals = qlf_pvals,
	      qlf_pvals_adj = qlf_pvals_adj,
	      significant = (pvals_adj < alpha_padj) & (abs(log2_rate_ratio) >= effect_thr),
      n_obs = ncol(counts),
      n_expr_obs = as.numeric(Matrix::rowSums(counts > 0)),
      predictor_mode = pred$mode,
      predictor_mode_requested = pred$predictor_mode_requested,
      predictor_mode_resolved = pred$predictor_mode_resolved,
      predictor_transform_scope = pred$predictor_transform_scope,
      predictor_transform_detail = pred$predictor_transform_detail,
      predictor_raw_mean = pred$raw_mean,
      predictor_raw_sd = pred$raw_sd,
      predictor_raw_min = pred$raw_min,
      predictor_raw_max = pred$raw_max,
      predictor_global_rank_n = pred$global_rank_n,
      predictor_global_rank_raw_mean = pred$global_rank_raw_mean,
      predictor_global_rank_raw_sd = pred$global_rank_raw_sd,
      predictor_global_rank_raw_min = pred$global_rank_raw_min,
      predictor_global_rank_raw_max = pred$global_rank_raw_max,
      used_slide_covariate = design_info$used_slide,
      design_cols = paste(design_info$design_cols, collapse = ";"),
      stringsAsFactors = FALSE
    )
    res <- merge(res, expr_summary, by = "names", sort = FALSE)
    res <- res[match(rownames(counts), res$names), , drop = FALSE]

    all_results[[length(all_results) + 1L]] <- res
    all_summary[[length(all_summary) + 1L]] <- data.frame(
      object_type = object_type,
      target_type = target_type,
      morphology_feature = morphology_feature,
      cell_line = cl,
      n_obs = ncol(counts),
      n_features_tested = nrow(counts),
      n_significant = sum(res$significant, na.rm = TRUE),
      predictor_mode = pred$mode,
      predictor_mode_requested = pred$predictor_mode_requested,
      predictor_mode_resolved = pred$predictor_mode_resolved,
      predictor_transform_scope = pred$predictor_transform_scope,
      predictor_transform_detail = pred$predictor_transform_detail,
      predictor_raw_mean = pred$raw_mean,
      predictor_raw_sd = pred$raw_sd,
      predictor_raw_min = pred$raw_min,
      predictor_raw_max = pred$raw_max,
      predictor_global_rank_n = pred$global_rank_n,
      used_slide_covariate = design_info$used_slide,
      elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      stringsAsFactors = FALSE
    )

    message(sprintf("[%s/%s/%s] %s: fitted %d %s; significant=%d",
                    object_type, morphology_feature, target_type, cl,
                    nrow(counts), target_type, sum(res$significant, na.rm = TRUE)))
  }

  list(
    results = nbglm_bind_rows(all_results),
    summary = nbglm_bind_rows(all_summary),
    skipped = if (length(all_skips)) nbglm_bind_rows(all_skips) else nbglm_empty_skip_df(),
    predictor_info = predictor_info
  )
}

nbglm_run_gene_mutant_status_interaction <- function(
  sce,
  object_type,
  morphology_feature,
  cell_lines_to_include = character(0),
  mutant_status_control_lines = character(0),
  mutant_status_mutant_lines = character(0),
  predictor_mode = "zscore",
  slide_covariate = TRUE,
  min_obs_per_cell_line = 100L,
  min_expr_obs = 20L,
  alpha_padj = 0.05,
  effect_thr = 0.15,
  global_rank_feature_patterns = c("nuclear_aberration")
) {
  target_type <- "genes"
  regression_type <- "mutant_status_interaction"
  pooled_label <- "pooled_control_vs_mutant"
  cd_all <- as.data.frame(SummarizedExperiment::colData(sce))
  if (!(morphology_feature %in% colnames(cd_all))) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "missing_morphology_feature",
        details = paste("Available columns:", paste(colnames(cd_all), collapse = ", "))
      )
    ))
  }

  if (!length(mutant_status_control_lines) || !length(mutant_status_mutant_lines)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "missing_mutant_status_line_sets",
        details = "Provide both mutant_status_control_lines and mutant_status_mutant_lines."
      )
    ))
  }

  cols <- nbglm_resolve_metadata_cols(sce)
  cell_lines <- nbglm_cell_lines(sce, cols$cell_line_col, cell_lines_to_include)
  if (!length(cell_lines)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(object_type, target_type, morphology_feature,
                               cell_line = pooled_label,
                               reason = "no_matching_cell_lines")
    ))
  }

  status_all <- tryCatch(
    nbglm_mutant_status(
      cd_all[[cols$cell_line_col]],
      control_lines = mutant_status_control_lines,
      mutant_lines = mutant_status_mutant_lines
    ),
    error = function(e) e
  )
  if (inherits(status_all, "error")) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "invalid_mutant_status_line_sets",
        details = conditionMessage(status_all)
      )
    ))
  }

  selected_mask <- as.character(cd_all[[cols$cell_line_col]]) %in% cell_lines
  classified_mask <- !is.na(status_all)
  keep_status <- selected_mask & classified_mask
  if (!any(keep_status)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "no_matching_mutant_status_lines",
        details = paste0(
          "Observed cell lines: ",
          paste(sort(unique(as.character(cd_all[[cols$cell_line_col]]))), collapse = ", ")
        )
      )
    ))
  }

  mode_info <- nbglm_resolve_predictor_mode(
    morphology_feature = morphology_feature,
    predictor_mode = predictor_mode,
    global_rank_feature_patterns = global_rank_feature_patterns
  )
  interaction_mode_info <- mode_info
  if (!identical(interaction_mode_info$resolved, "global_rank")) {
    interaction_mode_info$scope <- "pooled_control_mutant_interaction_model"
    interaction_mode_info$detail <- paste0(
      interaction_mode_info$detail,
      "; fitted in pooled control/mutant interaction model"
    )
  }

  global_info <- NULL
  if (identical(interaction_mode_info$resolved, "global_rank")) {
    global_info <- tryCatch(
      nbglm_make_global_rank_predictor(cd_all[[morphology_feature]], global_mask = keep_status),
      error = function(e) e
    )
    if (inherits(global_info, "error")) {
      return(list(
        results = data.frame(stringsAsFactors = FALSE),
        summary = data.frame(stringsAsFactors = FALSE),
        skipped = nbglm_skip_row(
          object_type, target_type, morphology_feature,
          cell_line = pooled_label,
          reason = "invalid_global_rank_predictor",
          details = conditionMessage(global_info)
        ),
        predictor_info = nbglm_predictor_info_row(
          object_type, target_type, morphology_feature, interaction_mode_info
        )
      ))
    }
  }
  predictor_info <- nbglm_predictor_info_row(
    object_type, target_type, morphology_feature, interaction_mode_info, global_info = global_info
  )

  t0 <- Sys.time()
  message(sprintf("[%s/%s/%s] starting mutant-status interaction genes",
                  object_type, morphology_feature, target_type))

  a0 <- sce[, keep_status, drop = FALSE]
  cd <- as.data.frame(SummarizedExperiment::colData(a0))
  status0 <- status_all[keep_status]
  global_x0 <- if (identical(interaction_mode_info$resolved, "global_rank")) {
    global_info$x[keep_status]
  } else {
    NULL
  }

  predictor_raw <- suppressWarnings(as.numeric(as.character(cd[[morphology_feature]])))
  keep_pred <- is.finite(predictor_raw)
  if (identical(interaction_mode_info$resolved, "global_rank")) {
    keep_pred <- keep_pred & is.finite(global_x0)
  }
  a0 <- a0[, keep_pred, drop = FALSE]
  cd <- as.data.frame(SummarizedExperiment::colData(a0))
  status0 <- status0[keep_pred]
  if (identical(interaction_mode_info$resolved, "global_rank")) {
    global_x0 <- global_x0[keep_pred]
  }

  status_counts <- table(factor(status0, levels = c("control", "mutant")))
  if (any(status_counts < as.integer(min_obs_per_cell_line))) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "too_few_observations_by_mutant_status",
        details = paste0(
          "n_control=", as.integer(status_counts[["control"]]),
          "; n_mutant=", as.integer(status_counts[["mutant"]]),
          "; min_obs_per_cell_line=", min_obs_per_cell_line
        )
      ),
      predictor_info = predictor_info
    ))
  }

  counts_all <- nbglm_get_counts(a0, label = paste(object_type, regression_type))
  sf <- nbglm_make_size_factors(counts_all)
  if (is.null(sf$size_factors)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "invalid_size_factors"
      ),
      predictor_info = predictor_info
    ))
  }

  finite_sf <- is.finite(sf$size_factors) & sf$size_factors > 0
  a0 <- a0[, finite_sf, drop = FALSE]
  counts_all <- counts_all[, finite_sf, drop = FALSE]
  size_factors <- sf$size_factors[finite_sf]
  status0 <- status0[finite_sf]
  if (identical(interaction_mode_info$resolved, "global_rank")) {
    global_x0 <- global_x0[finite_sf]
  }
  cd <- as.data.frame(SummarizedExperiment::colData(a0))

  status_counts <- table(factor(status0, levels = c("control", "mutant")))
  if (any(status_counts < as.integer(min_obs_per_cell_line))) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "too_few_observations_by_mutant_status_after_size_factor_filter",
        details = paste0(
          "n_control=", as.integer(status_counts[["control"]]),
          "; n_mutant=", as.integer(status_counts[["mutant"]]),
          "; min_obs_per_cell_line=", min_obs_per_cell_line
        )
      ),
      predictor_info = predictor_info
    ))
  }

  pred <- tryCatch(
    nbglm_prepare_model_predictor(
      cd[[morphology_feature]],
      mode_info = interaction_mode_info,
      global_x = global_x0,
      global_info = global_info
    ),
    error = function(e) e
  )
  if (inherits(pred, "error")) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "invalid_predictor",
        details = conditionMessage(pred)
      ),
      predictor_info = predictor_info
    ))
  }

  keep_genes <- Matrix::rowSums(counts_all > 0) >= as.integer(min_expr_obs)
  if (!any(keep_genes)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "no_features_pass_min_expr_obs",
        details = paste0("min_expr_obs=", min_expr_obs)
      ),
      predictor_info = predictor_info
    ))
  }

  counts <- counts_all[keep_genes, , drop = FALSE]
  design_info <- tryCatch(
    nbglm_build_mutant_status_interaction_design(
      col_data = cd,
      predictor = pred$x,
      mutant_status = status0,
      slide_col = cols$slide_col,
      slide_covariate = slide_covariate
    ),
    error = function(e) e
  )
  if (inherits(design_info, "error")) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "invalid_design",
        details = conditionMessage(design_info)
      ),
      predictor_info = predictor_info
    ))
  }

  design_rank <- qr(design_info$design)$rank
  if (design_rank < ncol(design_info$design)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "rank_deficient_design",
        details = paste0(
          "rank=", design_rank,
          "; n_cols=", ncol(design_info$design),
          "; design_cols=", paste(design_info$design_cols, collapse = ";")
        )
      ),
      predictor_info = predictor_info
    ))
  }

  predictor_idx <- which(colnames(design_info$design) == design_info$predictor_coef)
  interaction_idx <- which(colnames(design_info$design) == design_info$interaction_coef)
  mutant_status_idx <- which(colnames(design_info$design) == design_info$mutant_status_coef)
  if (length(predictor_idx) != 1 || length(interaction_idx) != 1) {
    stop("Could not find predictor or interaction coefficient in design: ",
         paste(colnames(design_info$design), collapse = ", "))
  }
  mutant_slope_contrast <- rep(0, ncol(design_info$design))
  mutant_slope_contrast[predictor_idx] <- 1
  mutant_slope_contrast[interaction_idx] <- 1

  y <- edgeR::DGEList(counts = counts)
  y$samples$lib.size <- as.numeric(size_factors)
  y$samples$norm.factors <- rep(1, ncol(counts))

  fit_out <- tryCatch({
    y <- nbglm_estimate_disp(y, design_info$design, object_type = object_type)
    fit <- edgeR::glmFit(y, design_info$design)
    lrt_interaction <- edgeR::glmLRT(fit, coef = interaction_idx)
    lrt_mutant_slope <- edgeR::glmLRT(fit, contrast = mutant_slope_contrast)
    ql_fit <- tryCatch(
      edgeR::glmQLFit(y, design_info$design, robust = TRUE),
      error = function(e) e
    )
    qlf_interaction <- ql_fit
    qlf_mutant_slope <- ql_fit
    if (!inherits(ql_fit, "error")) {
      qlf_interaction <- edgeR::glmQLFTest(ql_fit, coef = interaction_idx)
      qlf_mutant_slope <- edgeR::glmQLFTest(ql_fit, contrast = mutant_slope_contrast)
    }
    list(
      fit = fit,
      lrt_interaction = lrt_interaction,
      lrt_mutant_slope = lrt_mutant_slope,
      qlf_interaction = qlf_interaction,
      qlf_mutant_slope = qlf_mutant_slope
    )
  }, error = function(e) e)

  if (inherits(fit_out, "error")) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        cell_line = pooled_label,
        reason = "model_fit_failed",
        details = conditionMessage(fit_out)
      ),
      predictor_info = predictor_info
    ))
  }

  beta_control <- fit_out$fit$coefficients[, predictor_idx]
  beta_interaction <- fit_out$fit$coefficients[, interaction_idx]
  beta_mutant_status <- if (length(mutant_status_idx) == 1) {
    fit_out$fit$coefficients[, mutant_status_idx]
  } else {
    rep(NA_real_, length(beta_control))
  }
  beta_mutant <- beta_control + beta_interaction

  lr_interaction <- fit_out$lrt_interaction$table$LR
  pvals_interaction <- fit_out$lrt_interaction$table$PValue
  pvals_adj_interaction <- stats::p.adjust(pvals_interaction, method = "BH")
  lr_mutant_slope <- fit_out$lrt_mutant_slope$table$LR
  pvals_mutant_slope <- fit_out$lrt_mutant_slope$table$PValue
  pvals_adj_mutant_slope <- stats::p.adjust(pvals_mutant_slope, method = "BH")

  qlf_stat_interaction <- rep(NA_real_, length(beta_control))
  qlf_pvals_interaction <- rep(NA_real_, length(beta_control))
  qlf_pvals_adj_interaction <- rep(NA_real_, length(beta_control))
  qlf_stat_mutant_slope <- rep(NA_real_, length(beta_control))
  qlf_pvals_mutant_slope <- rep(NA_real_, length(beta_control))
  qlf_pvals_adj_mutant_slope <- rep(NA_real_, length(beta_control))
  if (inherits(fit_out$qlf_interaction, "error")) {
    warning(
      "QL fit failed for ",
      paste(object_type, target_type, morphology_feature, regression_type, sep = "/"),
      ": ",
      conditionMessage(fit_out$qlf_interaction)
    )
  } else {
    qlf_stat_interaction <- fit_out$qlf_interaction$table$F
    qlf_pvals_interaction <- fit_out$qlf_interaction$table$PValue
    qlf_pvals_adj_interaction <- stats::p.adjust(qlf_pvals_interaction, method = "BH")
    qlf_stat_mutant_slope <- fit_out$qlf_mutant_slope$table$F
    qlf_pvals_mutant_slope <- fit_out$qlf_mutant_slope$table$PValue
    qlf_pvals_adj_mutant_slope <- stats::p.adjust(qlf_pvals_mutant_slope, method = "BH")
  }

  log2_control <- beta_control / log(2)
  log2_interaction <- beta_interaction / log(2)
  log2_mutant <- beta_mutant / log(2)
  signed_lrt_z_interaction <- sign(beta_interaction) * sqrt(pmax(lr_interaction, 0))
  signed_lrt_z_mutant_slope <- sign(beta_mutant) * sqrt(pmax(lr_mutant_slope, 0))
  signed_qlf_stat_interaction <- sign(beta_interaction) * sqrt(pmax(qlf_stat_interaction, 0))
  signed_qlf_stat_mutant_slope <- sign(beta_mutant) * sqrt(pmax(qlf_stat_mutant_slope, 0))

  significant_interaction <- (pvals_adj_interaction < alpha_padj) &
    (abs(log2_interaction) >= effect_thr)
  significant_mutant_slope <- (pvals_adj_mutant_slope < alpha_padj) &
    (abs(log2_mutant) >= effect_thr)
  expr_summary <- nbglm_expression_summary(counts, size_factors)

  observed_lines <- as.character(cd[[cols$cell_line_col]])
  control_lines_observed <- sort(unique(observed_lines[status0 == "control"]))
  mutant_lines_observed <- sort(unique(observed_lines[status0 == "mutant"]))

  res <- data.frame(
    regression_type = regression_type,
    object_type = object_type,
    target_type = target_type,
    morphology_feature = morphology_feature,
    cell_line = pooled_label,
    names = rownames(counts),
    tested_coef = design_info$interaction_coef,
    beta0 = fit_out$fit$coefficients[, 1],
    beta_predictor_control = beta_control,
    beta_mutant_status = beta_mutant_status,
    beta_predictor_mutant_interaction = beta_interaction,
    beta_predictor_mutant = beta_mutant,
    log2_rate_ratio_control_per_predictor = log2_control,
    log2_rate_ratio_mutant_interaction_per_predictor = log2_interaction,
    log2_rate_ratio_mutant_per_predictor = log2_mutant,
    lr_stat = lr_interaction,
    signed_lrt_z = signed_lrt_z_interaction,
    pvals = pvals_interaction,
    pvals_adj = pvals_adj_interaction,
    qlf_stat = qlf_stat_interaction,
    signed_qlf_stat = signed_qlf_stat_interaction,
    qlf_pvals = qlf_pvals_interaction,
    qlf_pvals_adj = qlf_pvals_adj_interaction,
    significant = significant_interaction,
    lr_stat_mutant_slope = lr_mutant_slope,
    signed_lrt_z_mutant_slope = signed_lrt_z_mutant_slope,
    pvals_mutant_slope = pvals_mutant_slope,
    pvals_adj_mutant_slope = pvals_adj_mutant_slope,
    qlf_stat_mutant_slope = qlf_stat_mutant_slope,
    signed_qlf_stat_mutant_slope = signed_qlf_stat_mutant_slope,
    qlf_pvals_mutant_slope = qlf_pvals_mutant_slope,
    qlf_pvals_adj_mutant_slope = qlf_pvals_adj_mutant_slope,
    significant_mutant_slope = significant_mutant_slope,
    mutant_specific_candidate = significant_interaction & significant_mutant_slope,
    n_obs = ncol(counts),
    n_control_obs = as.integer(status_counts[["control"]]),
    n_mutant_obs = as.integer(status_counts[["mutant"]]),
    n_expr_obs = as.numeric(Matrix::rowSums(counts > 0)),
    predictor_mode = pred$mode,
    predictor_mode_requested = pred$predictor_mode_requested,
    predictor_mode_resolved = pred$predictor_mode_resolved,
    predictor_transform_scope = pred$predictor_transform_scope,
    predictor_transform_detail = pred$predictor_transform_detail,
    predictor_raw_mean = pred$raw_mean,
    predictor_raw_sd = pred$raw_sd,
    predictor_raw_min = pred$raw_min,
    predictor_raw_max = pred$raw_max,
    predictor_global_rank_n = pred$global_rank_n,
    predictor_global_rank_raw_mean = pred$global_rank_raw_mean,
    predictor_global_rank_raw_sd = pred$global_rank_raw_sd,
    predictor_global_rank_raw_min = pred$global_rank_raw_min,
    predictor_global_rank_raw_max = pred$global_rank_raw_max,
    used_slide_covariate = design_info$used_slide,
    design_cols = paste(design_info$design_cols, collapse = ";"),
    control_lines = paste(mutant_status_control_lines, collapse = ";"),
    mutant_lines = paste(mutant_status_mutant_lines, collapse = ";"),
    control_lines_observed = paste(control_lines_observed, collapse = ";"),
    mutant_lines_observed = paste(mutant_lines_observed, collapse = ";"),
    stringsAsFactors = FALSE
  )
  res <- merge(res, expr_summary, by = "names", sort = FALSE)
  res <- res[match(rownames(counts), res$names), , drop = FALSE]

  summary <- data.frame(
    regression_type = regression_type,
    object_type = object_type,
    target_type = target_type,
    morphology_feature = morphology_feature,
    cell_line = pooled_label,
    n_obs = ncol(counts),
    n_control_obs = as.integer(status_counts[["control"]]),
    n_mutant_obs = as.integer(status_counts[["mutant"]]),
    n_features_tested = nrow(counts),
    n_significant_interaction = sum(res$significant, na.rm = TRUE),
    n_significant_mutant_slope = sum(res$significant_mutant_slope, na.rm = TRUE),
    n_mutant_specific_candidates = sum(res$mutant_specific_candidate, na.rm = TRUE),
    predictor_mode = pred$mode,
    predictor_mode_requested = pred$predictor_mode_requested,
    predictor_mode_resolved = pred$predictor_mode_resolved,
    predictor_transform_scope = pred$predictor_transform_scope,
    predictor_transform_detail = pred$predictor_transform_detail,
    predictor_raw_mean = pred$raw_mean,
    predictor_raw_sd = pred$raw_sd,
    predictor_raw_min = pred$raw_min,
    predictor_raw_max = pred$raw_max,
    predictor_global_rank_n = pred$global_rank_n,
    used_slide_covariate = design_info$used_slide,
    control_lines = paste(mutant_status_control_lines, collapse = ";"),
    mutant_lines = paste(mutant_status_mutant_lines, collapse = ";"),
    control_lines_observed = paste(control_lines_observed, collapse = ";"),
    mutant_lines_observed = paste(mutant_lines_observed, collapse = ";"),
    elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")),
    stringsAsFactors = FALSE
  )

  message(sprintf("[%s/%s/%s] mutant-status interaction fitted %d genes; interaction-significant=%d",
                  object_type, morphology_feature, target_type,
                  nrow(counts), sum(res$significant, na.rm = TRUE)))

  list(
    results = res,
    summary = summary,
    skipped = nbglm_empty_skip_df(),
    predictor_info = predictor_info
  )
}

nbglm_read_decoupler_scores <- function(decoupler_basepath, object_type, target_type) {
  spec <- nbglm_decoupler_spec(object_type, target_type)
  csv_path <- file.path(decoupler_basepath, spec$filename)
  if (!file.exists(csv_path)) {
    stop("Missing decoupleR CSV: ", csv_path)
  }

  message("Reading decoupleR CSV: ", csv_path)
  df <- utils::read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 2) stop("decoupleR CSV has too few columns: ", csv_path)

  first_col <- colnames(df)[1]
  row_id <- as.character(df[[first_col]])

  availability_idx <- match(spec$availability_col, colnames(df))
  if (is.na(availability_idx)) {
    stop("Availability column '", spec$availability_col, "' not found in ", csv_path)
  }

  trailing_cols <- colnames(df)[seq.int(availability_idx + 1L, ncol(df))]
  score_cols <- trailing_cols[
    trailing_cols != "decoupleR_export_row" &
      !Reduce(`|`, lapply(spec$reduction_prefixes, function(prefix) startsWith(trailing_cols, prefix)))
  ]
  if (!length(score_cols)) stop("No decoupleR score columns found in ", csv_path)

  score_df <- as.data.frame(lapply(df[score_cols], function(x) suppressWarnings(as.numeric(x))),
                            check.names = FALSE, stringsAsFactors = FALSE)
  rownames(score_df) <- make.unique(row_id)

  export_ids <- list()
  make_export_key <- function(cols) {
    if (!all(cols %in% colnames(df))) return(NULL)
    parts <- lapply(cols, function(col) {
      x <- trimws(as.character(df[[col]]))
      x[is.na(x)] <- ""
      x
    })
    do.call(paste, c(parts, sep = "|"))
  }
  first_existing_col <- function(candidates) {
    hits <- candidates[candidates %in% colnames(df)]
    if (length(hits)) hits[[1]] else NA_character_
  }

  slide_col <- first_existing_col(c("Slide.Name", "Slide Name", "slide_name", "slide"))
  field_col <- first_existing_col(c("field", "Field", "field_key"))
  patch_col <- first_existing_col(c("patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"))
  local_col <- first_existing_col(c("local_id", "local_id_key"))

  if (!is.na(slide_col) && "object_id" %in% colnames(df)) {
    export_ids$slide_object_id <- make_export_key(c(slide_col, "object_id"))
  }
  if (!any(is.na(c(slide_col, field_col, patch_col, local_col)))) {
    export_ids$slide_field_patch_local_id <- make_export_key(c(slide_col, field_col, patch_col, local_col))
  }
  export_ids[[first_col]] <- row_id
  if ("object_id" %in% colnames(df)) {
    export_ids$object_id <- as.character(df$object_id)
  }
  if ("decoupleR_cell_id" %in% colnames(df)) {
    export_ids$decoupleR_cell_id <- as.character(df$decoupleR_cell_id)
  }

  list(
    scores = score_df,
    export_ids = export_ids,
    score_cols = score_cols,
    source_file = csv_path,
    availability_col = spec$availability_col
  )
}

nbglm_choose_decoupler_alignment <- function(sce, decoupler_export) {
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  sce_ids <- list()
  make_sce_key <- function(cols) {
    if (!all(cols %in% colnames(cd))) return(NULL)
    parts <- lapply(cols, function(col) {
      x <- trimws(as.character(cd[[col]]))
      x[is.na(x)] <- ""
      x
    })
    do.call(paste, c(parts, sep = "|"))
  }
  first_existing_col <- function(candidates) {
    hits <- candidates[candidates %in% colnames(cd)]
    if (length(hits)) hits[[1]] else NA_character_
  }

  slide_col <- first_existing_col(c("Slide.Name", "Slide Name", "slide_name", "slide"))
  field_col <- first_existing_col(c("field", "Field", "field_key"))
  patch_col <- first_existing_col(c("patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"))
  local_col <- first_existing_col(c("local_id", "local_id_key"))

  if (!is.na(slide_col) && "object_id" %in% colnames(cd)) {
    sce_ids$slide_object_id <- make_sce_key(c(slide_col, "object_id"))
  }
  if (!any(is.na(c(slide_col, field_col, patch_col, local_col)))) {
    sce_ids$slide_field_patch_local_id <- make_sce_key(c(slide_col, field_col, patch_col, local_col))
  }
  sce_ids$colnames <- colnames(sce)
  if ("object_id" %in% colnames(cd)) {
    sce_ids$object_id <- as.character(cd$object_id)
  }
  if ("decoupleR_cell_id" %in% colnames(cd)) {
    sce_ids$decoupleR_cell_id <- as.character(cd$decoupleR_cell_id)
  }

  best <- list(overlap = -1L, sce_key = NA_character_, export_key = NA_character_,
               sce_values = NULL, export_values = NULL)
  for (sce_key in names(sce_ids)) {
    sce_values <- as.character(sce_ids[[sce_key]])
    valid_sce <- !is.na(sce_values) & nzchar(sce_values)
    if (anyDuplicated(sce_values[valid_sce])) next
    for (export_key in names(decoupler_export$export_ids)) {
      export_values <- as.character(decoupler_export$export_ids[[export_key]])
      valid_export <- !is.na(export_values) & nzchar(export_values)
      if (anyDuplicated(export_values[valid_export])) next
      overlap <- sum(sce_values[valid_sce] %in% export_values[valid_export])
      if (overlap > best$overlap) {
        best <- list(
          overlap = overlap,
          sce_key = sce_key,
          export_key = export_key,
          sce_values = sce_values,
          export_values = export_values
        )
      }
    }
  }

  if (best$overlap <= 0) {
    stop("Could not align decoupleR scores to SCE columns. Tried SCE keys: ",
         paste(names(sce_ids), collapse = ", "), "; export keys: ",
         paste(names(decoupler_export$export_ids), collapse = ", "))
  }

  index <- match(best$sce_values, best$export_values)
  if (anyNA(index)) {
    missing_n <- sum(is.na(index))
    stop(
      "Partial decoupleR alignment only matched ", best$overlap, "/", length(best$sce_values),
      " SCE columns using ", best$sce_key, " -> ", best$export_key,
      ". Missing rows: ", missing_n,
      ". Recompute decoupleR scores or align using stable slide/object keys."
    )
  }
  list(
    index = index,
    sce_key = best$sce_key,
    export_key = best$export_key,
    matched_obs = best$overlap
  )
}

nbglm_lm_fit_one <- function(y, design, coef_idx) {
  fit <- stats::lm.fit(x = design, y = y)
  if (is.na(fit$coefficients[coef_idx])) {
    return(list(beta = NA_real_, se = NA_real_, statistic = NA_real_, pval = NA_real_,
                df_resid = fit$df.residual, sigma = NA_real_))
  }
  if (fit$df.residual <= 0) {
    return(list(beta = fit$coefficients[coef_idx], se = NA_real_,
                statistic = NA_real_, pval = NA_real_,
                df_resid = fit$df.residual, sigma = NA_real_))
  }

  qr_obj <- fit$qr
  rank <- fit$rank
  piv <- qr_obj$pivot[seq_len(rank)]
  r <- qr.R(qr_obj)[seq_len(rank), seq_len(rank), drop = FALSE]
  cov_rank <- chol2inv(r)
  rss <- sum(fit$residuals^2)
  sigma2 <- rss / fit$df.residual
  cov_full <- matrix(NA_real_, nrow = ncol(design), ncol = ncol(design))
  cov_full[piv, piv] <- cov_rank * sigma2

  se <- sqrt(cov_full[coef_idx, coef_idx])
  stat <- fit$coefficients[coef_idx] / se
  pval <- 2 * stats::pt(abs(stat), df = fit$df.residual, lower.tail = FALSE)
  list(beta = fit$coefficients[coef_idx], se = se, statistic = stat, pval = pval,
       df_resid = fit$df.residual, sigma = sqrt(sigma2))
}

nbglm_run_decoupler_condition <- function(
  sce,
  decoupler_export,
  object_type,
  target_type,
  morphology_feature,
  cell_lines_to_include = character(0),
  predictor_mode = "zscore",
  slide_covariate = TRUE,
  min_obs_per_cell_line = 100L,
  alpha_padj = 0.05,
  effect_thr = 0.15,
  global_rank_feature_patterns = c("nuclear_aberration")
) {
  cd_all <- as.data.frame(SummarizedExperiment::colData(sce))
  if (!(morphology_feature %in% colnames(cd_all))) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(
        object_type, target_type, morphology_feature,
        reason = "missing_morphology_feature",
        details = paste("Available columns:", paste(colnames(cd_all), collapse = ", "))
      )
    ))
  }

  cols <- nbglm_resolve_metadata_cols(sce)
  cell_lines <- nbglm_cell_lines(sce, cols$cell_line_col, cell_lines_to_include)
  if (!length(cell_lines)) {
    return(list(
      results = data.frame(stringsAsFactors = FALSE),
      summary = data.frame(stringsAsFactors = FALSE),
      skipped = nbglm_skip_row(object_type, target_type, morphology_feature,
                               reason = "no_matching_cell_lines")
    ))
  }

  mode_info <- nbglm_resolve_predictor_mode(
    morphology_feature = morphology_feature,
    predictor_mode = predictor_mode,
    global_rank_feature_patterns = global_rank_feature_patterns
  )
  global_info <- NULL
  if (identical(mode_info$resolved, "global_rank")) {
    global_mask <- as.character(cd_all[[cols$cell_line_col]]) %in% cell_lines
    global_info <- tryCatch(
      nbglm_make_global_rank_predictor(cd_all[[morphology_feature]], global_mask = global_mask),
      error = function(e) e
    )
    if (inherits(global_info, "error")) {
      return(list(
        results = data.frame(stringsAsFactors = FALSE),
        summary = data.frame(stringsAsFactors = FALSE),
        skipped = nbglm_skip_row(
          object_type, target_type, morphology_feature,
          reason = "invalid_global_rank_predictor",
          details = conditionMessage(global_info)
        ),
        predictor_info = nbglm_predictor_info_row(object_type, target_type, morphology_feature, mode_info)
      ))
    }
  }
  predictor_info <- nbglm_predictor_info_row(
    object_type, target_type, morphology_feature, mode_info, global_info = global_info
  )

  alignment <- nbglm_choose_decoupler_alignment(sce, decoupler_export)
  aligned_scores <- decoupler_export$scores[alignment$index, , drop = FALSE]
  rownames(aligned_scores) <- colnames(sce)

  all_results <- list()
  all_summary <- list()
  all_skips <- list()

  for (cl in cell_lines) {
    t0 <- Sys.time()
    message(sprintf("[%s/%s/%s] %s: starting decoupleR linear models",
                    object_type, morphology_feature, target_type, cl))

    keep_cl <- as.character(SummarizedExperiment::colData(sce)[[cols$cell_line_col]]) == cl
    cd <- cd_all[keep_cl, , drop = FALSE]
    score_sub <- aligned_scores[keep_cl, , drop = FALSE]
    predictor_raw <- suppressWarnings(as.numeric(as.character(cd[[morphology_feature]])))
    global_x_cl <- if (identical(mode_info$resolved, "global_rank")) global_info$x[keep_cl] else NULL
    finite_predictor <- is.finite(predictor_raw)
    if (identical(mode_info$resolved, "global_rank")) {
      finite_predictor <- finite_predictor & is.finite(global_x_cl)
    }
    score_available <- rowSums(is.finite(as.matrix(score_sub))) > 0
    keep <- finite_predictor & score_available

    if (sum(keep) < min_obs_per_cell_line) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "too_few_observations",
        details = paste0("n_obs=", sum(keep), "; min_obs_per_cell_line=", min_obs_per_cell_line)
      )
      next
    }

    cd_model <- cd[keep, , drop = FALSE]
    score_model <- score_sub[keep, , drop = FALSE]
    global_x_model <- if (identical(mode_info$resolved, "global_rank")) global_x_cl[keep] else NULL
    pred <- tryCatch(
      nbglm_prepare_model_predictor(
        cd_model[[morphology_feature]],
        mode_info = mode_info,
        global_x = global_x_model,
        global_info = global_info
      ),
      error = function(e) e
    )
    if (inherits(pred, "error")) {
      all_skips[[length(all_skips) + 1L]] <- nbglm_skip_row(
        object_type, target_type, morphology_feature, cl,
        reason = "invalid_predictor",
        details = conditionMessage(pred)
      )
      next
    }

    design_info <- nbglm_build_design(
      col_data = cd_model,
      predictor = pred$x,
      slide_col = cols$slide_col,
      slide_covariate = slide_covariate
    )
    coef_idx <- which(colnames(design_info$design) == design_info$predictor_coef)
    if (length(coef_idx) != 1) {
      stop("Could not find predictor coefficient in design: ",
           paste(colnames(design_info$design), collapse = ", "))
    }

    feature_results <- vector("list", ncol(score_model))
    for (j in seq_len(ncol(score_model))) {
      nm <- colnames(score_model)[j]
      y <- suppressWarnings(as.numeric(score_model[[j]]))
      finite_y <- is.finite(y)
      if (sum(finite_y) < min_obs_per_cell_line) {
        feature_results[[j]] <- data.frame(
          object_type = object_type,
          target_type = target_type,
          morphology_feature = morphology_feature,
          cell_line = cl,
          names = nm,
          beta_predictor = NA_real_,
          beta_predictor_se = NA_real_,
          t_stat = NA_real_,
          pvals = NA_real_,
          pvals_adj = NA_real_,
          significant = FALSE,
          n_obs = sum(finite_y),
          mean_score = mean(y[finite_y], na.rm = TRUE),
          sd_score = stats::sd(y[finite_y], na.rm = TRUE),
          predictor_mode = pred$mode,
          predictor_mode_requested = pred$predictor_mode_requested,
          predictor_mode_resolved = pred$predictor_mode_resolved,
          predictor_transform_scope = pred$predictor_transform_scope,
          predictor_transform_detail = pred$predictor_transform_detail,
          predictor_raw_mean = pred$raw_mean,
          predictor_raw_sd = pred$raw_sd,
          predictor_raw_min = pred$raw_min,
          predictor_raw_max = pred$raw_max,
          predictor_global_rank_n = pred$global_rank_n,
          predictor_global_rank_raw_mean = pred$global_rank_raw_mean,
          predictor_global_rank_raw_sd = pred$global_rank_raw_sd,
          predictor_global_rank_raw_min = pred$global_rank_raw_min,
          predictor_global_rank_raw_max = pred$global_rank_raw_max,
          used_slide_covariate = design_info$used_slide,
          design_cols = paste(design_info$design_cols, collapse = ";"),
          alignment_sce_key = alignment$sce_key,
          alignment_export_key = alignment$export_key,
          decoupler_source_file = decoupler_export$source_file,
          stringsAsFactors = FALSE
        )
        next
      }

      design_j <- if (all(finite_y)) {
        design_info$design
      } else {
        nbglm_build_design(
          col_data = cd_model[finite_y, , drop = FALSE],
          predictor = pred$x[finite_y],
          slide_col = cols$slide_col,
          slide_covariate = slide_covariate
        )$design
      }
      coef_idx_j <- which(colnames(design_j) == "predictor")
      fit_j <- tryCatch(nbglm_lm_fit_one(y[finite_y], design_j, coef_idx_j),
                        error = function(e) e)
      if (inherits(fit_j, "error")) {
        feature_results[[j]] <- data.frame(
          object_type = object_type,
          target_type = target_type,
          morphology_feature = morphology_feature,
          cell_line = cl,
          names = nm,
          beta_predictor = NA_real_,
          beta_predictor_se = NA_real_,
          t_stat = NA_real_,
          pvals = NA_real_,
          pvals_adj = NA_real_,
          significant = FALSE,
          n_obs = sum(finite_y),
          mean_score = mean(y[finite_y], na.rm = TRUE),
          sd_score = stats::sd(y[finite_y], na.rm = TRUE),
          predictor_mode = pred$mode,
          predictor_mode_requested = pred$predictor_mode_requested,
          predictor_mode_resolved = pred$predictor_mode_resolved,
          predictor_transform_scope = pred$predictor_transform_scope,
          predictor_transform_detail = pred$predictor_transform_detail,
          predictor_raw_mean = pred$raw_mean,
          predictor_raw_sd = pred$raw_sd,
          predictor_raw_min = pred$raw_min,
          predictor_raw_max = pred$raw_max,
          predictor_global_rank_n = pred$global_rank_n,
          predictor_global_rank_raw_mean = pred$global_rank_raw_mean,
          predictor_global_rank_raw_sd = pred$global_rank_raw_sd,
          predictor_global_rank_raw_min = pred$global_rank_raw_min,
          predictor_global_rank_raw_max = pred$global_rank_raw_max,
          used_slide_covariate = design_info$used_slide,
          design_cols = paste(design_info$design_cols, collapse = ";"),
          alignment_sce_key = alignment$sce_key,
          alignment_export_key = alignment$export_key,
          decoupler_source_file = decoupler_export$source_file,
          stringsAsFactors = FALSE
        )
      } else {
        feature_results[[j]] <- data.frame(
          object_type = object_type,
          target_type = target_type,
          morphology_feature = morphology_feature,
          cell_line = cl,
          names = nm,
          beta_predictor = fit_j$beta,
          beta_predictor_se = fit_j$se,
          t_stat = fit_j$statistic,
          pvals = fit_j$pval,
          pvals_adj = NA_real_,
          significant = FALSE,
          n_obs = sum(finite_y),
          mean_score = mean(y[finite_y], na.rm = TRUE),
          sd_score = stats::sd(y[finite_y], na.rm = TRUE),
          predictor_mode = pred$mode,
          predictor_mode_requested = pred$predictor_mode_requested,
          predictor_mode_resolved = pred$predictor_mode_resolved,
          predictor_transform_scope = pred$predictor_transform_scope,
          predictor_transform_detail = pred$predictor_transform_detail,
          predictor_raw_mean = pred$raw_mean,
          predictor_raw_sd = pred$raw_sd,
          predictor_raw_min = pred$raw_min,
          predictor_raw_max = pred$raw_max,
          predictor_global_rank_n = pred$global_rank_n,
          predictor_global_rank_raw_mean = pred$global_rank_raw_mean,
          predictor_global_rank_raw_sd = pred$global_rank_raw_sd,
          predictor_global_rank_raw_min = pred$global_rank_raw_min,
          predictor_global_rank_raw_max = pred$global_rank_raw_max,
          used_slide_covariate = design_info$used_slide,
          design_cols = paste(design_info$design_cols, collapse = ";"),
          alignment_sce_key = alignment$sce_key,
          alignment_export_key = alignment$export_key,
          decoupler_source_file = decoupler_export$source_file,
          stringsAsFactors = FALSE
        )
      }
    }

    res <- nbglm_bind_rows(feature_results)
    if (nrow(res) > 0) {
      res$pvals_adj <- stats::p.adjust(res$pvals, method = "BH")
      res$significant <- (res$pvals_adj < alpha_padj) & (abs(res$beta_predictor) >= effect_thr)
    }
    all_results[[length(all_results) + 1L]] <- res
    all_summary[[length(all_summary) + 1L]] <- data.frame(
      object_type = object_type,
      target_type = target_type,
      morphology_feature = morphology_feature,
      cell_line = cl,
      n_obs = sum(keep),
      n_features_tested = ncol(score_model),
      n_significant = sum(res$significant, na.rm = TRUE),
      predictor_mode = pred$mode,
      predictor_mode_requested = pred$predictor_mode_requested,
      predictor_mode_resolved = pred$predictor_mode_resolved,
      predictor_transform_scope = pred$predictor_transform_scope,
      predictor_transform_detail = pred$predictor_transform_detail,
      predictor_raw_mean = pred$raw_mean,
      predictor_raw_sd = pred$raw_sd,
      predictor_raw_min = pred$raw_min,
      predictor_raw_max = pred$raw_max,
      predictor_global_rank_n = pred$global_rank_n,
      used_slide_covariate = design_info$used_slide,
      alignment_sce_key = alignment$sce_key,
      alignment_export_key = alignment$export_key,
      matched_obs = alignment$matched_obs,
      elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      stringsAsFactors = FALSE
    )

    message(sprintf("[%s/%s/%s] %s: fitted %d features; significant=%d",
                    object_type, morphology_feature, target_type, cl,
                    ncol(score_model), sum(res$significant, na.rm = TRUE)))
  }

  list(
    results = nbglm_bind_rows(all_results),
    summary = nbglm_bind_rows(all_summary),
    skipped = if (length(all_skips)) nbglm_bind_rows(all_skips) else nbglm_empty_skip_df(),
    predictor_info = predictor_info
  )
}

nbglm_write_condition_outputs <- function(out_dir, result) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  nbglm_write_csv(result$results, file.path(out_dir, "regression_results_all_cell_lines.csv"))
  nbglm_write_csv(result$summary, file.path(out_dir, "model_summary_by_cell_line.csv"))
  skipped <- result$skipped
  if (is.null(skipped) || !nrow(skipped)) skipped <- nbglm_empty_skip_df()
  nbglm_write_csv(skipped, file.path(out_dir, "skipped_models.csv"))
  if (!is.null(result$predictor_info) && nrow(result$predictor_info) > 0) {
    nbglm_write_csv(result$predictor_info, file.path(out_dir, "predictor_transform.csv"))
  }
}

nbglm_write_mutant_status_interaction_outputs <- function(out_dir, result) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  nbglm_write_csv(
    result$results,
    file.path(out_dir, "mutant_status_interaction_regression_results.csv")
  )
  nbglm_write_csv(
    result$summary,
    file.path(out_dir, "mutant_status_interaction_model_summary.csv")
  )
  skipped <- result$skipped
  if (is.null(skipped) || !nrow(skipped)) skipped <- nbglm_empty_skip_df()
  nbglm_write_csv(skipped, file.path(out_dir, "mutant_status_interaction_skipped_models.csv"))
  if (!is.null(result$predictor_info) && nrow(result$predictor_info) > 0) {
    nbglm_write_csv(
      result$predictor_info,
      file.path(out_dir, "mutant_status_interaction_predictor_transform.csv")
    )
  }
}

nbglm_myotube_nucleus_count <- function(col_data) {
  if ("n_nuclei" %in% colnames(col_data)) {
    return(list(
      count = suppressWarnings(as.numeric(as.character(col_data$n_nuclei))),
      source = "n_nuclei"
    ))
  }

  if ("n_myonuclei" %in% colnames(col_data)) {
    return(list(
      count = suppressWarnings(as.numeric(as.character(col_data$n_myonuclei))),
      source = "n_myonuclei"
    ))
  }

  count_cols <- intersect(c("n_normal_nuclei", "n_abnormal_nuclei"), colnames(col_data))
  if (length(count_cols) > 0) {
    counts <- lapply(count_cols, function(col) {
      x <- suppressWarnings(as.numeric(as.character(col_data[[col]])))
      x[!is.finite(x)] <- 0
      x
    })
    total <- Reduce(`+`, counts)
    return(list(count = total, source = paste(count_cols, collapse = "+")))
  }

  NULL
}

nbglm_filter_myonuclei <- function(sce, filter_myonuclei = TRUE,
                                   myonucleus_col = "is_myonucleus") {
  if (!isTRUE(filter_myonuclei)) return(sce)

  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  if (!(myonucleus_col %in% colnames(cd))) {
    warning(
      "[myonuclei] filter_myonuclei=TRUE but column '", myonucleus_col,
      "' was not found; assuming input RDS is already restricted to myonuclei."
    )
    return(sce)
  }

  keep <- to_myonucleus_flag(cd[[myonucleus_col]])
  message(sprintf(
    "[myonuclei] %s filter retained %d/%d objects.",
    myonucleus_col,
    sum(keep, na.rm = TRUE),
    length(keep)
  ))
  sce[, keep, drop = FALSE]
}

nbglm_filter_myotubes_by_myonuclei <- function(sce, min_myonuclei_per_myotube = 1L) {
  if (is.null(min_myonuclei_per_myotube) ||
      is.na(min_myonuclei_per_myotube) ||
      as.numeric(min_myonuclei_per_myotube) <= 0) {
    return(sce)
  }

  min_myonuclei_per_myotube <- as.numeric(min_myonuclei_per_myotube)
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  count_info <- nbglm_myotube_nucleus_count(cd)
  if (is.null(count_info)) {
    stop(
      "[myotubes] Cannot apply min_myonuclei_per_myotube=",
      min_myonuclei_per_myotube,
      " because none of these columns were found: n_nuclei, n_myonuclei, ",
      "n_normal_nuclei, n_abnormal_nuclei. Available columns: ",
      paste(colnames(cd), collapse = ", ")
    )
  }

  counts <- count_info$count
  keep <- is.finite(counts) & counts >= min_myonuclei_per_myotube
  SummarizedExperiment::colData(sce)$n_myonuclei_for_filter <- counts
  message(sprintf(
    "[myotubes] min_myonuclei_per_myotube >= %s using %s retained %d/%d myotubes.",
    format(min_myonuclei_per_myotube, trim = TRUE, scientific = FALSE),
    count_info$source,
    sum(keep),
    length(keep)
  ))
  sce[, keep, drop = FALSE]
}

nbglm_filter_loaded_object <- function(sce, object_type,
                                       min_myonuclei_per_myotube = 1L,
                                       filter_myonuclei = TRUE,
                                       myonucleus_col = "is_myonucleus") {
  if (identical(object_type, "myotubes")) {
    return(nbglm_filter_myotubes_by_myonuclei(
      sce,
      min_myonuclei_per_myotube = min_myonuclei_per_myotube
    ))
  }

  if (identical(object_type, "myonuclei")) {
    return(nbglm_filter_myonuclei(
      sce,
      filter_myonuclei = filter_myonuclei,
      myonucleus_col = myonucleus_col
    ))
  }

  sce
}

nbglm_load_sce_for_object <- function(object_type, myonuclei_rds_path, myotube_rds_path,
                                      min_myonuclei_per_myotube = 1L,
                                      filter_myonuclei = TRUE,
                                      myonucleus_col = "is_myonucleus") {
  path <- if (identical(object_type, "myonuclei")) myonuclei_rds_path else myotube_rds_path
  stop_if_missing_file(path, paste0(object_type, "_rds_path"))
  message("[", object_type, "] Reading RDS: ", path)
  obj <- readRDS(path)
  if (inherits(obj, "SummarizedExperiment") && !inherits(obj, "SingleCellExperiment")) {
    obj <- methods::as(obj, "SingleCellExperiment")
  }
  if (!inherits(obj, "SingleCellExperiment")) {
    stop("[", object_type, "] Expected SingleCellExperiment-compatible RDS, got: ",
         paste(class(obj), collapse = ", "))
  }
  nbglm_filter_loaded_object(
    obj,
    object_type = object_type,
    min_myonuclei_per_myotube = min_myonuclei_per_myotube,
    filter_myonuclei = filter_myonuclei,
    myonucleus_col = myonucleus_col
  )
}

nbglm_features_for_object <- function(object_type,
                                      myonuclei_morphology_features = character(0),
                                      myotube_morphology_features = character(0),
                                      morphology_features = character(0)) {
  legacy_features <- nbglm_parse_list_arg(morphology_features)
  if (identical(object_type, "myonuclei")) {
    return(nbglm_first_nonempty(myonuclei_morphology_features, legacy_features))
  }
  if (identical(object_type, "myotubes")) {
    return(nbglm_first_nonempty(myotube_morphology_features, legacy_features))
  }
  character(0)
}

nbglm_predictor_plan_df <- function(object_types,
                                    myonuclei_morphology_features = character(0),
                                    myotube_morphology_features = character(0),
                                    morphology_features = character(0),
                                    predictor_mode = "auto",
                                    global_rank_feature_patterns = c("nuclear_aberration")) {
  object_types <- nbglm_canonical_object_type(object_types)
  rows <- list()
  for (object_type in object_types) {
    features <- nbglm_features_for_object(
      object_type = object_type,
      myonuclei_morphology_features = myonuclei_morphology_features,
      myotube_morphology_features = myotube_morphology_features,
      morphology_features = morphology_features
    )
    if (!length(features)) next
    for (feature in features) {
      mode_info <- nbglm_resolve_predictor_mode(
        morphology_feature = feature,
        predictor_mode = predictor_mode,
        global_rank_feature_patterns = global_rank_feature_patterns
      )
      rows[[length(rows) + 1L]] <- data.frame(
        object_type = object_type,
        morphology_feature = feature,
        predictor_mode_requested = mode_info$requested,
        predictor_mode_resolved = mode_info$resolved,
        predictor_transform_scope = mode_info$scope,
        predictor_transform_detail = mode_info$detail,
        stringsAsFactors = FALSE
      )
    }
  }
  nbglm_bind_rows(rows)
}

nbglm_run_workflow <- function(
  myonuclei_rds_path,
  myotube_rds_path,
  decoupler_basepath,
  output_dir,
  object_types,
  target_types,
  myonuclei_morphology_features = character(0),
  myotube_morphology_features = character(0),
  morphology_features = character(0),
  cell_lines_to_include = character(0),
  run_mutant_status_interaction = TRUE,
  mutant_status_control_lines = character(0),
  mutant_status_mutant_lines = character(0),
  predictor_mode = "auto",
  global_rank_feature_patterns = c("nuclear_aberration"),
  slide_covariate = TRUE,
  min_obs_per_cell_line = 100L,
  min_expr_obs = 20L,
  alpha_padj = 0.05,
  effect_thr = 0.15,
  min_myonuclei_per_myotube = 1L,
  filter_myonuclei = TRUE,
  myonucleus_col = "is_myonucleus"
) {
  object_types <- nbglm_canonical_object_type(object_types)
  target_types <- nbglm_canonical_target_type(target_types)
  myonuclei_morphology_features <- nbglm_parse_list_arg(myonuclei_morphology_features)
  myotube_morphology_features <- nbglm_parse_list_arg(myotube_morphology_features)
  morphology_features <- nbglm_parse_list_arg(morphology_features)
  cell_lines_to_include <- nbglm_parse_list_arg(cell_lines_to_include)
  run_mutant_status_interaction <- tolower(trimws(as.character(run_mutant_status_interaction)[1])) %in%
    c("1", "true", "t", "yes", "y")
  mutant_status_control_lines <- nbglm_parse_list_arg(mutant_status_control_lines)
  mutant_status_mutant_lines <- nbglm_parse_list_arg(mutant_status_mutant_lines)
  global_rank_feature_patterns <- nbglm_parse_list_arg(global_rank_feature_patterns)

  missing_feature_lists <- vapply(object_types, function(object_type) {
    !length(nbglm_features_for_object(
      object_type = object_type,
      myonuclei_morphology_features = myonuclei_morphology_features,
      myotube_morphology_features = myotube_morphology_features,
      morphology_features = morphology_features
    ))
  }, logical(1))
  if (any(missing_feature_lists)) {
    stop(
      "No morphology features were provided for object type(s): ",
      paste(object_types[missing_feature_lists], collapse = ", "),
      ". Use --myonuclei_morphology_features and/or --myotube_morphology_features."
    )
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  decoupler_cache <- list()

  for (object_type in object_types) {
    sce <- nbglm_load_sce_for_object(
      object_type = object_type,
      myonuclei_rds_path = myonuclei_rds_path,
      myotube_rds_path = myotube_rds_path,
      min_myonuclei_per_myotube = min_myonuclei_per_myotube,
      filter_myonuclei = filter_myonuclei,
      myonucleus_col = myonucleus_col
    )

    object_morphology_features <- nbglm_features_for_object(
      object_type = object_type,
      myonuclei_morphology_features = myonuclei_morphology_features,
      myotube_morphology_features = myotube_morphology_features,
      morphology_features = morphology_features
    )

    for (morphology_feature in object_morphology_features) {
      for (target_type in target_types) {
        condition_name <- paste(
          nbglm_safe_slug(object_type),
          nbglm_safe_slug(morphology_feature),
          nbglm_safe_slug(target_type),
          sep = "-"
        )
        condition_out_dir <- file.path(output_dir, condition_name)
        message("==== Condition: ", condition_name, " ====")

        result <- if (identical(target_type, "genes")) {
          nbglm_run_gene_condition(
            sce = sce,
            object_type = object_type,
            morphology_feature = morphology_feature,
            cell_lines_to_include = cell_lines_to_include,
            predictor_mode = predictor_mode,
            global_rank_feature_patterns = global_rank_feature_patterns,
            slide_covariate = slide_covariate,
            min_obs_per_cell_line = min_obs_per_cell_line,
            min_expr_obs = min_expr_obs,
            alpha_padj = alpha_padj,
            effect_thr = effect_thr
          )
        } else {
          if (is.null(decoupler_basepath)) {
            list(
              results = data.frame(stringsAsFactors = FALSE),
              summary = data.frame(stringsAsFactors = FALSE),
              skipped = nbglm_skip_row(
                object_type, target_type, morphology_feature,
                reason = "missing_decoupler_basepath",
                details = "Provide --decoupler_basepath for decoupleR targets."
              )
            )
          } else {
            cache_key <- paste(object_type, target_type, sep = "::")
            if (is.null(decoupler_cache[[cache_key]])) {
              decoupler_cache[[cache_key]] <- tryCatch(
                nbglm_read_decoupler_scores(decoupler_basepath, object_type, target_type),
                error = function(e) e
              )
            }
            if (inherits(decoupler_cache[[cache_key]], "error")) {
              list(
                results = data.frame(stringsAsFactors = FALSE),
                summary = data.frame(stringsAsFactors = FALSE),
                skipped = nbglm_skip_row(
                  object_type, target_type, morphology_feature,
                  reason = "decoupler_load_failed",
                  details = conditionMessage(decoupler_cache[[cache_key]])
                )
              )
            } else {
              nbglm_run_decoupler_condition(
                sce = sce,
                decoupler_export = decoupler_cache[[cache_key]],
                object_type = object_type,
                target_type = target_type,
                morphology_feature = morphology_feature,
                cell_lines_to_include = cell_lines_to_include,
                predictor_mode = predictor_mode,
                global_rank_feature_patterns = global_rank_feature_patterns,
                slide_covariate = slide_covariate,
                min_obs_per_cell_line = min_obs_per_cell_line,
                alpha_padj = alpha_padj,
                effect_thr = effect_thr
              )
            }
          }
        }

        nbglm_write_condition_outputs(condition_out_dir, result)
        message("Saved condition outputs: ", condition_out_dir)

        if (identical(target_type, "genes") && isTRUE(run_mutant_status_interaction)) {
          interaction_result <- nbglm_run_gene_mutant_status_interaction(
            sce = sce,
            object_type = object_type,
            morphology_feature = morphology_feature,
            cell_lines_to_include = cell_lines_to_include,
            mutant_status_control_lines = mutant_status_control_lines,
            mutant_status_mutant_lines = mutant_status_mutant_lines,
            predictor_mode = predictor_mode,
            global_rank_feature_patterns = global_rank_feature_patterns,
            slide_covariate = slide_covariate,
            min_obs_per_cell_line = min_obs_per_cell_line,
            min_expr_obs = min_expr_obs,
            alpha_padj = alpha_padj,
            effect_thr = effect_thr
          )
          nbglm_write_mutant_status_interaction_outputs(condition_out_dir, interaction_result)
          message("Saved mutant-status interaction outputs: ", condition_out_dir)
        }
      }
    }
  }

  invisible(TRUE)
}
