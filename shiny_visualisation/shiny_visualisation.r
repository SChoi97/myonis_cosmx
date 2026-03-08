#!/usr/bin/env Rscript

required_pkgs <- c(
  "shiny", "ggplot2", "dplyr", "tidyr", "stringr", "DT",
  "Matrix", "SingleCellExperiment", "SummarizedExperiment", "tibble"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing required R packages: ", paste(missing_pkgs, collapse = ", "), "\n",
    "Install CRAN packages with install.packages(...).\n",
    "For Bioconductor packages (SingleCellExperiment, SummarizedExperiment), use BiocManager::install(...)."
  )
}

suppressPackageStartupMessages({
  library(shiny)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(DT)
  library(Matrix)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(tibble)
})

PREFERRED_CELL_LINE_ORDER <- c("NCRM1", "NCRM5", "L302P-Corr", "L302P", "1174", "1175")

parse_cli_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]

    if (!startsWith(a, "--")) {
      stop("Invalid argument format: ", a, "\nExpected --key value or --key=value")
    }

    if (grepl("=", a, fixed = TRUE)) {
      parts <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
      key <- gsub("-", "_", parts[[1]])
      val <- if (length(parts) > 1) paste(parts[-1], collapse = "=") else ""
      out[[key]] <- val
      i <- i + 1L
      next
    }

    key <- gsub("-", "_", sub("^--", "", a))
    if (i == length(args)) {
      stop("Missing value for argument: ", a)
    }
    next_a <- args[[i + 1L]]
    if (startsWith(next_a, "--")) {
      stop("Missing value for argument: ", a)
    }

    out[[key]] <- next_a
    i <- i + 2L
  }
  out
}

pick_col <- function(df, candidates, label, required = TRUE) {
  hit <- candidates[candidates %in% colnames(df)][1]
  if (is.na(hit)) {
    if (required) {
      stop("Missing required column for ", label, ". Available: ", paste(colnames(df), collapse = ", "))
    }
    return(NA_character_)
  }
  hit
}

strip_ens_version <- function(x) {
  sub("\\.[0-9]+$", "", as.character(x))
}

parse_gene_query <- function(query_text) {
  genes <- stringr::str_split(query_text, ",", simplify = FALSE)[[1]]
  genes <- trimws(genes)
  genes <- genes[nzchar(genes)]
  unique(genes)
}

resolve_genes <- function(requested_genes, available_genes) {
  available <- unique(as.character(available_genes))
  available <- available[nzchar(available)]

  available_lower <- tolower(available)
  available_strip <- strip_ens_version(available)
  available_strip_lower <- tolower(available_strip)

  map_one <- function(g) {
    if (g %in% available) return(g)

    idx <- which(available_lower == tolower(g))
    if (length(idx) > 0) return(available[idx[1]])

    g_strip <- strip_ens_version(g)

    idx <- which(available_strip == g_strip)
    if (length(idx) > 0) return(available[idx[1]])

    idx <- which(available_strip_lower == tolower(g_strip))
    if (length(idx) > 0) return(available[idx[1]])

    NA_character_
  }

  mapped <- vapply(requested_genes, map_one, character(1))
  unresolved <- requested_genes[is.na(mapped)]
  matched <- unique(stats::na.omit(mapped))

  mapping <- tibble::tibble(
    requested_gene = requested_genes,
    resolved_gene = mapped,
    status = ifelse(is.na(mapped), "unresolved", "resolved")
  )

  list(matched = matched, unresolved = unresolved, mapping = mapping)
}

extract_results_by_cell_line <- function(obj, source_label) {
  if (is.list(obj) && !is.null(obj$results_by_cell_line)) {
    return(obj$results_by_cell_line)
  }

  if (is.list(obj) && length(obj) > 0) {
    all_df <- all(vapply(obj, function(x) is.data.frame(x), logical(1)))
    if (all_df) return(obj)
  }

  stop("Could not extract `results_by_cell_line` from ", source_label)
}

choose_deg_file <- function(deg_dir, dataset_key, ext) {
  expected <- file.path(deg_dir, paste0(dataset_key, "_deg_results.", ext))
  if (file.exists(expected)) return(expected)

  candidates <- list.files(
    deg_dir,
    pattern = paste0("deg_results\\.", ext, "$"),
    full.names = TRUE,
    ignore.case = TRUE
  )

  if (length(candidates) == 0) return(NA_character_)

  keyed <- candidates[grepl(dataset_key, basename(candidates), ignore.case = TRUE)]
  if (length(keyed) > 0) return(keyed[1])

  if (length(candidates) == 1) return(candidates[1])

  candidates[1]
}

load_deg_results <- function(deg_dir, dataset_key) {
  if (!dir.exists(deg_dir)) {
    stop("DEG directory not found: ", deg_dir)
  }

  rds_file <- choose_deg_file(deg_dir, dataset_key, "rds")
  rdata_file <- choose_deg_file(deg_dir, dataset_key, "RData")

  if (!is.na(rds_file) && file.exists(rds_file)) {
    obj <- readRDS(rds_file)
    return(extract_results_by_cell_line(obj, rds_file))
  }

  if (!is.na(rdata_file) && file.exists(rdata_file)) {
    env <- new.env(parent = emptyenv())
    load(rdata_file, envir = env)
    if (!exists("results_by_cell_line", envir = env, inherits = FALSE)) {
      stop("`results_by_cell_line` not found in ", rdata_file)
    }
    return(get("results_by_cell_line", envir = env, inherits = FALSE))
  }

  stop("No DEG result file found in ", deg_dir, " (looked for *_deg_results.rds/.RData)")
}

standardise_deg_long <- function(results_by_cell_line, dataset_label) {
  if (length(results_by_cell_line) == 0) {
    stop("results_by_cell_line is empty for dataset ", dataset_label)
  }

  required_cols <- c("names", "log2fc_xenium_eps", "pvals_adj")
  bad_lines <- names(results_by_cell_line)[!vapply(results_by_cell_line, function(df) {
    all(required_cols %in% colnames(as.data.frame(df)))
  }, logical(1))]

  if (length(bad_lines) > 0) {
    stop(
      "Missing required DEG columns (names, log2fc_xenium_eps, pvals_adj) in ",
      dataset_label, " cell lines: ", paste(bad_lines, collapse = ", ")
    )
  }

  pieces <- lapply(names(results_by_cell_line), function(cl) {
    df <- as.data.frame(results_by_cell_line[[cl]])
    tibble::tibble(
      dataset = dataset_label,
      cell_line = as.character(cl),
      gene_id = as.character(df$names),
      log2fc_xenium_eps = as.numeric(df$log2fc_xenium_eps),
      pvals_adj = as.numeric(df$pvals_adj),
      beta1 = if ("beta1" %in% colnames(df)) as.numeric(df$beta1) else NA_real_
    )
  })

  dplyr::bind_rows(pieces) %>%
    dplyr::filter(!is.na(gene_id), nzchar(gene_id)) %>%
    dplyr::distinct(dataset, cell_line, gene_id, .keep_all = TRUE)
}

map_myonuclei_condition <- function(cd) {
  class_col <- pick_col(
    cd,
    c("Predicted Class", "Predicted.Class", "predicted_class", "Classification", "classification"),
    "myonuclei class"
  )

  class_raw <- tolower(trimws(as.character(cd[[class_col]])))
  out <- ifelse(
    class_raw %in% c("0", "normal"),
    "Normal",
    ifelse(class_raw %in% c("1", "abnormal"), "Abnormal", NA_character_)
  )
  out
}

map_myotube_condition <- function(cd) {
  if (!"morphology_class" %in% colnames(cd)) {
    if (all(c("n_normal_nuclei", "n_abnormal_nuclei") %in% colnames(cd))) {
      morph <- rep(-1L, nrow(cd))
      morph[cd$n_normal_nuclei > 0 & cd$n_abnormal_nuclei == 0] <- 1L
      morph[cd$n_normal_nuclei == 0 & cd$n_abnormal_nuclei > 0] <- 2L
      morph[cd$n_normal_nuclei > 0 & cd$n_abnormal_nuclei > 0] <- 3L
      cd$morphology_class <- morph
    } else {
      stop("myotube data needs `morphology_class` or (`n_normal_nuclei`, `n_abnormal_nuclei`) columns")
    }
  }

  morph <- suppressWarnings(as.integer(as.character(cd$morphology_class)))
  ifelse(morph == 1L, "Normal", ifelse(morph == 2L, "Abnormal", NA_character_))
}

compute_size_factors <- function(cnt_fit) {
  totals <- Matrix::colSums(cnt_fit)
  nonzero <- totals[totals > 0]

  sf <- rep(1, length(totals))
  if (length(nonzero) > 0) {
    med_total <- stats::median(nonzero)
    if (is.finite(med_total) && med_total > 0) {
      sf <- totals / med_total
      sf_pos <- sf[is.finite(sf) & sf > 0]
      if (length(sf_pos) > 0) {
        sf[!is.finite(sf) | sf <= 0] <- min(sf_pos) * 1e-3
      } else {
        sf[] <- 1
      }
    }
  }

  sf
}

build_cell_line_order <- function(observed_lines) {
  observed <- unique(trimws(as.character(observed_lines)))
  observed <- observed[nzchar(observed)]

  extras <- setdiff(sort(observed), PREFERRED_CELL_LINE_ORDER)
  unique(c(PREFERRED_CELL_LINE_ORDER, extras))
}

prepare_dataset <- function(dataset_label, sce_obj, results_by_cell_line) {
  if (!"counts" %in% SummarizedExperiment::assayNames(sce_obj)) {
    stop(dataset_label, " object is missing a `counts` assay")
  }

  counts_all <- SummarizedExperiment::assay(sce_obj, "counts")
  if (!inherits(counts_all, "dgCMatrix")) counts_all <- as(counts_all, "dgCMatrix")

  cd <- as.data.frame(SummarizedExperiment::colData(sce_obj))
  cell_line_col <- pick_col(cd, c("Cell Line", "Cell.Line", "cell_line", "CellLine"), paste0(dataset_label, " cell line"))
  slide_col <- pick_col(cd, c("Slide Name", "Slide.Name", "slide_name", "slide"), paste0(dataset_label, " slide"), required = FALSE)
  cell_line <- trimws(as.character(cd[[cell_line_col]]))
  slide_id <- if (is.na(slide_col)) {
    rep("UNKNOWN", nrow(cd))
  } else {
    toupper(trimws(as.character(cd[[slide_col]])))
  }

  condition <- if (identical(dataset_label, "myonuclei")) {
    map_myonuclei_condition(cd)
  } else {
    map_myotube_condition(cd)
  }

  keep <- !is.na(condition) & nzchar(cell_line)
  if (!any(keep)) {
    stop("No usable cells/myotubes found after grouping for ", dataset_label)
  }

  counts <- counts_all[, keep, drop = FALSE]
  cell_line <- cell_line[keep]
  slide_id <- slide_id[keep]
  condition <- factor(condition[keep], levels = c("Normal", "Abnormal"))

  deg_long <- standardise_deg_long(results_by_cell_line, dataset_label)
  cell_line_order <- build_cell_line_order(c(names(results_by_cell_line), unique(cell_line)))
  gene_universe <- unique(c(rownames(counts), deg_long$gene_id))

  list(
    name = dataset_label,
    counts = counts,
    cell_line = cell_line,
    slide_id = slide_id,
    condition = condition,
    results_by_cell_line = results_by_cell_line,
    deg_long = deg_long,
    cell_line_order = cell_line_order,
    gene_universe = gene_universe
  )
}

compute_counts_summary <- function(ds, genes) {
  genes <- genes[genes %in% rownames(ds$counts)]
  if (length(genes) == 0) {
    return(tibble::tibble(
      dataset = character(), cell_line = character(), condition = character(), gene_id = character(),
      n_cells = integer(), mean_raw = numeric(), median_raw = numeric(),
      mean_normalized = numeric(), median_normalized = numeric()
    ))
  }

  out <- list()
  idx_out <- 1L

  for (cl in ds$cell_line_order) {
    idx_line <- which(ds$cell_line == cl)
    if (length(idx_line) == 0) next

    cnt_line <- ds$counts[, idx_line, drop = FALSE]
    cond_line <- as.character(ds$condition[idx_line])

    fit_genes <- rownames(cnt_line)
    if (cl %in% names(ds$results_by_cell_line)) {
      df_line <- as.data.frame(ds$results_by_cell_line[[cl]])
      if ("names" %in% colnames(df_line)) {
        fit_genes_line <- intersect(as.character(df_line$names), rownames(cnt_line))
        if (length(fit_genes_line) > 0) fit_genes <- fit_genes_line
      }
    }

    cnt_fit <- cnt_line[fit_genes, , drop = FALSE]
    sf <- compute_size_factors(cnt_fit)

    cnt_gene <- cnt_line[genes, , drop = FALSE]
    cnt_norm <- Matrix::t(Matrix::t(cnt_gene) / sf)

    for (cond in c("Normal", "Abnormal")) {
      idx_cond <- which(cond_line == cond)

      if (length(idx_cond) == 0) {
        mean_raw <- rep(NA_real_, length(genes))
        median_raw <- rep(NA_real_, length(genes))
        mean_norm <- rep(NA_real_, length(genes))
        median_norm <- rep(NA_real_, length(genes))
        n_cells <- 0L
      } else {
        raw_mat <- as.matrix(cnt_gene[, idx_cond, drop = FALSE])
        norm_mat <- as.matrix(cnt_norm[, idx_cond, drop = FALSE])

        mean_raw <- rowMeans(raw_mat, na.rm = TRUE)
        median_raw <- apply(raw_mat, 1, stats::median, na.rm = TRUE)

        mean_norm <- rowMeans(norm_mat, na.rm = TRUE)
        median_norm <- apply(norm_mat, 1, stats::median, na.rm = TRUE)

        n_cells <- ncol(raw_mat)
      }

      out[[idx_out]] <- tibble::tibble(
        dataset = ds$name,
        cell_line = cl,
        condition = cond,
        gene_id = genes,
        n_cells = n_cells,
        mean_raw = as.numeric(mean_raw),
        median_raw = as.numeric(median_raw),
        mean_normalized = as.numeric(mean_norm),
        median_normalized = as.numeric(median_norm)
      )
      idx_out <- idx_out + 1L
    }
  }

  dplyr::bind_rows(out)
}

compute_fc_summary <- function(ds, genes, eps = 1e-6) {
  genes <- genes[genes %in% rownames(ds$counts)]
  if (length(genes) == 0) {
    return(tibble::tibble(
      dataset = character(),
      cell_line = character(),
      gene_id = character(),
      fold_change = numeric(),
      fold_change_sd = numeric(),
      n_slides = integer()
    ))
  }

  out <- list()
  idx_out <- 1L

  for (cl in ds$cell_line_order) {
    idx_line <- which(ds$cell_line == cl)
    if (length(idx_line) == 0) {
      next
    }

    cnt_line <- ds$counts[, idx_line, drop = FALSE]
    cond_line <- as.character(ds$condition[idx_line])
    slide_line <- ds$slide_id[idx_line]

    mask_abn <- cond_line == "Abnormal"
    mask_norm <- cond_line == "Normal"

    if (sum(mask_abn) == 0 || sum(mask_norm) == 0) {
      out[[idx_out]] <- tibble::tibble(
        dataset = ds$name,
        cell_line = cl,
        gene_id = genes,
        fold_change = NA_real_,
        fold_change_sd = NA_real_,
        n_slides = 0L
      )
      idx_out <- idx_out + 1L
      next
    }

    fit_genes <- rownames(cnt_line)
    if (cl %in% names(ds$results_by_cell_line)) {
      df_line <- as.data.frame(ds$results_by_cell_line[[cl]])
      if ("names" %in% colnames(df_line)) {
        fit_genes_line <- intersect(as.character(df_line$names), rownames(cnt_line))
        if (length(fit_genes_line) > 0) fit_genes <- fit_genes_line
      }
    }

    cnt_fit <- cnt_line[fit_genes, , drop = FALSE]
    sf <- compute_size_factors(cnt_fit)

    cnt_gene <- cnt_line[genes, , drop = FALSE]

    sf_sum_abn <- sum(sf[mask_abn])
    sf_sum_norm <- sum(sf[mask_norm])

    if (!is.finite(sf_sum_abn) || !is.finite(sf_sum_norm) || sf_sum_abn <= 0 || sf_sum_norm <= 0) {
      out[[idx_out]] <- tibble::tibble(
        dataset = ds$name,
        cell_line = cl,
        gene_id = genes,
        fold_change = NA_real_,
        fold_change_sd = NA_real_,
        n_slides = 0L
      )
      idx_out <- idx_out + 1L
      next
    }

    mean_abn <- as.numeric(Matrix::rowSums(cnt_gene[, mask_abn, drop = FALSE]) / sf_sum_abn)
    mean_norm <- as.numeric(Matrix::rowSums(cnt_gene[, mask_norm, drop = FALSE]) / sf_sum_norm)
    fc <- (mean_abn + eps) / (mean_norm + eps)

    slide_ratio <- dplyr::bind_rows(lapply(sort(unique(slide_line)), function(sl) {
      idx_sl <- which(slide_line == sl)
      idx_abn <- idx_sl[cond_line[idx_sl] == "Abnormal"]
      idx_norm <- idx_sl[cond_line[idx_sl] == "Normal"]
      if (length(idx_abn) == 0 || length(idx_norm) == 0) return(NULL)

      sf_sum_abn_sl <- sum(sf[idx_abn])
      sf_sum_norm_sl <- sum(sf[idx_norm])
      if (!is.finite(sf_sum_abn_sl) || !is.finite(sf_sum_norm_sl) || sf_sum_abn_sl <= 0 || sf_sum_norm_sl <= 0) {
        return(NULL)
      }

      mean_abn_sl <- as.numeric(Matrix::rowSums(cnt_gene[, idx_abn, drop = FALSE]) / sf_sum_abn_sl)
      mean_norm_sl <- as.numeric(Matrix::rowSums(cnt_gene[, idx_norm, drop = FALSE]) / sf_sum_norm_sl)

      tibble::tibble(
        gene_id = genes,
        slide = sl,
        slide_fc = (mean_abn_sl + eps) / (mean_norm_sl + eps)
      )
    }))

    if (nrow(slide_ratio) == 0) {
      sd_tbl <- tibble::tibble(gene_id = genes, fold_change_sd = 0, n_slides = 0L)
    } else {
      sd_tbl <- slide_ratio %>%
        dplyr::group_by(gene_id) %>%
        dplyr::summarise(
          fold_change_sd = ifelse(dplyr::n() > 1, stats::sd(slide_fc, na.rm = TRUE), 0),
          n_slides = dplyr::n(),
          .groups = "drop"
        )
    }

    out[[idx_out]] <- tibble::tibble(
      dataset = ds$name,
      cell_line = cl,
      gene_id = genes,
      fold_change = as.numeric(fc)
    ) %>%
      dplyr::left_join(sd_tbl, by = "gene_id")

    idx_out <- idx_out + 1L
  }

  dplyr::bind_rows(out)
}

build_plot_counts <- function(counts_summary) {
  if (nrow(counts_summary) == 0) {
    return(tibble::tibble(
      dataset = character(), cell_line = character(), condition = character(), gene_id = character(),
      mode = character(), mean = numeric(), median = numeric(), n_cells = integer()
    ))
  }

  dplyr::bind_rows(
    counts_summary %>%
      dplyr::transmute(
        dataset, cell_line, condition, gene_id, n_cells,
        mode = "Raw",
        mean = mean_raw,
        median = median_raw
      ),
    counts_summary %>%
      dplyr::transmute(
        dataset, cell_line, condition, gene_id, n_cells,
        mode = "Normalized",
        mean = mean_normalized,
        median = median_normalized
      )
  )
}

build_summary_table <- function(ds, deg_sel, counts_summary, fc_summary, gene_levels) {
  if (length(gene_levels) == 0) {
    return(tibble::tibble(
      gene_id = character(),
      cell_line = character(),
      fold_change = numeric(),
      fold_change_sd = numeric(),
      n_slides = integer(),
      log2fc_xenium_eps = numeric(),
      pvals_adj = numeric(),
      beta1 = numeric()
    ))
  }

  counts_wide <- if (nrow(counts_summary) > 0) {
    counts_summary %>%
      tidyr::pivot_wider(
        id_cols = c(dataset, cell_line, gene_id),
        names_from = condition,
        values_from = c(n_cells, mean_raw, median_raw, mean_normalized, median_normalized),
        names_glue = "{.value}_{condition}"
      )
  } else {
    tibble::tibble(dataset = character(), cell_line = character(), gene_id = character())
  }

  deg_compact <- deg_sel %>%
    dplyr::select(dataset, cell_line, gene_id, log2fc_xenium_eps, pvals_adj, beta1)

  fc_compact <- fc_summary %>%
    dplyr::select(dataset, cell_line, gene_id, fold_change, fold_change_sd, n_slides)

  full_index <- tidyr::expand_grid(
    dataset = ds$name,
    gene_id = gene_levels,
    cell_line = ds$cell_line_order
  )

  out <- full_index %>%
    dplyr::left_join(fc_compact, by = c("dataset", "cell_line", "gene_id")) %>%
    dplyr::left_join(deg_compact, by = c("dataset", "cell_line", "gene_id")) %>%
    dplyr::left_join(counts_wide, by = c("dataset", "cell_line", "gene_id")) %>%
    dplyr::mutate(
      gene_id = factor(gene_id, levels = gene_levels),
      cell_line = factor(cell_line, levels = ds$cell_line_order)
    ) %>%
    dplyr::arrange(gene_id, cell_line) %>%
    dplyr::mutate(
      gene_id = as.character(gene_id),
      cell_line = as.character(cell_line)
    ) %>%
    dplyr::select(-dataset)

  out
}

empty_result <- function() {
  list(
    genes = character(),
    unresolved = character(),
    mapping = tibble::tibble(requested_gene = character(), resolved_gene = character(), status = character()),
    deg = tibble::tibble(
      dataset = character(), cell_line = character(), gene_id = character(),
      log2fc_xenium_eps = numeric(), pvals_adj = numeric(), beta1 = numeric()
    ),
    fc_summary = tibble::tibble(
      dataset = character(), cell_line = character(), gene_id = character(),
      fold_change = numeric(), fold_change_sd = numeric(), n_slides = integer()
    ),
    counts_summary = tibble::tibble(),
    plot_counts = tibble::tibble(),
    table = tibble::tibble(gene_id = character(), cell_line = character())
  )
}

build_dataset_result <- function(ds, query_genes) {
  resolved <- resolve_genes(query_genes, ds$gene_universe)
  genes <- resolved$matched

  if (length(genes) == 0) {
    res <- empty_result()
    res$unresolved <- resolved$unresolved
    res$mapping <- resolved$mapping
    return(res)
  }

  deg_sel <- ds$deg_long %>%
    dplyr::filter(gene_id %in% genes)

  fc_summary <- compute_fc_summary(ds, genes)
  counts_summary <- compute_counts_summary(ds, genes)
  plot_counts <- build_plot_counts(counts_summary)
  summary_tbl <- build_summary_table(ds, deg_sel, counts_summary, fc_summary, genes)

  list(
    genes = genes,
    unresolved = resolved$unresolved,
    mapping = resolved$mapping,
    deg = deg_sel,
    fc_summary = fc_summary,
    counts_summary = counts_summary,
    plot_counts = plot_counts,
    table = summary_tbl
  )
}

mode_filter <- function(plot_df, count_mode) {
  if (count_mode == "raw") {
    return(plot_df %>% dplyr::filter(mode == "Raw"))
  }
  if (count_mode == "normalized") {
    return(plot_df %>% dplyr::filter(mode == "Normalized"))
  }
  plot_df
}

empty_plot <- function(msg) {
  ggplot2::ggplot() +
    ggplot2::theme_void() +
    ggplot2::xlim(-1, 1) +
    ggplot2::ylim(-1, 1) +
    ggplot2::annotate("text", x = 0, y = 0, label = msg, size = 5)
}

plot_fold_change <- function(dataset_title, fc_df, deg_df, gene_levels, cell_line_order,
                             padj_thr = 0.05, lfc_thr = 0.25) {
  if (nrow(fc_df) == 0) {
    return(empty_plot("No fold-change rows for selected genes."))
  }

  deg_compact <- deg_df %>%
    dplyr::select(cell_line, gene_id, log2fc_xenium_eps, pvals_adj) %>%
    dplyr::distinct(cell_line, gene_id, .keep_all = TRUE)

  df <- fc_df %>%
    dplyr::left_join(deg_compact, by = c("cell_line", "gene_id")) %>%
    dplyr::mutate(
      de_status = dplyr::case_when(
        !is.na(pvals_adj) & pvals_adj < padj_thr & !is.na(log2fc_xenium_eps) & log2fc_xenium_eps >= lfc_thr ~ "Up in abnormal",
        !is.na(pvals_adj) & pvals_adj < padj_thr & !is.na(log2fc_xenium_eps) & log2fc_xenium_eps <= -lfc_thr ~ "Down in abnormal",
        TRUE ~ "Non-significant"
      )
    ) %>%
    dplyr::mutate(
      gene_id = factor(gene_id, levels = gene_levels),
      cell_line = factor(cell_line, levels = cell_line_order),
      de_status = factor(
        de_status,
        levels = c("Up in abnormal", "Down in abnormal", "Non-significant")
      )
    )

  ncols <- min(4, max(1, length(gene_levels)))

  ggplot(df, aes(x = cell_line, y = fold_change, fill = de_status)) +
    geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
    geom_col(width = 0.72, color = "black", alpha = 0.85) +
    geom_errorbar(
      data = df %>% dplyr::filter(!is.na(fold_change), !is.na(fold_change_sd)),
      aes(ymin = pmax(0, fold_change - fold_change_sd), ymax = fold_change + fold_change_sd),
      width = 0.15,
      linewidth = 0.45
    ) +
    facet_wrap(~gene_id, scales = "free_y", ncol = ncols) +
    scale_fill_manual(
      values = c(
        "Up in abnormal" = "#D46A6A",
        "Down in abnormal" = "#586BA4",
        "Non-significant" = "#BDBDBD"
      ),
      drop = FALSE
    ) +
    labs(
      title = paste0(dataset_title, " | Fold change (abnormal / normal)"),
      x = "Cell line",
      y = "Fold change",
      fill = "DE status"
    ) +
    theme_bw(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_rect(fill = "grey95", color = "black")
    )
}

plot_count_metric <- function(dataset_title, plot_counts_df, metric_name, count_mode, gene_levels, cell_line_order) {
  df <- mode_filter(plot_counts_df, count_mode)

  if (nrow(df) == 0) {
    return(empty_plot("No count summary rows for selected genes."))
  }

  df <- df %>%
    dplyr::mutate(
      gene_id = factor(gene_id, levels = gene_levels),
      cell_line = factor(cell_line, levels = cell_line_order),
      mode = factor(mode, levels = c("Raw", "Normalized")),
      value = if (metric_name == "median") median else mean
    )

  metric_label <- if (metric_name == "median") "Median" else "Mean"

  ggplot(df, aes(x = cell_line, y = value, fill = condition)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.72, color = "black", alpha = 0.85) +
    facet_grid(mode ~ gene_id, scales = "free_y", drop = FALSE) +
    scale_fill_manual(values = c("Normal" = "#4C78A8", "Abnormal" = "#E45756")) +
    labs(
      title = paste0(dataset_title, " | ", metric_label, " counts by condition"),
      x = "Cell line",
      y = paste0(metric_label, " counts"),
      fill = "Condition"
    ) +
    theme_bw(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_rect(fill = "grey95", color = "black")
    )
}

table_for_mode <- function(tbl, count_mode) {
  base_cols <- c(
    "gene_id", "cell_line", "fold_change", "fold_change_sd", "n_slides",
    "pvals_adj", "log2fc_xenium_eps", "beta1",
    "n_cells_Normal", "n_cells_Abnormal"
  )
  raw_cols <- c("mean_raw_Normal", "mean_raw_Abnormal", "median_raw_Normal", "median_raw_Abnormal")
  norm_cols <- c(
    "mean_normalized_Normal", "mean_normalized_Abnormal",
    "median_normalized_Normal", "median_normalized_Abnormal"
  )

  keep <- switch(
    count_mode,
    raw = c(base_cols, raw_cols),
    normalized = c(base_cols, norm_cols),
    both = c(base_cols, raw_cols, norm_cols)
  )

  keep <- keep[keep %in% colnames(tbl)]
  out <- tbl[, keep, drop = FALSE]

  num_cols <- names(out)[vapply(out, is.numeric, logical(1))]
  for (nm in num_cols) {
    out[[nm]] <- signif(out[[nm]], 4)
  }

  out
}

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))

required_args <- c("myonuclei_rds", "myotube_rds", "myonuclei_deg_dir", "myotube_deg_dir")
missing_args <- required_args[!required_args %in% names(args)]
if (length(missing_args) > 0) {
  stop(
    "Missing required arguments: ", paste(missing_args, collapse = ", "), "\n",
    "Required: --myonuclei_rds --myotube_rds --myonuclei_deg_dir --myotube_deg_dir"
  )
}

host <- if (!is.null(args$host)) as.character(args$host) else "127.0.0.1"
port <- if (!is.null(args$port)) suppressWarnings(as.integer(args$port)) else 3838L

if (is.na(port) || !is.finite(port) || port <= 0) {
  stop("Invalid --port value: ", args$port)
}

if (!file.exists(args$myonuclei_rds)) stop("myonuclei_rds not found: ", args$myonuclei_rds)
if (!file.exists(args$myotube_rds)) stop("myotube_rds not found: ", args$myotube_rds)
if (!dir.exists(args$myonuclei_deg_dir)) stop("myonuclei_deg_dir not found: ", args$myonuclei_deg_dir)
if (!dir.exists(args$myotube_deg_dir)) stop("myotube_deg_dir not found: ", args$myotube_deg_dir)

message("Loading RDS objects...")
myonuclei_obj <- readRDS(args$myonuclei_rds)
myotube_obj <- readRDS(args$myotube_rds)

message("Loading DEG outputs...")
myonuclei_deg <- load_deg_results(args$myonuclei_deg_dir, "myonuclei")
myotube_deg <- load_deg_results(args$myotube_deg_dir, "myotube")

message("Preparing datasets...")
myonuclei_data <- prepare_dataset("myonuclei", myonuclei_obj, myonuclei_deg)
myotube_data <- prepare_dataset("myotube", myotube_obj, myotube_deg)

ui <- fluidPage(
  titlePanel("CosMx Gene Search and Visualisation"),
  sidebarLayout(
    sidebarPanel(
      textInput(
        "gene_query",
        "Genes (comma-separated)",
        value = "FN1"
      ),
      selectInput(
        "count_mode",
        "Count summary mode",
        choices = c("Raw" = "raw", "Normalized" = "normalized", "Both" = "both"),
        selected = "both"
      ),
      actionButton("run_query", "Generate plots"),
      tags$hr(),
      helpText("Fold-change bars are computed from within-slide abnormal/normal ratios and show SD across slides."),
      helpText("Adjusted p-values (pvals_adj) are shown in the summary table."),
      helpText("Count summaries are computed separately for Normal and Abnormal groups in each cell line.")
    ),
    mainPanel(
      uiOutput("global_warning"),
      tabsetPanel(
        tabPanel(
          "Myonuclei",
          uiOutput("warn_myonuclei"),
          plotOutput("myonuclei_fc", height = "480px"),
          plotOutput("myonuclei_median", height = "520px"),
          plotOutput("myonuclei_mean", height = "520px"),
          DTOutput("myonuclei_table")
        ),
        tabPanel(
          "Myotubes",
          uiOutput("warn_myotube"),
          plotOutput("myotube_fc", height = "480px"),
          plotOutput("myotube_median", height = "520px"),
          plotOutput("myotube_mean", height = "520px"),
          DTOutput("myotube_table")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  query_state <- eventReactive(input$run_query, {
    requested_genes <- parse_gene_query(input$gene_query)

    if (length(requested_genes) == 0) {
      return(list(
        error = "Please enter at least one gene.",
        requested = character(),
        myonuclei = empty_result(),
        myotube = empty_result()
      ))
    }

    list(
      error = NULL,
      requested = requested_genes,
      myonuclei = build_dataset_result(myonuclei_data, requested_genes),
      myotube = build_dataset_result(myotube_data, requested_genes)
    )
  }, ignoreNULL = FALSE)

  output$global_warning <- renderUI({
    qs <- query_state()

    if (!is.null(qs$error)) {
      return(tags$div(style = "color:#a94442;font-weight:700;margin-bottom:10px;", qs$error))
    }

    msgs <- character()
    if (length(qs$myonuclei$unresolved) > 0) {
      msgs <- c(msgs, paste0("Myonuclei unresolved: ", paste(qs$myonuclei$unresolved, collapse = ", ")))
    }
    if (length(qs$myotube$unresolved) > 0) {
      msgs <- c(msgs, paste0("Myotubes unresolved: ", paste(qs$myotube$unresolved, collapse = ", ")))
    }

    if (length(msgs) == 0) return(NULL)

    tags$div(
      style = "color:#8a6d3b;background:#fcf8e3;border:1px solid #faebcc;padding:8px;margin-bottom:10px;",
      lapply(msgs, function(m) tags$div(m))
    )
  })

  output$warn_myonuclei <- renderUI({
    qs <- query_state()
    if (is.null(qs$error) && length(qs$myonuclei$genes) == 0) {
      return(tags$div(style = "color:#a94442;font-weight:600;", "No resolvable genes found in myonuclei dataset."))
    }
    NULL
  })

  output$warn_myotube <- renderUI({
    qs <- query_state()
    if (is.null(qs$error) && length(qs$myotube$genes) == 0) {
      return(tags$div(style = "color:#a94442;font-weight:600;", "No resolvable genes found in myotube dataset."))
    }
    NULL
  })

  output$myonuclei_fc <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_fold_change(
      dataset_title = "Myonuclei",
      fc_df = qs$myonuclei$fc_summary,
      deg_df = qs$myonuclei$deg,
      gene_levels = qs$myonuclei$genes,
      cell_line_order = myonuclei_data$cell_line_order
    )
  }, res = 120)

  output$myonuclei_median <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_count_metric(
      dataset_title = "Myonuclei",
      plot_counts_df = qs$myonuclei$plot_counts,
      metric_name = "median",
      count_mode = input$count_mode,
      gene_levels = qs$myonuclei$genes,
      cell_line_order = myonuclei_data$cell_line_order
    )
  }, res = 120)

  output$myonuclei_mean <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_count_metric(
      dataset_title = "Myonuclei",
      plot_counts_df = qs$myonuclei$plot_counts,
      metric_name = "mean",
      count_mode = input$count_mode,
      gene_levels = qs$myonuclei$genes,
      cell_line_order = myonuclei_data$cell_line_order
    )
  }, res = 120)

  output$myonuclei_table <- renderDT({
    qs <- query_state()
    tbl <- table_for_mode(qs$myonuclei$table, input$count_mode)

    datatable(
      tbl,
      rownames = FALSE,
      options = list(pageLength = 12, scrollX = TRUE)
    )
  })

  output$myotube_fc <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_fold_change(
      dataset_title = "Myotubes",
      fc_df = qs$myotube$fc_summary,
      deg_df = qs$myotube$deg,
      gene_levels = qs$myotube$genes,
      cell_line_order = myotube_data$cell_line_order
    )
  }, res = 120)

  output$myotube_median <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_count_metric(
      dataset_title = "Myotubes",
      plot_counts_df = qs$myotube$plot_counts,
      metric_name = "median",
      count_mode = input$count_mode,
      gene_levels = qs$myotube$genes,
      cell_line_order = myotube_data$cell_line_order
    )
  }, res = 120)

  output$myotube_mean <- renderPlot({
    qs <- query_state()
    if (!is.null(qs$error)) return(empty_plot(qs$error))

    plot_count_metric(
      dataset_title = "Myotubes",
      plot_counts_df = qs$myotube$plot_counts,
      metric_name = "mean",
      count_mode = input$count_mode,
      gene_levels = qs$myotube$genes,
      cell_line_order = myotube_data$cell_line_order
    )
  }, res = 120)

  output$myotube_table <- renderDT({
    qs <- query_state()
    tbl <- table_for_mode(qs$myotube$table, input$count_mode)

    datatable(
      tbl,
      rownames = FALSE,
      options = list(pageLength = 12, scrollX = TRUE)
    )
  })
}

app <- shinyApp(ui = ui, server = server)

message("Shiny server starting at: http://", host, ":", port)
message("Press Ctrl+C in this terminal to stop the server.")

shiny::runApp(app, host = host, port = port, launch.browser = FALSE)
