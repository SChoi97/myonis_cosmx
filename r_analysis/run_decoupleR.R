#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(decoupleR)
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
MYONUCLEI_RDS_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/processed_myonuclei.rds"
MYOTUBE_RDS_PATH <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/processed_myotube_filtered.rds"
OUTPUT_DIR <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/decoupleR"

MIN_COUNTS <- 100L
MIN_CELLS <- 100L
DROP_PATTERN <- "SystemControl|Negative"

PATHWAY_ORGANISM <- "human"
PATHWAY_TOP <- 500L
PATHWAY_MINSIZE <- 5L

TF_ORGANISM <- "human"
TF_SPLIT_COMPLEXES <- FALSE
TF_MINSIZE <- 5L

RUN_DIMRED <- TRUE
MAX_PCS <- 10L
SEED <- 0L

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Supports:
#   --KEY value
#   --key=value
#   key=value
# Usage example:
# Rscript run_decoupleR.R \
#   --MYONUCLEI_RDS_PATH /path/processed_myonuclei.rds \
#   --MYOTUBE_RDS_PATH /path/processed_myotube_filtered.rds \
#   --OUTPUT_DIR /path/decoupleR \
#   --MIN_COUNTS 100 \
#   --MIN_CELLS 100 \
#   --DROP_PATTERN 'SystemControl|Negative' \
#   --RUN_DIMRED TRUE
overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
drop_pattern_set_null <- FALSE
if ("DROP_PATTERN" %in% names(overrides)) {
  raw_drop <- tolower(trimws(overrides[["DROP_PATTERN"]]))
  if (raw_drop %in% c("null", "none")) {
    drop_pattern_set_null <- TRUE
  }
}
apply_cli_overrides(overrides)
if (drop_pattern_set_null) {
  DROP_PATTERN <- NULL
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
stop_if_missing_file(MYONUCLEI_RDS_PATH, "MYONUCLEI_RDS_PATH")
stop_if_missing_file(MYOTUBE_RDS_PATH, "MYOTUBE_RDS_PATH")

set.seed(SEED)

get_assay_data <- function(obj, assay, layer) {
  tryCatch(
    SeuratObject::GetAssayData(obj, assay = assay, layer = layer),
    error = function(e) {
      slot_name <- if (identical(layer, "scale.data")) "scale.data" else layer
      SeuratObject::GetAssayData(obj, assay = assay, slot = slot_name)
    }
  )
}

set_assay_data <- function(obj, assay, layer, new_data) {
  tryCatch(
    SeuratObject::SetAssayData(
      object = obj,
      assay = assay,
      layer = layer,
      new.data = new_data
    ),
    error = function(e) {
      slot_name <- if (identical(layer, "scale.data")) "scale.data" else layer
      SeuratObject::SetAssayData(
        object = obj,
        assay = assay,
        slot = slot_name,
        new.data = new_data
      )
    }
  )
}

has_assay_layer <- function(obj, assay, layer) {
  ok <- TRUE
  tryCatch(
    get_assay_data(obj, assay = assay, layer = layer),
    error = function(e) {
      ok <<- FALSE
      NULL
    }
  )
  ok
}

coerce_dense_numeric_matrix <- function(x, label = "matrix") {
  if (inherits(x, "table")) {
    dims <- dim(x)
    x <- matrix(
      data = as.numeric(x),
      nrow = dims[1],
      ncol = dims[2],
      dimnames = dimnames(x)
    )
  } else if (inherits(x, "data.frame")) {
    x <- as.matrix(x)
  } else if (inherits(x, "Matrix")) {
    x <- as.matrix(x)
  } else if (is.array(x) && length(dim(x)) == 2L && !is.matrix(x)) {
    dims <- dim(x)
    x <- matrix(
      data = as.numeric(x),
      nrow = dims[1],
      ncol = dims[2],
      dimnames = dimnames(x)
    )
  }

  if (!is.matrix(x)) {
    stop("Expected a 2D matrix-like object for ", label, ", got: ", paste(class(x), collapse = ", "))
  }

  storage.mode(x) <- "double"
  x
}

coerce_sparse_numeric_matrix <- function(x, label = "matrix") {
  if (inherits(x, "dgCMatrix")) {
    return(x)
  }

  if (inherits(x, "Matrix")) {
    return(methods::as(x, "dgCMatrix"))
  }

  x <- coerce_dense_numeric_matrix(x, label = label)
  methods::as(Matrix::Matrix(x, sparse = TRUE), "dgCMatrix")
}

create_activity_assay <- function(data_mat) {
  data_sparse <- coerce_sparse_numeric_matrix(data_mat, label = "activity assay")
  if ("CreateAssay5Object" %in% getNamespaceExports("SeuratObject")) {
    SeuratObject::CreateAssay5Object(data = data_sparse)
  } else {
    SeuratObject::CreateAssayObject(data = data_sparse)
  }
}

resolve_base_assay_name <- function(seu, preferred = c("originalexp", "RNA")) {
  assay_names <- names(seu@assays)
  default_assay <- tryCatch(DefaultAssay(seu), error = function(e) NA_character_)
  candidates <- unique(c(preferred, default_assay, assay_names))

  for (assay_name in candidates) {
    if (is.na(assay_name) || !assay_name %in% assay_names) {
      next
    }
    if (has_assay_layer(seu, assay_name, "counts")) {
      return(assay_name)
    }
  }

  stop(
    "Could not find a Seurat assay with a counts layer. Available assays: ",
    paste(assay_names, collapse = ", ")
  )
}

convert_to_seurat <- function(obj, label) {
  if (inherits(obj, "Seurat")) {
    assay_name <- resolve_base_assay_name(obj)
    DefaultAssay(obj) <- assay_name
    return(list(seu = obj, base_assay = assay_name, input_class = class(obj)[1]))
  }

  if (inherits(obj, "SummarizedExperiment") && !inherits(obj, "SingleCellExperiment")) {
    obj <- as(obj, "SingleCellExperiment")
  }

  if (!inherits(obj, "SingleCellExperiment")) {
    stop(
      "[", label, "] Unsupported input class: ",
      paste(class(obj), collapse = ", "),
      ". Expected a Seurat or SingleCellExperiment-derived object."
    )
  }

  obj <- ensure_counts_assay(
    sce = obj,
    obj_name = label,
    candidates = c("X", "raw_counts", "raw", "matrix")
  )

  meta <- as.data.frame(SummarizedExperiment::colData(obj))
  rownames(meta) <- colnames(obj)

  seu <- as.Seurat(
    obj,
    counts = "counts",
    data = NULL,
    assay = NULL,
    project = label
  )
  seu@meta.data <- meta[colnames(seu), , drop = FALSE]

  assay_name <- resolve_base_assay_name(seu)
  DefaultAssay(seu) <- assay_name

  list(seu = seu, base_assay = assay_name, input_class = class(obj)[1])
}

scale_rows <- function(mat) {
  mat <- coerce_dense_numeric_matrix(mat, label = "scaled activity matrix")
  scaled <- t(scale(t(mat)))
  scaled[!is.finite(scaled)] <- 0
  rownames(scaled) <- rownames(mat)
  colnames(scaled) <- colnames(mat)
  scaled
}

prep_sf_seurat <- function(
  seu,
  assay_name,
  min_counts = MIN_COUNTS,
  min_cells = MIN_CELLS,
  drop_pattern = DROP_PATTERN
) {
  DefaultAssay(seu) <- assay_name

  counts <- get_assay_data(seu, assay = assay_name, layer = "counts")
  counts <- methods::as(counts, "dgCMatrix")

  keep_cells <- Matrix::colSums(counts) >= min_counts
  if (!any(keep_cells)) {
    stop("No cells passed MIN_COUNTS filtering for assay '", assay_name, "'.")
  }
  seu <- subset(seu, cells = colnames(counts)[keep_cells])

  counts <- get_assay_data(seu, assay = assay_name, layer = "counts")
  counts <- methods::as(counts, "dgCMatrix")

  keep_genes <- Matrix::rowSums(counts > 0) >= min_cells
  if (!is.null(drop_pattern) && nzchar(drop_pattern)) {
    keep_genes <- keep_genes & !grepl(drop_pattern, rownames(counts), ignore.case = TRUE)
  }
  if (!any(keep_genes)) {
    stop("No genes passed MIN_CELLS / DROP_PATTERN filtering for assay '", assay_name, "'.")
  }
  seu <- subset(seu, features = rownames(counts)[keep_genes])

  counts <- get_assay_data(seu, assay = assay_name, layer = "counts")
  counts <- methods::as(counts, "dgCMatrix")

  total_counts <- Matrix::colSums(counts)
  nonzero_totals <- total_counts[total_counts > 0]
  if (!length(nonzero_totals)) {
    stop("All cells have zero total counts after filtering for assay '", assay_name, "'.")
  }

  median_total <- stats::median(nonzero_totals)
  size_factors <- total_counts / median_total
  sf_pos <- size_factors[size_factors > 0]
  if (!length(sf_pos)) {
    stop("No positive size factors were computed for assay '", assay_name, "'.")
  }
  size_factors[size_factors <= 0] <- min(sf_pos) * 1e-3

  norm <- counts %*% Matrix::Diagonal(x = 1 / size_factors)
  rownames(norm) <- rownames(counts)
  colnames(norm) <- colnames(counts)
  norm@x <- log1p(norm@x)

  seu <- set_assay_data(seu, assay = assay_name, layer = "data", new_data = norm)
  seu$size_factor <- size_factors[colnames(seu)]
  seu@misc[[paste0(assay_name, "_median_total")]] <- median_total

  message(
    sprintf(
      "Prepared assay '%s': %d cells, %d genes after filtering",
      assay_name,
      ncol(seu),
      nrow(get_assay_data(seu, assay = assay_name, layer = "counts"))
    )
  )

  seu
}

activity_tbl_to_matrix <- function(acts, label, value_col = "score") {
  required_cols <- c("source", "condition", value_col)
  missing_cols <- setdiff(required_cols, colnames(acts))
  if (length(missing_cols) > 0) {
    stop(
      "[", label, "] Activity table is missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  acts_df <- as.data.frame(acts[, required_cols, drop = FALSE])
  acts_df <- acts_df[stats::complete.cases(acts_df), , drop = FALSE]
  if (nrow(acts_df) == 0) {
    stop("[", label, "] Activity table is empty after removing missing values.")
  }

  formula_str <- sprintf("%s ~ source + condition", value_col)
  coerce_dense_numeric_matrix(
    stats::xtabs(stats::as.formula(formula_str), data = acts_df),
    label = paste0(label, " activity table: ", value_col)
  )
}

extract_activity_result_matrices <- function(acts, label) {
  score_mat <- activity_tbl_to_matrix(acts, label = label, value_col = "score")

  pvalue_mat <- if ("p_value" %in% colnames(acts)) {
    align_activity_matrix_to_template(
      activity_tbl_to_matrix(acts, label = label, value_col = "p_value"),
      template_mat = score_mat,
      fill = NaN
    )
  } else {
    warning("[", label, "] decoupleR output is missing 'p_value'; exporting NaN p-values.")
    matrix(
      NaN,
      nrow = nrow(score_mat),
      ncol = ncol(score_mat),
      dimnames = dimnames(score_mat)
    )
  }

  zscore_mat <- scale_rows(score_mat)

  list(
    score = score_mat,
    zscore = zscore_mat,
    pvalue = pvalue_mat
  )
}

add_activity_assay <- function(seu, activity_mat, assay_name, ident_col = "Cell.Line") {
  activity_mat <- coerce_dense_numeric_matrix(activity_mat, label = assay_name)
  common_cells <- intersect(colnames(seu), colnames(activity_mat))
  if (!length(common_cells)) {
    stop("No overlapping cells between Seurat object and assay '", assay_name, "' activity matrix.")
  }

  seu <- subset(seu, cells = common_cells)
  activity_mat <- activity_mat[, colnames(seu), drop = FALSE]
  activity_scaled <- scale_rows(activity_mat)

  seu[[assay_name]] <- create_activity_assay(activity_mat)
  seu <- set_assay_data(seu, assay = assay_name, layer = "scale.data", new_data = activity_scaled)
  DefaultAssay(seu) <- assay_name

  if (ident_col %in% colnames(seu[[]])) {
    Idents(seu) <- ident_col
  }

  seu
}

align_activity_matrix_to_template <- function(activity_mat, template_mat, fill = NaN) {
  activity_mat <- coerce_dense_numeric_matrix(activity_mat, label = "activity template alignment")
  template_mat <- coerce_dense_numeric_matrix(template_mat, label = "activity template")

  aligned <- matrix(
    fill,
    nrow = nrow(template_mat),
    ncol = ncol(template_mat),
    dimnames = dimnames(template_mat)
  )

  common_rows <- intersect(rownames(template_mat), rownames(activity_mat))
  common_cols <- intersect(colnames(template_mat), colnames(activity_mat))
  if (length(common_rows) && length(common_cols)) {
    aligned[common_rows, common_cols] <- activity_mat[common_rows, common_cols, drop = FALSE]
  }

  aligned
}

build_csv_output_paths <- function(label) {
  stem <- if (identical(label, "myotube")) "myotubes" else label
  list(
    ulm = file.path(OUTPUT_DIR, paste0(stem, "_ulm.csv")),
    ulm_zscore = file.path(OUTPUT_DIR, paste0(stem, "_ulm_zscore.csv")),
    ulm_pvalue = file.path(OUTPUT_DIR, paste0(stem, "_ulm_pvalue.csv")),
    mlm = file.path(OUTPUT_DIR, paste0(stem, "_mlm.csv")),
    mlm_zscore = file.path(OUTPUT_DIR, paste0(stem, "_mlm_zscore.csv")),
    mlm_pvalue = file.path(OUTPUT_DIR, paste0(stem, "_mlm_pvalue.csv"))
  )
}

build_full_cell_metadata <- function(seu, assay_name, min_counts = MIN_COUNTS) {
  all_cells <- colnames(seu)
  meta <- seu[[]]
  meta <- meta[all_cells, , drop = FALSE]
  rownames(meta) <- all_cells

  counts <- get_assay_data(seu, assay = assay_name, layer = "counts")
  counts <- methods::as(counts, "dgCMatrix")
  total_counts <- Matrix::colSums(counts)
  total_counts <- as.numeric(total_counts[all_cells])

  meta$decoupleR_cell_id <- all_cells
  meta$decoupleR_input_order <- seq_along(all_cells)
  meta$decoupleR_total_counts <- total_counts
  meta$decoupleR_passed_min_counts <- total_counts >= as.integer(min_counts)

  meta
}

align_activity_matrix_to_cells <- function(activity_mat, all_cells, fill = NaN) {
  activity_mat <- coerce_dense_numeric_matrix(activity_mat, label = "aligned activity matrix")
  aligned <- matrix(
    fill,
    nrow = nrow(activity_mat),
    ncol = length(all_cells),
    dimnames = list(rownames(activity_mat), all_cells)
  )
  common_cells <- intersect(all_cells, colnames(activity_mat))
  if (length(common_cells)) {
    aligned[, common_cells] <- activity_mat[, common_cells, drop = FALSE]
  }
  aligned
}

align_embedding_matrix_to_cells <- function(embedding_mat, all_cells, fill = NaN) {
  embedding_mat <- coerce_dense_numeric_matrix(embedding_mat, label = "aligned embedding matrix")
  aligned <- matrix(
    fill,
    nrow = length(all_cells),
    ncol = ncol(embedding_mat),
    dimnames = list(all_cells, colnames(embedding_mat))
  )
  common_cells <- intersect(all_cells, rownames(embedding_mat))
  if (length(common_cells)) {
    aligned[common_cells, ] <- embedding_mat[common_cells, , drop = FALSE]
  }
  aligned
}

extract_aligned_reduction <- function(seu, reduction_name, all_cells, prefix) {
  if (!(reduction_name %in% names(seu@reductions))) {
    return(NULL)
  }

  emb <- coerce_dense_numeric_matrix(
    Embeddings(seu, reduction_name),
    label = paste0(reduction_name, " embeddings")
  )
  colnames(emb) <- paste0(prefix, seq_len(ncol(emb)))
  align_embedding_matrix_to_cells(emb, all_cells = all_cells, fill = NaN)
}

matrix_rows_to_numeric_df <- function(mat, label = "matrix") {
  mat <- coerce_dense_numeric_matrix(mat, label = label)
  out <- as.data.frame(mat, check.names = FALSE, stringsAsFactors = FALSE)
  rownames(out) <- rownames(mat)
  out
}

matrix_rows_to_csv_df <- function(mat, label = "matrix") {
  mat <- coerce_dense_numeric_matrix(mat, label = label)
  cols <- lapply(seq_len(ncol(mat)), function(j) {
    values <- as.character(mat[, j])
    values[is.na(mat[, j])] <- "NaN"
    values
  })
  out <- as.data.frame(cols, check.names = FALSE, stringsAsFactors = FALSE)
  colnames(out) <- colnames(mat)
  rownames(out) <- rownames(mat)
  out
}

build_activity_export_df <- function(
  meta_df,
  score_mat,
  available_cells,
  availability_col,
  pca_mat = NULL,
  umap_mat = NULL
) {
  meta_df <- meta_df[colnames(score_mat), , drop = FALSE]
  meta_df[[availability_col]] <- rownames(meta_df) %in% available_cells

  score_cells <- t(coerce_dense_numeric_matrix(score_mat, label = availability_col))
  rownames(score_cells) <- rownames(meta_df)

  blocks <- list(meta_df)

  if (!is.null(pca_mat)) {
    pca_mat <- pca_mat[rownames(meta_df), , drop = FALSE]
    blocks[[length(blocks) + 1L]] <- matrix_rows_to_csv_df(
      pca_mat,
      label = paste0(availability_col, " pca export")
    )
  }

  if (!is.null(umap_mat)) {
    umap_mat <- umap_mat[rownames(meta_df), , drop = FALSE]
    blocks[[length(blocks) + 1L]] <- matrix_rows_to_csv_df(
      umap_mat,
      label = paste0(availability_col, " umap export")
    )
  }

  blocks[[length(blocks) + 1L]] <- matrix_rows_to_csv_df(
    score_cells,
    label = paste0(availability_col, " score export")
  )

  out <- do.call(cbind, blocks)
  out$decoupleR_export_row <- seq_len(nrow(out))
  out
}

write_activity_csv <- function(export_df, output_path) {
  utils::write.csv(export_df, file = output_path, row.names = FALSE, na = "")
}

run_activity_dimred <- function(
  obj,
  assay_name,
  pca_name,
  umap_name,
  umap_key,
  max_pcs = MAX_PCS,
  seed = SEED
) {
  DefaultAssay(obj) <- assay_name
  features_use <- rownames(obj[[assay_name]])

  if (length(features_use) < 2L || ncol(obj) < 2L) {
    stop("Need at least 2 features and 2 cells to run PCA/UMAP for assay '", assay_name, "'.")
  }

  max_rank <- min(length(features_use), ncol(obj) - 1L)
  if (max_rank < 2L) {
    stop("Need at least 2 usable PCs for assay '", assay_name, "'.")
  }

  npcs_use <- min(as.integer(max_pcs), max_rank)

  obj <- RunPCA(
    object = obj,
    assay = assay_name,
    features = features_use,
    npcs = npcs_use,
    reduction.name = pca_name,
    verbose = FALSE
  )

  dims_use <- seq_len(min(as.integer(max_pcs), ncol(Embeddings(obj, pca_name))))
  if (length(dims_use) < 2L) {
    stop("Need at least 2 PCA dimensions to run UMAP for assay '", assay_name, "'.")
  }

  obj <- RunUMAP(
    object = obj,
    reduction = pca_name,
    dims = dims_use,
    reduction.name = umap_name,
    reduction.key = umap_key,
    seed.use = as.integer(seed),
    verbose = FALSE
  )

  obj
}

load_pathway_net <- function() {
  message("Loading PROGENy network...")
  tryCatch(
    decoupleR::get_progeny(
      organism = PATHWAY_ORGANISM,
      top = as.integer(PATHWAY_TOP)
    ),
    error = function(e) {
      stop(
        "Failed to load PROGENy network with decoupleR::get_progeny(). ",
        "This usually requires OmniPath access or a valid local cache. Original error: ",
        conditionMessage(e)
      )
    }
  )
}

load_tf_net <- function() {
  message("Loading CollecTRI network...")
  tryCatch(
    decoupleR::get_collectri(
      organism = TF_ORGANISM,
      split_complexes = TF_SPLIT_COMPLEXES
    ),
    error = function(e) {
      stop(
        "Failed to load CollecTRI network with decoupleR::get_collectri(). ",
        "This usually requires OmniPath access or a valid local cache. Original error: ",
        conditionMessage(e)
      )
    }
  )
}

build_output_path <- function(input_path) {
  stem <- tools::file_path_sans_ext(basename(input_path))
  file.path(OUTPUT_DIR, paste0(stem, "_decoupleR.rds"))
}

attach_provenance <- function(
  seu,
  label,
  input_path,
  input_class,
  base_assay,
  filtered_cell_ids,
  filtered_feature_count,
  pathway_scores,
  pathway_zscores,
  pathway_pvalues,
  tf_scores,
  tf_zscores,
  tf_pvalues,
  csv_paths,
  aligned_reductions = list()
) {
  csv_paths <- Filter(Negate(is.null), csv_paths)
  csv_paths <- lapply(csv_paths, normalizePath, winslash = "/", mustWork = FALSE)

  aligned_reductions <- Filter(Negate(is.null), aligned_reductions)
  reduction_names <- names(aligned_reductions)
  aligned_reductions <- lapply(reduction_names, function(name) {
    matrix_rows_to_numeric_df(
      aligned_reductions[[name]],
      label = paste0(name, " aligned reduction")
    )
  })
  names(aligned_reductions) <- reduction_names

  decouple_misc <- list(
    label = label,
    input_path = normalizePath(input_path),
    input_class = input_class,
    base_assay = base_assay,
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    filtering = list(
      min_counts = as.integer(MIN_COUNTS),
      min_cells = as.integer(MIN_CELLS),
      drop_pattern = DROP_PATTERN,
      cells_in_input = ncol(seu),
      cells_used_for_activity = length(filtered_cell_ids),
      cells_without_activity = ncol(seu) - length(filtered_cell_ids)
    ),
    pathway = list(
      method = "MLM",
      score_storage = "misc$decoupleR$aligned_scores$pathwaysmlm",
      zscore_storage = "misc$decoupleR$aligned_zscores$pathwaysmlm",
      pvalue_storage = "misc$decoupleR$aligned_pvalues$pathwaysmlm",
      score_csv = csv_paths[["mlm"]],
      zscore_csv = csv_paths[["mlm_zscore"]],
      pvalue_csv = csv_paths[["mlm_pvalue"]],
      organism = PATHWAY_ORGANISM,
      top = as.integer(PATHWAY_TOP),
      minsize = as.integer(PATHWAY_MINSIZE),
      pca = if ("pathwaysmlm_pca" %in% names(aligned_reductions)) "pathwaysmlm_pca" else NULL,
      umap = if ("pathwaysmlm_umap" %in% names(aligned_reductions)) "pathwaysmlm_umap" else NULL
    ),
    tf = list(
      method = "ULM",
      score_storage = "misc$decoupleR$aligned_scores$tfsulm",
      zscore_storage = "misc$decoupleR$aligned_zscores$tfsulm",
      pvalue_storage = "misc$decoupleR$aligned_pvalues$tfsulm",
      score_csv = csv_paths[["ulm"]],
      zscore_csv = csv_paths[["ulm_zscore"]],
      pvalue_csv = csv_paths[["ulm_pvalue"]],
      organism = TF_ORGANISM,
      split_complexes = TF_SPLIT_COMPLEXES,
      minsize = as.integer(TF_MINSIZE),
      pca = if ("tfsulm_pca" %in% names(aligned_reductions)) "tfsulm_pca" else NULL,
      umap = if ("tfsulm_umap" %in% names(aligned_reductions)) "tfsulm_umap" else NULL
    ),
    dimensions = list(
      cells = ncol(seu),
      base_features = nrow(get_assay_data(seu, assay = base_assay, layer = "counts")),
      base_features_used_for_activity = filtered_feature_count,
      pathways = nrow(pathway_scores),
      tfs = nrow(tf_scores)
    ),
    csv_exports = csv_paths,
    filtered_cell_ids = filtered_cell_ids,
    aligned_scores = list(
      pathwaysmlm = matrix_rows_to_numeric_df(
        t(pathway_scores),
        label = "pathwaysmlm aligned scores"
      ),
      pathwaysmlm_zscore = matrix_rows_to_numeric_df(
        t(pathway_zscores),
        label = "pathwaysmlm aligned z-scores"
      ),
      pathwaysmlm_pvalue = matrix_rows_to_numeric_df(
        t(pathway_pvalues),
        label = "pathwaysmlm aligned p-values"
      ),
      tfsulm = matrix_rows_to_numeric_df(
        t(tf_scores),
        label = "tfsulm aligned scores"
      ),
      tfsulm_zscore = matrix_rows_to_numeric_df(
        t(tf_zscores),
        label = "tfsulm aligned z-scores"
      ),
      tfsulm_pvalue = matrix_rows_to_numeric_df(
        t(tf_pvalues),
        label = "tfsulm aligned p-values"
      )
    ),
    aligned_zscores = list(
      pathwaysmlm = matrix_rows_to_numeric_df(
        t(pathway_zscores),
        label = "pathwaysmlm aligned z-scores"
      ),
      tfsulm = matrix_rows_to_numeric_df(
        t(tf_zscores),
        label = "tfsulm aligned z-scores"
      )
    ),
    aligned_pvalues = list(
      pathwaysmlm = matrix_rows_to_numeric_df(
        t(pathway_pvalues),
        label = "pathwaysmlm aligned p-values"
      ),
      tfsulm = matrix_rows_to_numeric_df(
        t(tf_pvalues),
        label = "tfsulm aligned p-values"
      )
    ),
    aligned_reductions = aligned_reductions,
    seed = as.integer(SEED)
  )

  seu@misc$decoupleR <- decouple_misc
  seu
}

process_object <- function(input_path, label, pathway_net, tf_net) {
  message(sprintf("[%s] Reading input: %s", label, input_path))
  obj <- readRDS(input_path)
  converted <- convert_to_seurat(obj, label = label)
  rm(obj)
  gc(verbose = FALSE)

  seu <- converted$seu
  base_assay <- converted$base_assay
  all_cells <- colnames(seu)

  message(
    sprintf(
      "[%s] Loaded %s object; base assay='%s'; %d cells before filtering",
      label,
      converted$input_class,
      base_assay,
      ncol(seu)
    )
  )

  full_meta <- build_full_cell_metadata(
    seu = seu,
    assay_name = base_assay,
    min_counts = MIN_COUNTS
  )

  seu_work <- prep_sf_seurat(
    seu = seu,
    assay_name = base_assay,
    min_counts = MIN_COUNTS,
    min_cells = MIN_CELLS,
    drop_pattern = DROP_PATTERN
  )
  filtered_cells <- colnames(seu_work)

  size_factors_full <- rep(NaN, length(all_cells))
  names(size_factors_full) <- all_cells
  size_factors_full[filtered_cells] <- as.numeric(seu_work$size_factor)

  full_meta$decoupleR_included_in_activity <- all_cells %in% filtered_cells
  full_meta$decoupleR_excluded_reason <- ifelse(
    full_meta$decoupleR_included_in_activity,
    "",
    "filtered_by_min_counts"
  )
  full_meta$decoupleR_size_factor <- size_factors_full[all_cells]
  seu@meta.data <- full_meta[all_cells, , drop = FALSE]

  expr_mat <- coerce_dense_numeric_matrix(
    get_assay_data(seu_work, assay = base_assay, layer = "data"),
    label = paste0(label, " normalized expression")
  )
  message(sprintf("[%s] Running pathway MLM...", label))
  pathway_acts <- decoupleR::run_mlm(
    mat = expr_mat,
    network = pathway_net,
    .source = "source",
    .target = "target",
    .mor = "weight",
    minsize = as.integer(PATHWAY_MINSIZE)
  )

  message(sprintf("[%s] Running TF ULM...", label))
  tf_acts <- decoupleR::run_ulm(
    mat = expr_mat,
    network = tf_net,
    .source = "source",
    .target = "target",
    .mor = "mor",
    minsize = as.integer(TF_MINSIZE)
  )
  rm(expr_mat)
  gc(verbose = FALSE)

  pathway_res <- extract_activity_result_matrices(pathway_acts, label = "pathwaysmlm")
  tf_res <- extract_activity_result_matrices(tf_acts, label = "tfsulm")

  seu_work <- add_activity_assay(seu_work, activity_mat = pathway_res$score, assay_name = "pathwaysmlm")
  seu_work <- add_activity_assay(seu_work, activity_mat = tf_res$score, assay_name = "tfsulm")
  rm(pathway_acts, tf_acts)
  gc(verbose = FALSE)

  pathway_scores_full <- align_activity_matrix_to_cells(pathway_res$score, all_cells = all_cells, fill = NaN)
  pathway_zscores_full <- align_activity_matrix_to_cells(pathway_res$zscore, all_cells = all_cells, fill = NaN)
  pathway_pvalues_full <- align_activity_matrix_to_cells(pathway_res$pvalue, all_cells = all_cells, fill = NaN)

  tf_scores_full <- align_activity_matrix_to_cells(tf_res$score, all_cells = all_cells, fill = NaN)
  tf_zscores_full <- align_activity_matrix_to_cells(tf_res$zscore, all_cells = all_cells, fill = NaN)
  tf_pvalues_full <- align_activity_matrix_to_cells(tf_res$pvalue, all_cells = all_cells, fill = NaN)

  pathway_pca_full <- NULL
  pathway_umap_full <- NULL
  tf_pca_full <- NULL
  tf_umap_full <- NULL

  if (isTRUE(RUN_DIMRED)) {
    message(sprintf("[%s] Running pathway PCA/UMAP...", label))
    seu_work <- run_activity_dimred(
      obj = seu_work,
      assay_name = "pathwaysmlm",
      pca_name = "pathwaysmlm_pca",
      umap_name = "pathwaysmlm_umap",
      umap_key = "pmlmUMAP_",
      max_pcs = MAX_PCS,
      seed = SEED
    )

    message(sprintf("[%s] Running TF PCA/UMAP...", label))
    seu_work <- run_activity_dimred(
      obj = seu_work,
      assay_name = "tfsulm",
      pca_name = "tfsulm_pca",
      umap_name = "tfsulm_umap",
      umap_key = "tfsUMAP_",
      max_pcs = MAX_PCS,
      seed = SEED
    )

    pathway_pca_full <- extract_aligned_reduction(
      seu = seu_work,
      reduction_name = "pathwaysmlm_pca",
      all_cells = all_cells,
      prefix = "pathwaysmlm_pca_"
    )
    pathway_umap_full <- extract_aligned_reduction(
      seu = seu_work,
      reduction_name = "pathwaysmlm_umap",
      all_cells = all_cells,
      prefix = "pathwaysmlm_umap_"
    )
    tf_pca_full <- extract_aligned_reduction(
      seu = seu_work,
      reduction_name = "tfsulm_pca",
      all_cells = all_cells,
      prefix = "tfsulm_pca_"
    )
    tf_umap_full <- extract_aligned_reduction(
      seu = seu_work,
      reduction_name = "tfsulm_umap",
      all_cells = all_cells,
      prefix = "tfsulm_umap_"
    )
  }

  csv_paths <- build_csv_output_paths(label)

  mlm_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = pathway_scores_full,
    available_cells = filtered_cells,
    availability_col = "pathwaysmlm_score_available",
    pca_mat = pathway_pca_full,
    umap_mat = pathway_umap_full
  )
  mlm_zscore_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = pathway_zscores_full,
    available_cells = filtered_cells,
    availability_col = "pathwaysmlm_zscore_available",
    pca_mat = pathway_pca_full,
    umap_mat = pathway_umap_full
  )
  mlm_pvalue_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = pathway_pvalues_full,
    available_cells = filtered_cells,
    availability_col = "pathwaysmlm_pvalue_available",
    pca_mat = pathway_pca_full,
    umap_mat = pathway_umap_full
  )
  ulm_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = tf_scores_full,
    available_cells = filtered_cells,
    availability_col = "tfsulm_score_available",
    pca_mat = tf_pca_full,
    umap_mat = tf_umap_full
  )
  ulm_zscore_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = tf_zscores_full,
    available_cells = filtered_cells,
    availability_col = "tfsulm_zscore_available",
    pca_mat = tf_pca_full,
    umap_mat = tf_umap_full
  )
  ulm_pvalue_export_df <- build_activity_export_df(
    meta_df = full_meta[all_cells, , drop = FALSE],
    score_mat = tf_pvalues_full,
    available_cells = filtered_cells,
    availability_col = "tfsulm_pvalue_available",
    pca_mat = tf_pca_full,
    umap_mat = tf_umap_full
  )

  write_activity_csv(mlm_export_df, csv_paths$mlm)
  write_activity_csv(mlm_zscore_export_df, csv_paths$mlm_zscore)
  write_activity_csv(mlm_pvalue_export_df, csv_paths$mlm_pvalue)
  write_activity_csv(ulm_export_df, csv_paths$ulm)
  write_activity_csv(ulm_zscore_export_df, csv_paths$ulm_zscore)
  write_activity_csv(ulm_pvalue_export_df, csv_paths$ulm_pvalue)
  message(sprintf("[%s] Saved MLM score CSV: %s", label, csv_paths$mlm))
  message(sprintf("[%s] Saved MLM z-score CSV: %s", label, csv_paths$mlm_zscore))
  message(sprintf("[%s] Saved MLM p-value CSV: %s", label, csv_paths$mlm_pvalue))
  message(sprintf("[%s] Saved ULM score CSV: %s", label, csv_paths$ulm))
  message(sprintf("[%s] Saved ULM z-score CSV: %s", label, csv_paths$ulm_zscore))
  message(sprintf("[%s] Saved ULM p-value CSV: %s", label, csv_paths$ulm_pvalue))

  seu <- attach_provenance(
    seu = seu,
    label = label,
    input_path = input_path,
    input_class = converted$input_class,
    base_assay = base_assay,
    filtered_cell_ids = filtered_cells,
    filtered_feature_count = nrow(get_assay_data(seu_work, assay = base_assay, layer = "counts")),
    pathway_scores = pathway_scores_full,
    pathway_zscores = pathway_zscores_full,
    pathway_pvalues = pathway_pvalues_full,
    tf_scores = tf_scores_full,
    tf_zscores = tf_zscores_full,
    tf_pvalues = tf_pvalues_full,
    csv_paths = csv_paths,
    aligned_reductions = list(
      pathwaysmlm_pca = pathway_pca_full,
      pathwaysmlm_umap = pathway_umap_full,
      tfsulm_pca = tf_pca_full,
      tfsulm_umap = tf_umap_full
    )
  )
  DefaultAssay(seu) <- base_assay

  out_path <- build_output_path(input_path)
  saveRDS(seu, out_path, compress = FALSE)
  message(sprintf("[%s] Saved decoupleR-ready Seurat object: %s", label, out_path))

  invisible(out_path)
}

pathway_net <- load_pathway_net()
tf_net <- load_tf_net()

myonuclei_out <- process_object(
  input_path = MYONUCLEI_RDS_PATH,
  label = "myonuclei",
  pathway_net = pathway_net,
  tf_net = tf_net
)

myotube_out <- process_object(
  input_path = MYOTUBE_RDS_PATH,
  label = "myotube",
  pathway_net = pathway_net,
  tf_net = tf_net
)

message("Finished.")
message("  myonuclei: ", myonuclei_out)
message("  myotube:   ", myotube_out)
