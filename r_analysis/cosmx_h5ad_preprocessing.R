#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(S4Vectors)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
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

# ---------------- USER PARAMETERS ----------------
# Required inputs (.rready.h5ad files)
nuclei_combined_h5ad <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/r_ready/greedy_nuclei_combined.rready.h5ad"
myotube_combined_h5ad <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/r_ready/greedy_myotube_combined.rready.h5ad"
filtered_nuclei_h5ad <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/r_ready/greedy_filtered_nuclei.rready.h5ad"
metadata_csv_path <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/r_ready/greedy_classifier_metadata.csv"
# Optional shorthand: derive the 4 input paths above from savepath + prefix
savepath <- NULL
prefix <- ""

# Metadata behavior
classification_column <- "Predicted Class"
sigmoid_logits_column <- "Sigmoid Logits"
metadata_cell_line_candidates <- c("Cell Line", "Cell.Line", "cell_line", "CellLine")

# Output
output_dir <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds"
output_myonuclei_filename <- "processed_myonuclei.rds"
output_myotube_filename <- "processed_myotube_filtered.rds"

# Cache behavior
cache_in_same_dir_as_h5ad <- TRUE
cache_dir <- output_dir
force_rebuild_cache <- FALSE

# Assay mapping
counts_assay_candidates <- c("X", "raw_counts", "layer_counts", "raw", "matrix", "data")

# Filtering thresholds
min_cell_total_nuclei <- 100L
min_gene_ncells_nuclei <- 100L
min_cell_total_myotube <- 100L
min_gene_ncells_myotube <- 100L
remove_gene_pattern <- "SystemControl|Negative"

# Morphology settings
morphology_reduced_dim_name <- "morphology_features"
morphology_feature_names <- c("area_px2", "perimeter_px", "major_axis_length_px")
morphology_feature_indices <- c(1L, 2L, 4L)

# Myonuclei settings
is_myonucleus_column <- "is_myonucleus"
myotube_id_column <- "myotube_id"
myotube_id_unassigned_value <- -1L

# Class mapping (requested earlier: 0 = Normal, 1 = Abnormal)
normal_class_values <- c("0", "normal")
abnormal_class_values <- c("1", "abnormal")

# Join-key candidates
slide_col_candidates <- c("Slide Name", "Slide.Name", "slide_name", "slide")
field_col_candidates <- c("field", "Field", "field_key")
patch_col_candidates <- c("patch_idx", "Patch", "patch", "patch.id", "patch_idx_key")
cell_line_col_candidates <- c("Cell Line", "Cell.Line", "cell_line", "CellLine", "cell_line_key")
local_id_col_candidates <- c("local_id", "local.id", "Local ID", "Local.ID")
myotube_id_col_candidates <- c("myotube_id", "local_id", "local.id")

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Usage example:
# Rscript cosmx_h5ad_preprocessing.R \
#   --nuclei_combined_h5ad /path/nuclei_combined.rready.h5ad \
#   --myotube_combined_h5ad /path/myotube_combined.rready.h5ad \
#   --filtered_nuclei_h5ad /path/filtered_nuclei.rready.h5ad \
#   --metadata_csv_path /path/metadata_df.csv \
#   --output_dir /path/output \
#   --classification_column 'Predicted Class' \
#   --sigmoid_logits_column 'Sigmoid Logits'
#
# Or shorthand (mirrors cosmx_h5ad_to_rready.py outputs):
# Rscript cosmx_h5ad_preprocessing.R \
#   --savepath /path/to/r_ready \
#   --prefix greedy_ \
#   --output_dir /path/to/rds

overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))

# If savepath/prefix are provided, auto-derive input file paths unless explicitly set.
if ("savepath" %in% names(overrides) || "prefix" %in% names(overrides)) {
  savepath_cli <- if ("savepath" %in% names(overrides)) overrides[["savepath"]] else savepath
  prefix_cli <- if ("prefix" %in% names(overrides)) overrides[["prefix"]] else prefix
  if (!is.null(savepath_cli) && nzchar(as.character(savepath_cli))) {
    if (is.null(prefix_cli)) prefix_cli <- ""
    derived <- list(
      nuclei_combined_h5ad = file.path(savepath_cli, paste0(prefix_cli, "nuclei_combined.rready.h5ad")),
      myotube_combined_h5ad = file.path(savepath_cli, paste0(prefix_cli, "myotube_combined.rready.h5ad")),
      filtered_nuclei_h5ad = file.path(savepath_cli, paste0(prefix_cli, "filtered_nuclei.rready.h5ad")),
      metadata_csv_path = file.path(savepath_cli, paste0(prefix_cli, "classifier_metadata.csv"))
    )
    for (nm in names(derived)) {
      if (!(nm %in% names(overrides))) {
        overrides[[nm]] <- derived[[nm]]
      }
    }
  }
}

apply_cli_overrides(overrides)

# ---------------- VALIDATION ----------------
if (!requireNamespace("anndataR", quietly = TRUE)) {
  stop("Please install anndataR first: BiocManager::install('anndataR')")
}

stop_if_missing_file(nuclei_combined_h5ad, "nuclei_combined_h5ad")
stop_if_missing_file(myotube_combined_h5ad, "myotube_combined_h5ad")
stop_if_missing_file(filtered_nuclei_h5ad, "filtered_nuclei_h5ad")
stop_if_missing_file(metadata_csv_path, "metadata_csv_path")

if (!cache_in_same_dir_as_h5ad) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("========== CONFIG ==========\n")
cat("nuclei_combined_h5ad:   ", nuclei_combined_h5ad, "\n", sep = "")
cat("myotube_combined_h5ad:  ", myotube_combined_h5ad, "\n", sep = "")
cat("filtered_nuclei_h5ad:   ", filtered_nuclei_h5ad, "\n", sep = "")
cat("metadata_csv_path:      ", metadata_csv_path, "\n", sep = "")
cat("classification_column:  ", classification_column, "\n", sep = "")
cat("sigmoid_logits_column:  ", sigmoid_logits_column, "\n", sep = "")
cat("output_dir:             ", output_dir, "\n", sep = "")
cat("force_rebuild_cache:    ", force_rebuild_cache, "\n", sep = "")
cat("min_cell_total_nuclei:  ", min_cell_total_nuclei, "\n", sep = "")
cat("min_gene_ncells_nuclei: ", min_gene_ncells_nuclei, "\n", sep = "")
cat("min_cell_total_myotube: ", min_cell_total_myotube, "\n", sep = "")
cat("min_gene_ncells_myotube:", min_gene_ncells_myotube, "\n", sep = "")
cat("============================\n\n")

# ---------------- LOAD/CACHE H5AD ----------------
nuclei_combined <- load_sce_from_h5ad(
  h5ad_path = nuclei_combined_h5ad,
  force = force_rebuild_cache,
  cache_in_same_dir_as_h5ad = cache_in_same_dir_as_h5ad,
  cache_dir = cache_dir
)
myotube_combined <- load_sce_from_h5ad(
  h5ad_path = myotube_combined_h5ad,
  force = force_rebuild_cache,
  cache_in_same_dir_as_h5ad = cache_in_same_dir_as_h5ad,
  cache_dir = cache_dir
)
filtered_nuclei <- load_sce_from_h5ad(
  h5ad_path = filtered_nuclei_h5ad,
  force = force_rebuild_cache,
  cache_in_same_dir_as_h5ad = cache_in_same_dir_as_h5ad,
  cache_dir = cache_dir
)

raw_counts <- list(
  nuclei_cells = ncol(nuclei_combined),
  nuclei_genes = nrow(nuclei_combined),
  myotube_cells = ncol(myotube_combined),
  myotube_genes = nrow(myotube_combined),
  filtered_nuclei_cells = ncol(filtered_nuclei),
  filtered_nuclei_genes = nrow(filtered_nuclei)
)

# ---------------- ENSURE COUNTS ASSAY ----------------
nuclei_combined <- ensure_counts_assay(nuclei_combined, "nuclei_combined", counts_assay_candidates)
myotube_combined <- ensure_counts_assay(myotube_combined, "myotube_combined", counts_assay_candidates)
filtered_nuclei <- ensure_counts_assay(filtered_nuclei, "filtered_nuclei", counts_assay_candidates)

# ---------------- ADD/REPAIR MYONUCLEUS FLAG ----------------
if (!(is_myonucleus_column %in% colnames(colData(filtered_nuclei)))) {
  if (!(myotube_id_column %in% colnames(colData(filtered_nuclei)))) {
    stop(
      "Neither '", is_myonucleus_column, "' nor '", myotube_id_column,
      "' is present in filtered_nuclei colData."
    )
  }
  myotube_id_raw <- colData(filtered_nuclei)[[myotube_id_column]]
  myotube_id_num <- suppressWarnings(as.numeric(as.character(myotube_id_raw)))
  is_myonucleus <- if (all(is.na(myotube_id_num))) {
    normalize_label(myotube_id_raw) != as.character(myotube_id_unassigned_value)
  } else {
    !is.na(myotube_id_num) & (myotube_id_num != myotube_id_unassigned_value)
  }
  colData(filtered_nuclei)[[is_myonucleus_column]] <- as.integer(is_myonucleus)
}

# ---------------- ADD MORPHOLOGY FEATURES ----------------
rd_names <- SingleCellExperiment::reducedDimNames(filtered_nuclei)
if (morphology_reduced_dim_name %in% rd_names) {
  morph <- SingleCellExperiment::reducedDim(filtered_nuclei, morphology_reduced_dim_name)
  if (max(morphology_feature_indices) <= ncol(morph)) {
    for (i in seq_along(morphology_feature_names)) {
      colData(filtered_nuclei)[[morphology_feature_names[i]]] <- morph[, morphology_feature_indices[i]]
    }
  } else {
    warning(
      "ReducedDim '", morphology_reduced_dim_name, "' has ", ncol(morph),
      " columns; expected >= ", max(morphology_feature_indices), ". Skipping morphology column assignment."
    )
  }
} else {
  warning(
    "ReducedDim '", morphology_reduced_dim_name,
    "' not found in filtered_nuclei input. Skipping morphology feature assignment."
  )
}

# ---------------- FILTER NUCLEI ----------------
cnt_nuc <- SummarizedExperiment::assay(filtered_nuclei, "counts")

cell_totals_nuc <- Matrix::colSums(cnt_nuc)
keep_cells_nuc <- cell_totals_nuc >= min_cell_total_nuclei
filtered_nuclei <- filtered_nuclei[, keep_cells_nuc]

cnt_nuc <- SummarizedExperiment::assay(filtered_nuclei, "counts")
gene_ncells_nuc <- Matrix::rowSums(cnt_nuc > 0)
keep_genes_nuc <- gene_ncells_nuc >= min_gene_ncells_nuclei
filtered_nuclei <- filtered_nuclei[keep_genes_nuc, ]

bad_genes_nuc <- grepl(remove_gene_pattern, rownames(filtered_nuclei), ignore.case = TRUE)
filtered_nuclei <- filtered_nuclei[!bad_genes_nuc, ]

# ---------------- FILTER MYOTUBES ----------------
cnt_tube <- SummarizedExperiment::assay(myotube_combined, "counts")

cell_totals_tube <- Matrix::colSums(cnt_tube)
keep_cells_tube <- cell_totals_tube >= min_cell_total_myotube
myotube_combined <- myotube_combined[, keep_cells_tube]

cnt_tube <- SummarizedExperiment::assay(myotube_combined, "counts")
gene_ncells_tube <- Matrix::rowSums(cnt_tube > 0)
keep_genes_tube <- gene_ncells_tube >= min_gene_ncells_myotube
myotube_combined <- myotube_combined[keep_genes_tube, ]

bad_genes_tube <- grepl(remove_gene_pattern, rownames(myotube_combined), ignore.case = TRUE)
myotube_combined <- myotube_combined[!bad_genes_tube, ]

# ---------------- METADATA SUMMARY ----------------
metadata_df <- readr::read_csv(metadata_csv_path, show_col_types = FALSE)
if (!(classification_column %in% names(metadata_df))) {
  stop(
    "classification_column='", classification_column,
    "' not found in metadata. Available: ", paste(names(metadata_df), collapse = ", ")
  )
}
if (!(sigmoid_logits_column %in% names(metadata_df))) {
  stop(
    "sigmoid_logits_column='", sigmoid_logits_column,
    "' not found in metadata. Available: ", paste(names(metadata_df), collapse = ", ")
  )
}

cell_line_col_meta <- pick_col(metadata_df, metadata_cell_line_candidates, "Cell Line")

metadata_summary <- metadata_df %>%
  mutate(
    cell_line = as.character(.data[[cell_line_col_meta]]),
    class_value = as.character(.data[[classification_column]])
  ) %>%
  count(cell_line, class_value, name = "n") %>%
  group_by(cell_line) %>%
  mutate(pct = round(100 * n / sum(n), 2)) %>%
  ungroup() %>%
  arrange(cell_line, desc(n))

# ---------------- MERGE METADATA INTO filtered_nuclei ----------------
metadata_df_norm <- metadata_df %>%
  add_alias_col("slide_name", slide_col_candidates, "Slide Name") %>%
  add_alias_col("field_key", field_col_candidates, "field") %>%
  add_alias_col("patch_idx_key", patch_col_candidates, "patch_idx") %>%
  add_alias_col("cell_line_key", cell_line_col_candidates, "Cell Line") %>%
  add_alias_col("local_id_key", local_id_col_candidates, "local_id") %>%
  add_alias_col(
    "classification_key",
    c(classification_column, "Classification", "classification", "Predicted Class", "predicted_class"),
    "classification",
    required = TRUE
  ) %>%
  add_alias_col(
    "sigmoid_logits_key",
    c(sigmoid_logits_column, "Sigmoid Logits", "sigmoid_logits", "sigmoid.logits"),
    "sigmoid_logits",
    required = TRUE
  ) %>%
  mutate(
    slide_name = toupper(as_key(slide_name)),
    field_key = as_key(field_key),
    cell_line_key = as_key(cell_line_key),
    patch_idx_key = suppressWarnings(as.integer(as.character(patch_idx_key))),
    local_id_key = suppressWarnings(as.integer(as.character(local_id_key))),
    classification_key = as.character(classification_key),
    sigmoid_logits_key = suppressWarnings(as.numeric(as.character(sigmoid_logits_key)))
  )

md_keep <- metadata_df_norm %>%
  select(
    slide_name, field_key, patch_idx_key, cell_line_key, local_id_key,
    classification_key, sigmoid_logits_key
  ) %>%
  rename(
    Classification = classification_key,
    Sigmoid_Logits = sigmoid_logits_key
  ) %>%
  distinct(slide_name, field_key, patch_idx_key, cell_line_key, local_id_key, .keep_all = TRUE)

cd_nuc <- as.data.frame(colData(filtered_nuclei)) %>%
  add_alias_col("slide_name", slide_col_candidates, "Slide Name") %>%
  add_alias_col("field_key", field_col_candidates, "field") %>%
  add_alias_col("patch_idx_key", patch_col_candidates, "patch_idx") %>%
  add_alias_col("cell_line_key", cell_line_col_candidates, "Cell Line") %>%
  add_alias_col("local_id_key", local_id_col_candidates, "local_id") %>%
  mutate(
    slide_name = toupper(as_key(slide_name)),
    field_key = as_key(field_key),
    cell_line_key = as_key(cell_line_key),
    patch_idx_key = suppressWarnings(as.integer(as.character(patch_idx_key))),
    local_id_key = suppressWarnings(as.integer(as.character(local_id_key))),
    .row_id = rownames(.)
  ) %>%
  select(-any_of(c("Classification", "Sigmoid_Logits")))

cd_nuc2 <- cd_nuc %>%
  left_join(
    md_keep,
    by = c("slide_name", "field_key", "patch_idx_key", "cell_line_key", "local_id_key"),
    multiple = "first"
  ) %>%
  arrange(match(.row_id, cd_nuc$.row_id))

colData(filtered_nuclei) <- S4Vectors::DataFrame(
  cd_nuc2 %>% select(-.row_id),
  row.names = cd_nuc$.row_id
)

classification_assigned <- sum(!is.na(colData(filtered_nuclei)$Classification))
classification_missing <- sum(is.na(colData(filtered_nuclei)$Classification))
sigmoid_logits_assigned <- sum(!is.na(colData(filtered_nuclei)$Sigmoid_Logits))
sigmoid_logits_missing <- sum(is.na(colData(filtered_nuclei)$Sigmoid_Logits))

# ---------------- SLIDE + CLASS SUMMARY STATS ----------------
cd_stats <- as.data.frame(colData(filtered_nuclei))
slide_col_stats <- pick_col(cd_stats, c("Slide Name", "Slide.Name", "slide_name", "slide", "slide_name"), "Slide Name")
class_col_stats <- pick_col(cd_stats, c("Classification"), "Classification", required = FALSE)

cnt_stats <- SummarizedExperiment::assay(filtered_nuclei, "counts")
cell_counts_stats <- Matrix::colSums(cnt_stats)
cell_unique_stats <- Matrix::colSums(cnt_stats > 0)

cd_stats <- cd_stats %>%
  mutate(
    slide_key = as.character(.data[[slide_col_stats]]),
    class_key = if (!is.na(class_col_stats)) as.character(.data[[class_col_stats]]) else NA_character_,
    .cell_counts = cell_counts_stats,
    .cell_unique = cell_unique_stats
  )

slide_stats_df <- cd_stats %>%
  group_by(slide_key) %>%
  summarise(
    `Avg Count (All)` = mean(.cell_counts),
    `Avg Unique Genes (All)` = mean(.cell_unique),
    .groups = "drop"
  )

if (!all(is.na(cd_stats$class_key))) {
  class_levels <- sort(unique(na.omit(cd_stats$class_key)))
  for (cls in class_levels) {
    cls_safe <- safe_colname_fragment(cls)
    tmp <- cd_stats %>%
      filter(class_key == cls) %>%
      group_by(slide_key) %>%
      summarise(
        avg_count = mean(.cell_counts),
        avg_unique = mean(.cell_unique),
        .groups = "drop"
      )
    slide_stats_df <- slide_stats_df %>%
      left_join(tmp, by = "slide_key") %>%
      rename(
        !!paste0("Class_", cls_safe, "_Avg_Count") := avg_count,
        !!paste0("Class_", cls_safe, "_Avg_Unique") := avg_unique
      )
  }
}

slide_stats_df <- slide_stats_df %>%
  rename(`Slide Name` = slide_key) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))

# ---------------- MYONUCLEI SUBSET ----------------
if (!(is_myonucleus_column %in% colnames(colData(filtered_nuclei)))) {
  stop(
    "Column '", is_myonucleus_column, "' not found in filtered_nuclei colData.\nAvailable: ",
    paste(colnames(colData(filtered_nuclei)), collapse = ", ")
  )
}

is_myonucleus_flag <- to_myonucleus_flag(colData(filtered_nuclei)[[is_myonucleus_column]])
myonuclei <- filtered_nuclei[, is_myonucleus_flag]

# ---------------- MYOTUBE-LEVEL NUCLEUS CLASS COUNTS ----------------
cd_nuclei_for_tube <- as.data.frame(colData(myonuclei))
cd_myotube <- as.data.frame(colData(myotube_combined))

slide_n <- pick_col(cd_nuclei_for_tube, c("Slide Name", "Slide.Name", "slide_name", "slide"), "Slide Name")
field_n <- pick_col(cd_nuclei_for_tube, c("field", "Field", "field_key"), "field")
patch_n <- pick_col(cd_nuclei_for_tube, c("patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"), "patch_idx")
tube_n <- pick_col(cd_nuclei_for_tube, myotube_id_col_candidates, "myotube_id/local_id")
class_n <- pick_col(cd_nuclei_for_tube, c("Classification", classification_column), "Classification")

slide_t <- pick_col(cd_myotube, c("Slide Name", "Slide.Name", "slide_name", "slide"), "Slide Name")
field_t <- pick_col(cd_myotube, c("field", "Field", "field_key"), "field")
patch_t <- pick_col(cd_myotube, c("patch_idx", "Patch", "patch", "patch.id", "patch_idx_key"), "patch_idx")
tube_t <- pick_col(cd_myotube, myotube_id_col_candidates, "myotube_id/local_id")

normal_values_norm <- normalize_label(normal_class_values)
abnormal_values_norm <- normalize_label(abnormal_class_values)

nuc_df <- cd_nuclei_for_tube %>%
  transmute(
    slide_key = toupper(as_key(.data[[slide_n]])),
    field_key = as_key(.data[[field_n]]),
    patch_key = as_key(.data[[patch_n]]),
    tube_key = as_key(.data[[tube_n]]),
    class_raw = as.character(.data[[class_n]])
  )

nuc_df$class_norm <- normalize_label(nuc_df$class_raw)
nuc_df$class_group <- NA_character_
nuc_df$class_group[nuc_df$class_norm %in% normal_values_norm] <- "Normal"
nuc_df$class_group[nuc_df$class_norm %in% abnormal_values_norm] <- "Abnormal"
nuc_df <- nuc_df %>% filter(!is.na(class_group))

counts_by_tube <- nuc_df %>%
  count(slide_key, field_key, patch_key, tube_key, class_group) %>%
  pivot_wider(names_from = class_group, values_from = n, values_fill = 0) %>%
  rename(
    n_normal_nuclei = Normal,
    n_abnormal_nuclei = Abnormal
  )

if (!"n_normal_nuclei" %in% names(counts_by_tube)) counts_by_tube$n_normal_nuclei <- 0L
if (!"n_abnormal_nuclei" %in% names(counts_by_tube)) counts_by_tube$n_abnormal_nuclei <- 0L

tube_df <- cd_myotube %>%
  transmute(
    slide_key = toupper(as_key(.data[[slide_t]])),
    field_key = as_key(.data[[field_t]]),
    patch_key = as_key(.data[[patch_t]]),
    tube_key = as_key(.data[[tube_t]]),
    .row_id = rownames(cd_myotube)
  )

merge_df <- tube_df %>%
  left_join(counts_by_tube, by = c("slide_key", "field_key", "patch_key", "tube_key")) %>%
  arrange(match(.row_id, tube_df$.row_id))

merge_df$n_normal_nuclei[is.na(merge_df$n_normal_nuclei)] <- 0L
merge_df$n_abnormal_nuclei[is.na(merge_df$n_abnormal_nuclei)] <- 0L

total_nuclei <- merge_df$n_normal_nuclei + merge_df$n_abnormal_nuclei
merge_df$pct_abnormal_nuclei <- ifelse(total_nuclei > 0, merge_df$n_abnormal_nuclei / total_nuclei, NA_real_)

# morphology_class: -1=no nuclei, 1=only normal, 2=only abnormal, 3=mixed
merge_df$morphology_class <- -1L
merge_df$morphology_class[merge_df$n_normal_nuclei > 0 & merge_df$n_abnormal_nuclei == 0] <- 1L
merge_df$morphology_class[merge_df$n_normal_nuclei == 0 & merge_df$n_abnormal_nuclei > 0] <- 2L
merge_df$morphology_class[merge_df$n_normal_nuclei > 0 & merge_df$n_abnormal_nuclei > 0] <- 3L

colData(myotube_combined)$n_normal_nuclei <- merge_df$n_normal_nuclei
colData(myotube_combined)$n_abnormal_nuclei <- merge_df$n_abnormal_nuclei
colData(myotube_combined)$pct_abnormal_nuclei <- merge_df$pct_abnormal_nuclei
colData(myotube_combined)$morphology_class <- merge_df$morphology_class

has_nuclei <- (colData(myotube_combined)$n_normal_nuclei > 0) |
              (colData(myotube_combined)$n_abnormal_nuclei > 0)
myotube_filtered <- myotube_combined[, has_nuclei]

# ---------------- SAVE OUTPUTS ----------------
myonuclei_file <- file.path(output_dir, output_myonuclei_filename)
myotube_file <- file.path(output_dir, output_myotube_filename)

saveRDS(myonuclei, myonuclei_file, compress = FALSE)
saveRDS(myotube_filtered, myotube_file, compress = FALSE)

# ---------------- SUMMARY PRINTS ----------------
cat("\n========== SUMMARY ==========\n")
cat("Nuclei\n")
cat("  Raw nuclei_combined: cells=", raw_counts$nuclei_cells, ", genes=", raw_counts$nuclei_genes, "\n", sep = "")
cat("  Raw filtered_nuclei input: cells=", raw_counts$filtered_nuclei_cells, ", genes=", raw_counts$filtered_nuclei_genes, "\n", sep = "")
cat("  Final filtered_nuclei (QC): cells=", ncol(filtered_nuclei), ", genes=", nrow(filtered_nuclei), "\n", sep = "")

cat("\nMyotubes\n")
cat("  Raw myotube_combined: cells=", raw_counts$myotube_cells, ", genes=", raw_counts$myotube_genes, "\n", sep = "")
cat("  Final myotube_combined (QC): cells=", ncol(myotube_combined), ", genes=", nrow(myotube_combined), "\n", sep = "")
cat("  Kept with >=1 nucleus: ", ncol(myotube_filtered), "\n", sep = "")

cat("\nMetadata merge\n")
cat("  classification_column used: ", classification_column, "\n", sep = "")
cat("  sigmoid_logits_column used: ", sigmoid_logits_column, "\n", sep = "")
cat("  Filtered nuclei with Classification assigned: ", classification_assigned, "\n", sep = "")
cat("  Filtered nuclei missing Classification: ", classification_missing, "\n", sep = "")
cat("  Filtered nuclei with Sigmoid_Logits assigned: ", sigmoid_logits_assigned, "\n", sep = "")
cat("  Filtered nuclei missing Sigmoid_Logits: ", sigmoid_logits_missing, "\n", sep = "")

cat("\nMyonuclei\n")
cat("  Count: ", ncol(myonuclei), "\n", sep = "")

cat("\nFiltered nuclei Classification counts:\n")
print(sort(table(colData(filtered_nuclei)$Classification, useNA = "ifany"), decreasing = TRUE))

cat("\nMyotube nucleus class counts (derived) summary:\n")
print(summary(colData(myotube_combined)$n_normal_nuclei))
print(summary(colData(myotube_combined)$n_abnormal_nuclei))

cat("\nMyotube morphology_class counts (-1 none, 1 normal-only, 2 abnormal-only, 3 mixed):\n")
print(table(colData(myotube_combined)$morphology_class, useNA = "ifany"))

cat("\nMetadata class distribution (first 20 rows):\n")
print(utils::head(metadata_summary, 20))

cat("\nSlide-level stats:\n")
print(slide_stats_df)

cat("\nSaved files:\n")
cat(" - ", myonuclei_file, "\n", sep = "")
cat(" - ", myotube_file, "\n", sep = "")
cat("=============================\n")
