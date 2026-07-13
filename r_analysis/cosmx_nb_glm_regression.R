#!/usr/bin/env Rscript

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

# ---------------- USER INPUTS ----------------
myonuclei_rds_path <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/nuclear_aberration_test_nucaber_v1.3_cosmx_1024x_64dim_2blocks_bs4_calibrated/processed_myonuclei.rds"
myotube_rds_path <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/rds/nuclear_aberration_test_nucaber_v1.3_cosmx_1024x_64dim_2blocks_bs4_calibrated/processed_myotube_filtered.rds"
decoupler_basepath <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/decoupleR/nuclear_aberration_test_nucaber_v1.3_cosmx_1024x_64dim_2blocks_bs4_calibrated"
output_dir <- "/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/cosmx_slides_combined/r_dataset/nb_glm_regression/nuclear_aberration_test_nucaber_v1.3_cosmx_1024x_64dim_2blocks_bs4_calibrated"

object_types <- c("myonuclei", "myotubes")
target_types <- c("genes", "decoupler_pathways", "decoupler_tfs")
myonuclei_morphology_features <- c(
  "nuclear_aberration_mean_sigmoid",
  "nuclear_aberration_total_sigmoid",
  "Sigmoid_Logits",
  "area_px2",
  "perimeter_px",
  "major_axis_length_px"
)
myotube_morphology_features <- c(
  "nuclear_aberration_myotube_total_sigmoid",
  "nuclear_aberration_myotube_mean_sigmoid",
  "nuclear_aberration_assigned_nuclei_total_sigmoid",
  "nuclear_aberration_assigned_nuclei_total_per_nuclear_area",
  "nuclear_aberration_myotube_total_per_nuclear_area",
  "pct_abnormal_nuclei"
)
morphology_features <- character(0)
cell_lines_to_include <- character(0)
run_mutant_status_interaction <- TRUE
mutant_status_control_lines <- c("NCRM1", "NCRM5", "L302P-Corr")
mutant_status_mutant_lines <- c("R249W", "K32Del", "L302P", "1175", "1174")

predictor_mode <- "auto"
global_rank_feature_patterns <- c("nuclear_aberration")
slide_covariate <- TRUE
min_obs_per_cell_line <- 100L
min_expr_obs <- 20L
alpha_padj <- 0.05
effect_thr <- 0.15
min_myonuclei_per_myotube <- 1L
filter_myonuclei <- TRUE
myonucleus_col <- "is_myonucleus"

# ---------------- OPTIONAL CLI OVERRIDES ----------------
# Supports:
#   --KEY value
#   --key=value
#   key=value
# List arguments may be comma-separated or space-separated.
# Example:
# Rscript cosmx_nb_glm_regression.R \
#   --myonuclei_rds_path /path/processed_myonuclei.rds \
#   --myotube_rds_path /path/processed_myotube_filtered.rds \
#   --decoupler_basepath /path/decoupleR \
#   --output_dir /path/nb_glm_regression \
#   --object_types myonuclei myotubes \
#   --target_types genes decoupler_pathways \
#   --myonuclei_morphology_features nuclear_aberration_mean_sigmoid area_px2 \
#   --myotube_morphology_features nuclear_aberration_myotube_total_per_nuclear_area pct_abnormal_nuclei \
#   --cell_lines_to_include NCRM1 \
#   --run_mutant_status_interaction TRUE \
#   --mutant_status_control_lines NCRM1 NCRM5 L302P-Corr \
#   --mutant_status_mutant_lines R249W K32Del L302P 1175 1174 \
#   --predictor_mode auto \
#   --global_rank_feature_patterns nuclear_aberration \
#   --slide_covariate TRUE \
#   --min_myonuclei_per_myotube 2 \
#   --filter_myonuclei TRUE \
#   --myonucleus_col is_myonucleus
overrides <- parse_cli_overrides(commandArgs(trailingOnly = TRUE))
apply_cli_overrides(overrides)

decoupler_basepath <- nbglm_parse_nullable_path(decoupler_basepath)
object_types <- nbglm_canonical_object_type(nbglm_parse_list_arg(object_types))
target_types <- nbglm_canonical_target_type(nbglm_parse_list_arg(target_types))
myonuclei_morphology_features <- nbglm_parse_list_arg(myonuclei_morphology_features)
myotube_morphology_features <- nbglm_parse_list_arg(myotube_morphology_features)
morphology_features <- nbglm_parse_list_arg(morphology_features)
cell_lines_to_include <- nbglm_parse_list_arg(cell_lines_to_include)
mutant_status_control_lines <- nbglm_parse_list_arg(mutant_status_control_lines)
mutant_status_mutant_lines <- nbglm_parse_list_arg(mutant_status_mutant_lines)
global_rank_feature_patterns <- nbglm_parse_list_arg(global_rank_feature_patterns)

if (!length(object_types)) stop("object_types cannot be empty.")
if (!length(target_types)) stop("target_types cannot be empty.")
if (!length(myonuclei_morphology_features) &&
    !length(myotube_morphology_features) &&
    !length(morphology_features)) {
  stop("No morphology features provided. Use myonuclei_morphology_features and/or myotube_morphology_features.")
}
if (!nzchar(output_dir)) stop("output_dir cannot be empty.")

if ("decoupler_pathways" %in% target_types || "decoupler_tfs" %in% target_types) {
  if (is.null(decoupler_basepath)) {
    warning("decoupler_basepath is NULL; decoupleR target conditions will be skipped.")
  } else if (!dir.exists(decoupler_basepath)) {
    stop("decoupler_basepath not found: ", decoupler_basepath)
  }
}

cat("========== COSMX NB-GLM REGRESSION CONFIG ==========\n")
cat("myonuclei_rds_path:     ", myonuclei_rds_path, "\n", sep = "")
cat("myotube_rds_path:       ", myotube_rds_path, "\n", sep = "")
cat("decoupler_basepath:     ", ifelse(is.null(decoupler_basepath), "NULL", decoupler_basepath), "\n", sep = "")
cat("output_dir:             ", output_dir, "\n", sep = "")
cat("object_types:           ", paste(object_types, collapse = ", "), "\n", sep = "")
cat("target_types:           ", paste(target_types, collapse = ", "), "\n", sep = "")
cat("myonuclei_morphology_features:", paste(myonuclei_morphology_features, collapse = ", "), "\n", sep = "")
cat("myotube_morphology_features:", paste(myotube_morphology_features, collapse = ", "), "\n", sep = "")
cat("legacy morphology_features:", ifelse(length(morphology_features), paste(morphology_features, collapse = ", "), "<none>"), "\n", sep = "")
cat("cell_lines_to_include:  ", ifelse(length(cell_lines_to_include), paste(cell_lines_to_include, collapse = ", "), "<all>"), "\n", sep = "")
cat("run_mutant_status_interaction:", run_mutant_status_interaction, "\n", sep = "")
cat("mutant_status_control_lines:", paste(mutant_status_control_lines, collapse = ", "), "\n", sep = "")
cat("mutant_status_mutant_lines:", paste(mutant_status_mutant_lines, collapse = ", "), "\n", sep = "")
cat("predictor_mode:         ", predictor_mode, "\n", sep = "")
cat("global_rank_feature_patterns:", paste(global_rank_feature_patterns, collapse = ", "), "\n", sep = "")
cat("slide_covariate:        ", slide_covariate, "\n", sep = "")
cat("min_obs_per_cell_line:  ", min_obs_per_cell_line, "\n", sep = "")
cat("min_expr_obs:           ", min_expr_obs, "\n", sep = "")
cat("alpha_padj:             ", alpha_padj, "\n", sep = "")
cat("effect_thr:             ", effect_thr, "\n", sep = "")
cat("min_myonuclei_per_myotube:", min_myonuclei_per_myotube, "\n", sep = "")
cat("filter_myonuclei:       ", filter_myonuclei, "\n", sep = "")
cat("myonucleus_col:         ", myonucleus_col, "\n", sep = "")
cat("====================================================\n\n")

cat("========== PREDICTOR TRANSFORM PLAN ==========\n")
predictor_plan <- nbglm_predictor_plan_df(
  object_types = object_types,
  myonuclei_morphology_features = myonuclei_morphology_features,
  myotube_morphology_features = myotube_morphology_features,
  morphology_features = morphology_features,
  predictor_mode = predictor_mode,
  global_rank_feature_patterns = global_rank_feature_patterns
)
print(predictor_plan, row.names = FALSE)
cat("==============================================\n\n")

t0_all <- Sys.time()
nbglm_run_workflow(
  myonuclei_rds_path = myonuclei_rds_path,
  myotube_rds_path = myotube_rds_path,
  decoupler_basepath = decoupler_basepath,
  output_dir = output_dir,
  object_types = object_types,
  target_types = target_types,
  myonuclei_morphology_features = myonuclei_morphology_features,
  myotube_morphology_features = myotube_morphology_features,
  morphology_features = morphology_features,
  cell_lines_to_include = cell_lines_to_include,
  run_mutant_status_interaction = run_mutant_status_interaction,
  mutant_status_control_lines = mutant_status_control_lines,
  mutant_status_mutant_lines = mutant_status_mutant_lines,
  predictor_mode = predictor_mode,
  global_rank_feature_patterns = global_rank_feature_patterns,
  slide_covariate = slide_covariate,
  min_obs_per_cell_line = min_obs_per_cell_line,
  min_expr_obs = min_expr_obs,
  alpha_padj = alpha_padj,
  effect_thr = effect_thr,
  min_myonuclei_per_myotube = min_myonuclei_per_myotube,
  filter_myonuclei = filter_myonuclei,
  myonucleus_col = myonucleus_col
)

message(sprintf(
  "CosMx regression workflow finished in %.1f minutes",
  as.numeric(difftime(Sys.time(), t0_all, units = "mins"))
))
