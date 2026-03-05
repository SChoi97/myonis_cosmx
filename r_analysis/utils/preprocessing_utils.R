get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }
  getwd()
}

parse_cli_overrides <- function(args) {
  out <- list()
  if (length(args) == 0) return(out)
  i <- 1L
  n <- length(args)
  while (i <= n) {
    a <- args[[i]]

    # Supports: key=value and --key=value
    if (grepl("=", a, fixed = TRUE)) {
      parts <- strsplit(a, "=", fixed = TRUE)[[1]]
      key <- parts[1]
      value <- paste(parts[-1], collapse = "=")
      key <- sub("^--", "", key)
      key <- gsub("-", "_", key, fixed = TRUE)
      if (!nzchar(key)) {
        stop("Invalid argument key in: ", a)
      }
      out[[key]] <- value
      i <- i + 1L
      next
    }

    # Supports: --key value, --key v1 v2 ..., and bare --flag (treated as TRUE)
    if (startsWith(a, "--")) {
      key <- sub("^--", "", a)
      key <- gsub("-", "_", key, fixed = TRUE)
      if (!nzchar(key)) {
        stop("Invalid argument key in: ", a)
      }

      j <- i + 1L
      vals <- character(0)
      while (j <= n && !startsWith(args[[j]], "--")) {
        vals <- c(vals, args[[j]])
        j <- j + 1L
      }

      if (length(vals) > 0) {
        # Multiple values are comma-joined for downstream coercion.
        out[[key]] <- paste(vals, collapse = ",")
        i <- j
      } else {
        out[[key]] <- "TRUE"
        i <- i + 1L
      }
      next
    }

    stop("Invalid argument (expected key=value, --key=value, or --key value): ", a)
  }
  out
}

coerce_override <- function(raw_value, current_value) {
  split_vals <- function(x) trimws(strsplit(as.character(x), ",", fixed = TRUE)[[1]])

  if (is.logical(current_value)) {
    vals <- split_vals(raw_value)
    out <- tolower(vals) %in% c("1", "true", "t", "yes", "y")
    if (length(current_value) <= 1) return(out[1])
    return(out)
  }
  if (is.integer(current_value)) {
    vals <- split_vals(raw_value)
    out <- as.integer(vals)
    if (length(current_value) <= 1) return(out[1])
    return(out)
  }
  if (is.numeric(current_value)) {
    vals <- split_vals(raw_value)
    out <- as.numeric(vals)
    if (length(current_value) <= 1) return(out[1])
    return(out)
  }
  if (is.character(current_value) && length(current_value) > 1) {
    return(split_vals(raw_value))
  }
  as.character(raw_value)
}

apply_cli_overrides <- function(overrides) {
  if (length(overrides) == 0) return(invisible(NULL))
  for (nm in names(overrides)) {
    if (!exists(nm, envir = .GlobalEnv, inherits = FALSE)) {
      warning("Ignoring unknown override: ", nm)
      next
    }
    cur <- get(nm, envir = .GlobalEnv, inherits = FALSE)
    new_val <- coerce_override(overrides[[nm]], cur)
    assign(nm, new_val, envir = .GlobalEnv)
  }
  invisible(NULL)
}

stop_if_missing_file <- function(path, label) {
  if (is.na(path) || !nzchar(path)) stop(label, " is required.")
  if (!file.exists(path)) stop(label, " not found: ", path)
}

pick_col <- function(df, candidates, label, required = TRUE) {
  hits <- candidates[candidates %in% names(df)]
  if (length(hits) == 0) {
    if (required) {
      stop(
        "Missing required column for ", label, ". Tried: ",
        paste(candidates, collapse = ", "),
        "\nAvailable columns: ",
        paste(names(df), collapse = ", ")
      )
    }
    return(NA_character_)
  }
  hits[1]
}

add_alias_col <- function(df, new_name, candidates, label = new_name, required = TRUE) {
  src <- pick_col(df, candidates, label = label, required = required)
  if (!is.na(src)) df[[new_name]] <- df[[src]]
  df
}

as_key <- function(x) trimws(as.character(x))

normalize_label <- function(x) tolower(trimws(as.character(x)))

to_myonucleus_flag <- function(x) {
  if (is.logical(x)) return(replace(x, is.na(x), FALSE))
  if (is.numeric(x) || is.integer(x)) return(!is.na(x) & x == 1)
  chr <- normalize_label(x)
  chr %in% c("1", "true", "t", "yes", "y")
}

safe_colname_fragment <- function(x) {
  x <- gsub("[^A-Za-z0-9]+", "_", as.character(x))
  gsub("^_+|_+$", "", x)
}

ensure_counts_assay <- function(sce, obj_name = "sce", candidates) {
  an <- SummarizedExperiment::assayNames(sce)
  if ("counts" %in% an) {
    message("[", obj_name, "] counts assay already present.")
    return(sce)
  }

  hit <- candidates[candidates %in% an][1]
  if (is.na(hit)) {
    stop(
      "[", obj_name, "] No counts-like assay found. Available assays: ",
      paste(an, collapse = ", ")
    )
  }

  message("[", obj_name, "] Mapping assay '", hit, "' -> 'counts'")
  SummarizedExperiment::assay(sce, "counts") <- SummarizedExperiment::assay(sce, hit)
  sce
}

cache_path_for <- function(h5ad_path, cache_in_same_dir_as_h5ad, cache_dir) {
  base <- paste0(tools::file_path_sans_ext(basename(h5ad_path)), ".sce.rds")
  if (cache_in_same_dir_as_h5ad) {
    file.path(dirname(h5ad_path), base)
  } else {
    file.path(cache_dir, base)
  }
}

build_cache <- function(h5ad_path, force, cache_in_same_dir_as_h5ad, cache_dir) {
  out <- cache_path_for(
    h5ad_path = h5ad_path,
    cache_in_same_dir_as_h5ad = cache_in_same_dir_as_h5ad,
    cache_dir = cache_dir
  )

  if (!force && file.exists(out) && file.info(out)$mtime >= file.info(h5ad_path)$mtime) {
    message("Cache up-to-date: ", out)
    return(out)
  }

  t0 <- Sys.time()
  message("Reading H5AD: ", h5ad_path)
  sce <- anndataR::read_h5ad(h5ad_path, as = "SingleCellExperiment")

  message("Writing cache: ", out)
  saveRDS(sce, out, compress = FALSE)
  rm(sce)
  gc(verbose = FALSE)

  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  message(sprintf("Done in %.1f sec: %s", dt, basename(out)))
  out
}

load_sce_from_h5ad <- function(h5ad_path, force, cache_in_same_dir_as_h5ad, cache_dir) {
  cache_file <- build_cache(
    h5ad_path = h5ad_path,
    force = force,
    cache_in_same_dir_as_h5ad = cache_in_same_dir_as_h5ad,
    cache_dir = cache_dir
  )
  readRDS(cache_file)
}

prepare_sce_for_concat <- function(sce, slide_label, slide_col = "Slide Name", prefix_cell_ids = TRUE) {
  cd <- as.data.frame(SummarizedExperiment::colData(sce))
  cd[[slide_col]] <- slide_label
  SummarizedExperiment::colData(sce) <- S4Vectors::DataFrame(cd, row.names = colnames(sce))

  rownames(sce) <- make.unique(as.character(rownames(sce)))
  current_colnames <- as.character(colnames(sce))
  if (prefix_cell_ids) {
    colnames(sce) <- make.unique(paste0(slide_label, "_", current_colnames))
  } else {
    colnames(sce) <- make.unique(current_colnames)
  }

  sce
}

align_counts_to_union_genes <- function(counts_mat, rownames_union, colnames_target) {
  if (is.null(rownames(counts_mat))) {
    stop("counts matrix is missing rownames; cannot align by genes.")
  }
  counts_mat <- as(counts_mat, "dgCMatrix")
  gene_map <- match(rownames(counts_mat), rownames_union)
  if (anyNA(gene_map)) {
    stop("Failed to map one or more genes to union rownames.")
  }

  # Sparse reconstruction into union gene space
  i_old <- counts_mat@i + 1L
  j <- rep(seq_len(ncol(counts_mat)), diff(counts_mat@p))
  x <- counts_mat@x
  i_new <- gene_map[i_old]

  Matrix::sparseMatrix(
    i = i_new,
    j = j,
    x = x,
    dims = c(length(rownames_union), ncol(counts_mat)),
    dimnames = list(rownames_union, colnames_target)
  )
}

combine_sce_outer <- function(sce_list, obj_name = "combined_sce", reduced_dim_name = NULL) {
  if (length(sce_list) == 0) {
    stop("combine_sce_outer received an empty list.")
  }

  all_genes <- unique(unlist(lapply(sce_list, rownames), use.names = FALSE))
  all_genes <- make.unique(as.character(all_genes))

  aligned_counts <- lapply(sce_list, function(sce) {
    align_counts_to_union_genes(
      counts_mat = SummarizedExperiment::assay(sce, "counts"),
      rownames_union = all_genes,
      colnames_target = colnames(sce)
    )
  })
  combined_counts <- do.call(cbind, aligned_counts)

  combined_coldata <- dplyr::bind_rows(lapply(sce_list, function(sce) {
    as.data.frame(SummarizedExperiment::colData(sce))
  }))
  combined_colnames <- unlist(lapply(sce_list, colnames), use.names = FALSE)
  rownames(combined_coldata) <- combined_colnames

  out <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = combined_counts),
    colData = S4Vectors::DataFrame(combined_coldata, row.names = combined_colnames)
  )

  # Preserve a named reducedDim when available across all inputs
  if (!is.null(reduced_dim_name) && nzchar(reduced_dim_name)) {
    rd_list <- lapply(sce_list, function(sce) {
      tryCatch(
        SingleCellExperiment::reducedDim(sce, reduced_dim_name),
        error = function(e) NULL
      )
    })
    if (all(vapply(rd_list, function(x) !is.null(x), logical(1)))) {
      dims <- unique(vapply(rd_list, ncol, integer(1)))
      if (length(dims) == 1) {
        rd_combined <- do.call(rbind, rd_list)
        rownames(rd_combined) <- colnames(out)
        SingleCellExperiment::reducedDim(out, reduced_dim_name) <- rd_combined
      } else {
        warning(
          "[", obj_name, "] reducedDim '", reduced_dim_name,
          "' has inconsistent dimensions across inputs. Skipping reducedDim merge."
        )
      }
    } else {
      warning(
        "[", obj_name, "] reducedDim '", reduced_dim_name,
        "' not present in all inputs. Skipping reducedDim merge."
      )
    }
  }

  out
}
