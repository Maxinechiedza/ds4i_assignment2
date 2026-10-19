# Owner: Jesse (Role A)
# Task 6: PCA baseline, CNN and Siamese embeddings, scored on the same
# test pairs (data/test_pairs.csv) so the three are directly comparable.
#
# For each embedding space we compute, from pairwise Euclidean distance:
#   - overall ROC AUC and equal error rate (EER) for same-writer matching
#   - AUC by digit (same-digit pairs only)
#   - mean distance for same-writer pairs, same-session vs cross-session
# plus nearest-centroid writer accuracy on the session-held-out test split
# (session 5): classify each test image to the closest training-writer
# centroid in embedding space. This gives the CNN and Siamese embeddings
# an identification-style score they weren't directly trained for, so all
# three spaces are judged the same way.
#
# AUC/EER follow the same definitions Nicole used for the Siamese network
# in Siamese.R, so the numbers are comparable across the report.
#
# Reads: outputs/cnn3_embeddings.csv, outputs/siamese_embeddings.csv,
#        data/test_pairs.csv, data/splits.csv (via R/00_data.R).
# Writes: outputs/comparison.csv (long format: model, scope, metric, value, n_pairs).

library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(pROC)

source("R/00_data.R")

SEED <- 2026
PCA_FIT_SAMPLE <- 5000 # images used to fit the PCA baseline (matches the Task 1 EDA sample size)
EMBED_DIM <- 32 # matches the CNN and Siamese embedding size

# ---- embeddings -----------------------------------------------------------

#' PCA baseline: fit on a sample of training images, project every image
#' onto the top EMBED_DIM components. Fitting on a sample (not all 15,000
#' training images) keeps this fast; it's a baseline, not a model we tune.
pca_embeddings <- function(images, metadata, seed = SEED) {
  set.seed(seed)
  train_idx <- which(metadata$split_main == "train")
  fit_idx <- sample(train_idx, PCA_FIT_SAMPLE)

  flatten <- function(idx) matrix(images[idx, , , 1], nrow = length(idx))

  pc <- prcomp(flatten(fit_idx), center = TRUE, scale. = FALSE, rank. = EMBED_DIM)

  all_idx <- seq_len(nrow(metadata))
  proj <- predict(pc, flatten(all_idx))
  colnames(proj) <- sprintf("e%02d", seq_len(EMBED_DIM))

  as_tibble(proj) %>% mutate(image_id = metadata$image_id[all_idx], .before = 1)
}

load_embeddings <- function(data) {
  list(
    pca = pca_embeddings(data$images, data$metadata),
    cnn = read_csv("outputs/cnn3_embeddings.csv", show_col_types = FALSE),
    siamese = read_csv("outputs/siamese_embeddings.csv", show_col_types = FALSE)
  )
}

# ---- pairwise distance -----------------------------------------------------------

#' Euclidean distance between embedding(id_1) and embedding(id_2) for every
#' row of `pairs`. Embeddings are matched to pairs by image_id, so order in
#' the embeddings file doesn't matter.
pair_distances <- function(embeddings, pairs) {
  emb_cols <- setdiff(names(embeddings), "image_id")
  emb_mat <- as.matrix(embeddings[emb_cols])
  rownames(emb_mat) <- embeddings$image_id

  left <- emb_mat[as.character(pairs$id_1), , drop = FALSE]
  right <- emb_mat[as.character(pairs$id_2), , drop = FALSE]
  sqrt(rowSums((left - right)^2))
}

# ---- metrics -----------------------------------------------------------

#' ROC AUC for same_writer vs distance (shorter distance = more likely a
#' match), and the equal error rate, found the same way as Siamese.R: scan
#' every observed distance as a candidate threshold and take the one where
#' the false-positive and false-negative rates are closest.
auc_and_eer <- function(distance, same_writer) {
  y <- as.integer(same_writer)
  auc <- as.numeric(pROC::auc(pROC::roc(
    response = y, predictor = -distance, levels = c(0, 1), direction = "<", quiet = TRUE
  )))

  thresholds <- sort(unique(distance))
  rates <- sapply(thresholds, function(t) {
    pred <- as.integer(distance <= t)
    c(fpr = mean(pred[y == 0] == 1), fnr = mean(pred[y == 1] == 0))
  })
  idx <- which.min(abs(rates["fpr", ] - rates["fnr", ]))
  eer <- mean(rates[, idx])

  list(auc = auc, eer = eer)
}

#' Classify each test-split image to its nearest training-writer centroid.
nearest_centroid_accuracy <- function(embeddings, metadata) {
  emb_cols <- setdiff(names(embeddings), "image_id")
  joined <- embeddings %>% inner_join(metadata, by = "image_id")

  centroids <- joined %>%
    filter(split_main == "train") %>%
    group_by(person_id) %>%
    summarise(across(all_of(emb_cols), mean), .groups = "drop")
  centroid_mat <- as.matrix(centroids[emb_cols])

  test_df <- filter(joined, split_main == "test")
  test_mat <- as.matrix(test_df[emb_cols])

  predicted <- apply(test_mat, 1, function(v) {
    d <- sqrt(rowSums(sweep(centroid_mat, 2, v, "-")^2))
    centroids$person_id[which.min(d)]
  })

  mean(predicted == test_df$person_id)
}

# ---- run as a script -------------------------------------------------------------

is_main <- identical(environment(), globalenv()) && sys.nframe() == 0

if (is_main) {
  data <- load_split_data()
  pairs <- read_csv("data/test_pairs.csv", show_col_types = FALSE)

  # digit of id_1 (both images share a digit whenever same_digit is TRUE)
  digit_lookup <- setNames(data$metadata$digit, data$metadata$image_id)
  pairs$digit <- unname(digit_lookup[as.character(pairs$id_1)])

  embeddings <- load_embeddings(data)

  rows <- list()

  for (model_name in names(embeddings)) {
    emb <- embeddings[[model_name]]
    dist_all <- pair_distances(emb, pairs)

    overall <- auc_and_eer(dist_all, pairs$same_writer)
    nc_acc <- nearest_centroid_accuracy(emb, data$metadata)

    rows[[length(rows) + 1]] <- tibble(
      model = model_name, scope = "overall",
      metric = c("auc", "eer", "nc_accuracy"),
      value = c(overall$auc, overall$eer, nc_acc),
      n_pairs = nrow(pairs)
    )

    # AUC by digit, same-digit pairs only
    for (d in sort(unique(pairs$digit[pairs$same_digit]))) {
      idx <- pairs$same_digit & pairs$digit == d
      digit_auc <- auc_and_eer(dist_all[idx], pairs$same_writer[idx])
      rows[[length(rows) + 1]] <- tibble(
        model = model_name, scope = paste0("digit:", d),
        metric = "auc", value = digit_auc$auc, n_pairs = sum(idx)
      )
    }

    # Mean distance for same-writer pairs, same session vs cross session
    # (cross-digit pairs have same_session = NA and are excluded here).
    for (sess in c("same", "cross")) {
      idx <- pairs$same_writer & pairs$same_digit &
        pairs$same_session == (sess == "same") & !is.na(pairs$same_session)
      rows[[length(rows) + 1]] <- tibble(
        model = model_name, scope = paste0("session:", sess),
        metric = "mean_distance", value = mean(dist_all[idx]), n_pairs = sum(idx)
      )
    }
  }

  comparison <- bind_rows(rows)
  write_csv(comparison, "outputs/comparison.csv")
  message("Wrote outputs/comparison.csv")
  print(comparison %>% filter(scope == "overall"))
}
