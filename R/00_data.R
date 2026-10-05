# Owner: Jesse (Role A)
# Loads handwriting.rds, scales images, and writes the two shared hand-off
# files: data/splits.csv and data/test_pairs.csv.
#
# Source this file to reuse load_handwriting()/make_splits()/make_test_pairs()
# without re-writing the CSVs (the writes only happen in the block guarded by
# `is_main`, at the bottom).

library(dplyr)
library(readr)
library(tidyr)
library(purrr)

SEED <- 2026

# ---- load -------------------------------------------------------------

#' Read handwriting.rds and scale pixel values to [0, 1].
load_handwriting <- function(path = "data/handwriting.rds") {
  raw <- readRDS(path)
  list(
    images = raw$images / 255,
    metadata = raw$metadata
  )
}

#' One-call loader for B and C: everything needed to start modelling.
#'
#' Reads the already-committed data/splits.csv rather than regenerating it --
#' don't call make_splits() yourself, the writer/session assignment must be
#' identical for everyone, and it's already fixed by the seed and committed.
#'
#' Usage (after `source("R/00_data.R")`):
#'   data <- load_split_data()
#'   train_idx <- which(data$metadata$split_main == "train")
#'   x_train <- data$images[train_idx, , , , drop = FALSE]
#'   y_train <- data$metadata$person_id[train_idx] - 1L   # keras wants 0-indexed labels
#'
#' Returns a list:
#'   images   -- 25000 x 28 x 28 x 1 array, scaled to [0, 1], same row order as metadata
#'   metadata -- image_id, person_id, digit, session, replicate, split_main, split_writer
load_split_data <- function(images_path = "data/handwriting.rds", splits_path = "data/splits.csv") {
  hw <- load_handwriting(images_path)
  splits <- read_csv(splits_path, show_col_types = FALSE)
  stopifnot(nrow(splits) == dim(hw$images)[1])
  list(images = hw$images, metadata = splits)
}

# ---- splits -------------------------------------------------------------

#' Build the shared split file.
#'
#' split_main: session-based generalization (a known writer, new day).
#'   sessions 1-3 = train, 4 = validation, 5 = test. Used by both the CNN
#'   and the Siamese network for their primary train/val/test split.
#'
#' split_writer: writer-based generalization (a person the model has never
#'   seen). 20 writers split 14 train / 3 val / 3 test. Only meaningful for
#'   the Siamese network's unseen-writer experiment (the CNN is a fixed
#'   20-class classifier and can't be evaluated on writers excluded from
#'   training).
make_splits <- function(metadata, seed = SEED) {
  set.seed(seed)

  main_lookup <- c(`1` = "train", `2` = "train", `3` = "train",
                    `4` = "val", `5` = "test")

  writer_ids <- sort(unique(metadata$person_id))
  stopifnot(length(writer_ids) == 20)
  shuffled <- sample(writer_ids)
  writer_split <- c(
    setNames(rep("train", 14), shuffled[1:14]),
    setNames(rep("val", 3), shuffled[15:17]),
    setNames(rep("test", 3), shuffled[18:20])
  )

  metadata %>%
    mutate(
      split_main = unname(main_lookup[as.character(session)]),
      split_writer = unname(writer_split[as.character(person_id)])
    ) %>%
    select(image_id, person_id, digit, session, replicate, split_main, split_writer)
}

# ---- test pairs -------------------------------------------------------------

#' Sample `n` same-writer and `n` different-writer pairs of image_ids from
#' `query_pool` (paired with itself, same-session) or against
#' `partner_pool` (cross-session), optionally restricted to one digit.
.sample_pairs <- function(query_pool, partner_pool, n_pos, n_neg, digit_fixed = TRUE) {
  by_writer_q <- split(query_pool$image_id, query_pool$person_id)
  by_writer_p <- split(partner_pool$image_id, partner_pool$person_id)
  writers <- intersect(names(by_writer_q), names(by_writer_p))

  pos <- map_dfr(seq_len(n_pos), function(i) {
    w <- sample(writers, 1)
    id1 <- sample(by_writer_q[[w]], 1)
    pool2 <- by_writer_p[[w]]
    # avoid pairing an image with itself when query/partner pools coincide
    pool2 <- pool2[pool2 != id1]
    if (length(pool2) == 0) pool2 <- by_writer_p[[w]]
    id2 <- sample(pool2, 1)
    tibble(id_1 = id1, id_2 = id2, same_writer = TRUE)
  })

  neg <- map_dfr(seq_len(n_neg), function(i) {
    w2 <- sample(writers, 2)
    id1 <- sample(by_writer_q[[w2[1]]], 1)
    id2 <- sample(by_writer_p[[w2[2]]], 1)
    tibble(id_1 = id1, id_2 = id2, same_writer = FALSE)
  })

  bind_rows(pos, neg)
}

#' Build the shared test-pairs file for evaluating same-writer matching.
#'
#' Every query image is drawn from session 5 (the held-out test session).
#' Three pair families, each balanced same/different writer:
#'   - same-session: partner also from session 5, same digit.
#'   - cross-session: partner from sessions 1-4, same digit (the realistic
#'     "check a new sample against enrolled ones" scenario).
#'   - cross-digit (smaller): partner from any session, a DIFFERENT digit,
#'     to test whether writer identity carries across digits.
#'
#' n_per_digit applies to each of same-session/cross-session per digit
#' (so total same-digit pairs = 5 digits x n_per_digit x 2 families).
#' n_cross_digit_total is split evenly across digit pairs.
make_test_pairs <- function(metadata, seed = SEED, n_per_digit = 100, n_cross_digit_total = 400) {
  set.seed(seed + 1)
  session5 <- filter(metadata, session == 5)
  sessions1to4 <- filter(metadata, session %in% 1:4)

  same_digit_pairs <- map_dfr(0:4, function(d) {
    pool_q <- filter(session5, digit == d)
    same_sess <- .sample_pairs(pool_q, pool_q, n_per_digit / 2, n_per_digit / 2) %>%
      mutate(same_digit = TRUE, same_session = TRUE)
    pool_cross <- filter(sessions1to4, digit == d)
    cross_sess <- .sample_pairs(pool_q, pool_cross, n_per_digit / 2, n_per_digit / 2) %>%
      mutate(same_digit = TRUE, same_session = FALSE)
    bind_rows(same_sess, cross_sess)
  })

  n_per_combo <- ceiling(n_cross_digit_total / (5 * 4))
  cross_digit_pairs <- map_dfr(0:4, function(d1) {
    map_dfr(setdiff(0:4, d1), function(d2) {
      pool_q <- filter(session5, digit == d1)
      pool_p <- filter(metadata, digit == d2)
      .sample_pairs(pool_q, pool_p, ceiling(n_per_combo / 2), ceiling(n_per_combo / 2)) %>%
        mutate(
          same_digit = FALSE,
          same_session = NA # mixed; not a meaningful axis for cross-digit pairs
        )
    })
  })

  bind_rows(same_digit_pairs, cross_digit_pairs) %>%
    distinct(id_1, id_2, .keep_all = TRUE) %>%
    select(id_1, id_2, same_writer, same_digit, same_session)
}

# ---- run as a script -------------------------------------------------------------

is_main <- identical(environment(), globalenv()) && sys.nframe() == 0

if (is_main) {
  hw <- load_handwriting("data/handwriting.rds")

  splits <- make_splits(hw$metadata)
  write_csv(splits, "data/splits.csv")
  message("Wrote data/splits.csv (", nrow(splits), " rows)")

  pairs <- make_test_pairs(hw$metadata)
  write_csv(pairs, "data/test_pairs.csv")
  message("Wrote data/test_pairs.csv (", nrow(pairs), " rows)")
}
