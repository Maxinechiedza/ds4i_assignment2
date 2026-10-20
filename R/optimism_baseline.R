# Owner: Jesse (Role A)
# Task 4: how much better does a simple baseline look on a random image
# split than on the session-held-out split (split_main)?
#
# Baseline model: k-NN on five simple per-image features (ink, width,
# height, centroid shift, slant -- the same features used in the Task 1
# EDA), predicting person_id out of 20 writers. The model itself is not
# the point; it's a cheap, fast stand-in used purely to expose how much
# a session-based split costs you in apparent accuracy versus a split
# that lets near-duplicate replicates of the same writer/session leak
# between train and test.
#
# Writes outputs/optimism_baseline.csv: one row per split with accuracy.

library(dplyr)
library(readr)
library(class)

source("R/00_data.R")

SEED <- 2026
K_NEIGHBOURS <- 15

# ---- features -----------------------------------------------------------

#' Five simple features per image: ink amount, bounding-box width/height,
#' distance of the centre of mass from the image centre, and a slant
#' estimate (same definitions as the Task 1 EDA figure).
extract_features <- function(images, metadata) {
  rows_idx <- 1:28
  cols_idx <- 1:28
  n_img <- dim(images)[1]

  feat <- tibble(
    image_id = metadata$image_id,
    ink = NA_real_, width = NA_real_, height = NA_real_,
    com_x = NA_real_, com_y = NA_real_, slant = NA_real_
  )

  for (i in seq_len(n_img)) {
    m <- images[i, , , 1]
    ink <- sum(m)
    rs <- which(rowSums(m) > 0)
    cs <- which(colSums(m) > 0)
    feat$ink[i] <- ink
    feat$width[i] <- if (length(cs)) max(cs) - min(cs) + 1 else NA
    feat$height[i] <- if (length(rs)) max(rs) - min(rs) + 1 else NA
    cx <- sum(m * matrix(cols_idx, 28, 28, byrow = TRUE)) / ink
    cy <- sum(m * matrix(rows_idx, 28, 28)) / ink
    feat$com_x[i] <- cx
    feat$com_y[i] <- cy
    mu11 <- sum(m * outer(rows_idx - cy, cols_idx - cx)) / ink
    mu02 <- sum(m * (rows_idx - cy)^2) / ink
    feat$slant[i] <- if (mu02 > 0) mu11 / mu02 else NA
  }

  feat %>%
    mutate(centroid_shift = sqrt((com_x - 14.5)^2 + (com_y - 14.5)^2)) %>%
    select(image_id, ink, width, height, centroid_shift, slant) %>%
    left_join(metadata, by = "image_id")
}

# ---- baseline model -------------------------------------------------------------

#' Fit a k-NN classifier on `train` features (standardised using train's own
#' mean/sd) and return test accuracy on `test`.
knn_accuracy <- function(train, test, feature_cols, k = K_NEIGHBOURS, seed = SEED) {
  mu <- colMeans(train[feature_cols])
  sdv <- apply(train[feature_cols], 2, sd)

  scale_with <- function(df) sweep(sweep(as.matrix(df[feature_cols]), 2, mu, "-"), 2, sdv, "/")

  set.seed(seed) # class::knn breaks ties at random
  pred <- knn(
    train = scale_with(train),
    test = scale_with(test),
    cl = factor(train$person_id),
    k = k
  )
  mean(as.character(pred) == as.character(test$person_id))
}

# ---- run as a script -------------------------------------------------------------

is_main <- identical(environment(), globalenv()) && sys.nframe() == 0

if (is_main) {
  data <- load_split_data()
  feat <- extract_features(data$images, data$metadata)
  feature_cols <- c("ink", "width", "height", "centroid_shift", "slant")

  # Session-held-out split: the real evaluation everyone else uses.
  # Train = sessions 1-3, test = session 5 (a known writer, a new day).
  session_train <- filter(feat, split_main == "train")
  session_test <- filter(feat, split_main == "test")
  session_acc <- knn_accuracy(session_train, session_test, feature_cols)

  # Random split: same train/test sizes, but images assigned to train/test
  # ignoring session entirely, so near-duplicate replicates of the same
  # writer/session/digit can land on both sides.
  set.seed(SEED)
  shuffled_ids <- sample(feat$image_id)
  n_train <- nrow(session_train)
  n_test <- nrow(session_test)
  random_train <- feat %>% filter(image_id %in% shuffled_ids[1:n_train])
  random_test <- feat %>% filter(image_id %in% shuffled_ids[(n_train + 1):(n_train + n_test)])
  random_acc <- knn_accuracy(random_train, random_test, feature_cols)

  results <- tibble(
    split = c("random", "session"),
    description = c(
      "Random image split (ignores session; replicate leakage possible)",
      "Session-held-out split (sessions 1-3 train, session 5 test)"
    ),
    n_train = c(nrow(random_train), nrow(session_train)),
    n_test = c(nrow(random_test), nrow(session_test)),
    accuracy = c(random_acc, session_acc)
  )

  write_csv(results, "outputs/optimism_baseline.csv")
  message("Wrote outputs/optimism_baseline.csv")
  print(results)
}
