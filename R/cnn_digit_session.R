# R/cnn_digit_session.R
# Owner: Maxine (Role B) - Task 6: CNN results by session and by digit
#
# Two experiments, both with the final CNN settings in `final` below:
#
# 1. Leave-one-session-out (LOSO): how much does accuracy depend on WHICH day is
#    the new day? Each session is held out in turn as the test set. Early stopping
#    uses the session before it (session 5 when session 1 is held out), and the
#    model trains on the other three. 5 sessions x 3 seeds = 15 fits: test accuracy
#    moves 2-3 points between seeds, so one seed per session could mistake seed
#    noise for a session effect.
#    Holding out session 5 gives exactly the main split (train 1-3, validate 4,
#    test 5), so that fold with seed 2026 should reproduce the main test result
#    (56.6%) on the laptop that made it - a check that the code is right.
#    Caveat for the write-up: the settings were tuned with session 4 as validation,
#    so the fold that TESTS on session 4 is slightly optimistic.
#
# 2. Digit-only CNNs: can ONE digit on its own identify the writer? For each digit,
#    train, validate and test on that digit only, with the main session split
#    (3,000 training images instead of 15,000). 5 digits x 3 seeds = 15 fits;
#    3 seeds because 1,000 test images per digit make one fit noisy.
#    Compare with the full model's accuracy on the same digit (fig-cnn-digit): if
#    the full model does better, the other digits are teaching it about the writer.
#
# Writes outputs/cnn_loso.csv, outputs/cnn_loso_by_digit.csv and
# outputs/cnn_digit_only.csv. Each fit is saved as soon as it finishes, and
# re-running the script skips finished fits. About 20-25 minutes on a laptop.

library(keras3)
use_backend("tensorflow")
source("R/00_data.R")
source("R/cnn_functions.R")

# The final model's settings. If R/tune_cnn_reg.R switches the final model,
# change dropout and l2 here to match R/cnn.R BEFORE running this script.
final <- list(n_conv = 3, dropout = 0.3, learning_rate = 1e-3, l2 = 0)
seeds <- c(2026, 2027, 2028)   # same seeds as the tuning scripts

data <- load_split_data()
meta <- data$metadata

# Fit with the final settings, then score the test images
fit_and_score <- function(train, val, test, seed) {
  start <- Sys.time()
  fit <- do.call(fit_cnn, c(list(train = train, val = val, seed = seed), final))
  test_scores <- fit$model %>% evaluate(test$x, test$y, verbose = 0)
  val_scores  <- fit$model %>% evaluate(val$x, val$y, verbose = 0)
  probs <- predict(fit$model, test$x, batch_size = 512, verbose = 0)
  list(
    summary = tibble(
      seed          = seed,
      n_train       = nrow(train$meta),
      n_test        = nrow(test$meta),
      test_accuracy = test_scores$accuracy,
      test_top3     = test_scores$top3,
      test_loss     = test_scores$loss,
      val_accuracy  = val_scores$accuracy,
      epochs_run    = length(fit$history$metrics$loss),
      best_epoch    = which.max(fit$history$metrics$val_accuracy),
      minutes       = round(as.numeric(difftime(Sys.time(), start, units = "mins")), 2)
    ),
    # column j of the softmax output is writer j
    correct = max.col(probs, ties.method = "first") == test$meta$person_id
  )
}

done <- function(file) if (file.exists(file)) read_csv(file, show_col_types = FALSE) else NULL

# ---- 1. Leave-one-session-out ---------------------------------------------

loso_file       <- "outputs/cnn_loso.csv"
loso_digit_file <- "outputs/cnn_loso_by_digit.csv"

for (k in 1:5) {
  val_session    <- if (k == 1) 5 else k - 1
  train_sessions <- setdiff(1:5, c(k, val_session))

  train <- get_subset(meta$session %in% train_sessions, data)
  val   <- get_subset(meta$session == val_session, data)
  test  <- get_subset(meta$session == k, data)

  for (s in seeds) {
    prev <- done(loso_file)
    if (!is.null(prev) && any(prev$test_session == k & prev$seed == s)) next

    res <- fit_and_score(train, val, test, seed = s)

    # accuracy for each digit within the held-out session (written first, so a
    # crash cannot leave a fold marked as done without its digit rows)
    by_digit <- test$meta %>%
      mutate(correct = res$correct) %>%
      group_by(digit) %>%
      summarise(n = n(), accuracy = mean(correct), .groups = "drop") %>%
      mutate(test_session = k, seed = s, .before = 1)
    write_csv(by_digit, loso_digit_file, append = file.exists(loso_digit_file))

    row <- res$summary %>%
      mutate(test_session = k, val_session = val_session,
             train_sessions = paste(train_sessions, collapse = "+"), .before = 1)
    write_csv(row, loso_file, append = file.exists(loso_file))

    message(sprintf("[LOSO] test session %d, seed %d (validate on %d, train on %s) -> test accuracy %.3f (best epoch %d of %d)",
                    k, s, val_session, row$train_sessions, row$test_accuracy,
                    row$best_epoch, row$epochs_run))
  }
}

# ---- 2. Digit-only CNNs ----------------------------------------------------

digit_file <- "outputs/cnn_digit_only.csv"

for (d in 0:4) {
  for (s in seeds) {
    prev <- done(digit_file)
    if (!is.null(prev) && any(prev$digit == d & prev$seed == s)) next

    train <- get_subset(meta$split_main == "train" & meta$digit == d, data)
    val   <- get_subset(meta$split_main == "val"   & meta$digit == d, data)
    test  <- get_subset(meta$split_main == "test"  & meta$digit == d, data)

    res <- fit_and_score(train, val, test, seed = s)
    row <- res$summary %>% mutate(digit = d, .before = 1)
    write_csv(row, digit_file, append = file.exists(digit_file))

    message(sprintf("[digit %d, seed %d] test accuracy %.3f (best epoch %d of %d)",
                    d, s, row$test_accuracy, row$best_epoch, row$epochs_run))
  }
}

# ---- 3. Summaries ----------------------------------------------------------

# Every fit, then accuracy for each held-out session (mean over seeds)
loso <- read_csv(loso_file, show_col_types = FALSE) %>% arrange(test_session, seed)
print(loso %>% select(test_session, seed, train_sessions, test_accuracy, test_top3,
                      val_accuracy, best_epoch, epochs_run), n = Inf)

by_session <- loso %>%
  group_by(test_session) %>%
  summarise(mean_acc = mean(test_accuracy), sd_acc = sd(test_accuracy),
            mean_top3 = mean(test_top3), .groups = "drop")
print(by_session)

# Is the spread BETWEEN sessions bigger than the spread between seeds?
# (same idea as range / seed noise in Task 5; seed noise pooled over the 5 sessions)
seed_sd <- sqrt(mean(by_session$sd_acc^2))
message(sprintf("Between sessions: range of means %.3f; seed noise (pooled SD) %.3f; range / seed noise = %.1f",
                diff(range(by_session$mean_acc)), seed_sd,
                diff(range(by_session$mean_acc)) / seed_sd))

# Is the digit ranking the same in every held-out session? (1 = best digit, seeds averaged)
loso_digit <- read_csv(loso_digit_file, show_col_types = FALSE) %>%
  group_by(test_session, digit) %>%
  summarise(accuracy = mean(accuracy), .groups = "drop")
print(loso_digit %>%
        group_by(test_session) %>%
        mutate(rank = rank(-accuracy)) %>%
        ungroup() %>%
        select(test_session, digit, rank) %>%
        pivot_wider(names_from = test_session, values_from = rank, names_prefix = "session_"))

# Digit-only models (mean over 3 seeds) next to the full model on the same test images
digit_only <- read_csv(digit_file, show_col_types = FALSE) %>%
  group_by(digit) %>%
  summarise(digit_only_acc = mean(test_accuracy), seed_sd = sd(test_accuracy), .groups = "drop")
if (file.exists("outputs/cnn_test_predictions.csv")) {
  full <- read_csv("outputs/cnn_test_predictions.csv", show_col_types = FALSE) %>%
    group_by(digit) %>%
    summarise(full_model_acc = mean(true_writer == predicted_writer), .groups = "drop")
  digit_only <- left_join(digit_only, full, by = "digit") %>%
    mutate(full_minus_digit_only = full_model_acc - digit_only_acc)
}
print(digit_only)
