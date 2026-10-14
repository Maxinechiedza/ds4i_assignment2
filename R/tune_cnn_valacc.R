# R/tune_cnn_valacc.R
# Owner: Maxine (Role B) - Task 5 follow-up: does the stopping rule change the depth result?
#
# Why this exists: in R/tune_cnn.R every fit used early stopping on VALIDATION LOSS.
# The 3-layer learning curve showed validation loss rising from epoch 3 (the model
# becomes overconfident) while validation ACCURACY kept improving to about epoch 8.
# So loss-based stopping cut the 3-layer model off early.
#
# This script re-runs ONLY the depth comparison that matters for the final choice
# (2 vs 3 convolution layers, dropout 0.3, learning rate 0.001) with early stopping
# on validation ACCURACY, then compares the stopping rules side by side.
#   2 configurations x 3 seeds = 6 fits. Everything else is identical to tune_cnn.R.
#
# Run 1 used patience 5: one slow 2-layer run (seed 2028) stopped at epoch 8 after
# a lucky spike at epoch 3, because validation accuracy is noisy.
# Run 2 uses patience 10, the same rule as the final models in R/cnn.R.
# Run 3 also raises the epoch cap from 50 to 150: with patience 10 every 2-layer run
# trained all 50 epochs (best epochs 40-48), so it was still improving when cut off.
# Each patience value writes its own file, so earlier results are kept.
# Validation (session 4) only - no test set here.

library(keras3)
use_backend("tensorflow")
source("R/00_data.R")
source("R/cnn_functions.R")

patience   <- 10    # run 1 used 5
max_epochs <- 150   # runs 1 and 2 used 50

data  <- load_split_data()
train <- get_split("train", data)   # sessions 1-3
val   <- get_split("val", data)     # session 4

# ---- 1. The runs -----------------------------------------------------------

configs <- tibble(
  varied        = c("baseline", "n_conv"),
  n_conv        = c(2, 3),
  dropout       = 0.3,
  learning_rate = 1e-3
)
seeds <- c(2026, 2027, 2028)   # same seeds as tune_cnn.R

runs <- crossing(configs, seed = seeds) %>%
  mutate(run_id = paste(varied, n_conv, dropout, learning_rate, seed, sep = "_"))

out_file <- if (max_epochs == 50) sprintf("outputs/tuning_cnn_valacc_p%d.csv", patience) else
  sprintf("outputs/tuning_cnn_valacc_p%d_e%d.csv", patience, max_epochs)
done_ids <- if (file.exists(out_file)) read_csv(out_file, show_col_types = FALSE)$run_id else character(0)

# ---- 2. Fit each run -------------------------------------------------------

for (i in seq_len(nrow(runs))) {
  r <- runs[i, ]
  if (r$run_id %in% done_ids) next

  clear_session()
  set_random_seed(r$seed)

  model <- build_cnn(n_conv = r$n_conv, dropout = r$dropout, learning_rate = r$learning_rate)

  start <- Sys.time()
  history <- model %>% fit(
    train$x, train$y,
    validation_data = list(val$x, val$y),
    epochs = max_epochs, batch_size = 64, shuffle = TRUE,
    callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = patience,
                                             restore_best_weights = TRUE)),
    verbose = 0
  )
  scores <- model %>% evaluate(val$x, val$y, verbose = 0)   # weights from the best epoch

  result <- r %>% mutate(
    stopping     = "val_accuracy",
    patience     = patience,
    max_epochs   = max_epochs,
    val_accuracy = scores$accuracy,
    val_top3     = scores$top3,
    val_loss     = scores$loss,
    epochs_run   = length(history$metrics$loss),
    best_epoch   = which.max(history$metrics$val_accuracy),   # the epoch whose weights were kept
    params       = count_params(model),
    minutes      = round(as.numeric(difftime(Sys.time(), start, units = "mins")), 2)
  )
  write_csv(result, out_file, append = file.exists(out_file))

  message(sprintf("[%d/%d] n_conv=%d seed=%d -> val accuracy %.3f (best epoch %d of %d)",
                  i, nrow(runs), r$n_conv, r$seed, scores$accuracy,
                  result$best_epoch, result$epochs_run))
}

# ---- 3. Compare the stopping rules -----------------------------------------

rule_files <- c(
  "val_loss, patience 5, max 50"       = "outputs/tuning_cnn.csv",
  "val_accuracy, patience 5, max 50"   = "outputs/tuning_cnn_valacc_p5.csv",
  "val_accuracy, patience 10, max 50"  = "outputs/tuning_cnn_valacc_p10.csv",
  "val_accuracy, patience 10, max 150" = "outputs/tuning_cnn_valacc_p10_e150.csv"
)
rule_files <- rule_files[file.exists(rule_files)]

all_runs <- imap_dfr(rule_files, function(file, rule) {
  read_csv(file, show_col_types = FALSE) %>%
    filter(varied %in% c("baseline", "n_conv"), n_conv %in% c(2, 3)) %>%
    mutate(stopping = rule) %>%
    select(stopping, n_conv, seed, val_accuracy, val_top3, epochs_run)
}) %>%
  mutate(stopping = factor(stopping, levels = names(rule_files)))

# Every individual run, so a single failed run is easy to spot
print(all_runs %>% arrange(stopping, n_conv, seed), n = Inf)

# Mean and seed spread for each depth under each stopping rule
by_rule <- all_runs %>%
  group_by(stopping, n_conv) %>%
  summarise(mean_acc  = mean(val_accuracy), sd_acc = sd(val_accuracy),
            mean_top3 = mean(val_top3),     mean_epochs = mean(epochs_run),
            .groups = "drop")
print(by_rule)

# The 3-layer minus 2-layer gap under each rule, with its standard error
# (difference of two means, 3 seeds each)
depth_gap <- by_rule %>%
  group_by(stopping) %>%
  summarise(gap_3_minus_2 = mean_acc[n_conv == 3] - mean_acc[n_conv == 2],
            se_gap        = sqrt(sd_acc[n_conv == 3]^2 / 3 + sd_acc[n_conv == 2]^2 / 3),
            .groups = "drop") %>%
  mutate(gap_over_se = gap_3_minus_2 / se_gap)
print(depth_gap)
