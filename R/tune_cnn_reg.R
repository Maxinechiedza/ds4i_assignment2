# R/tune_cnn_reg.R
# Owner: Maxine (Role B) - Task 5 follow-up: does regularisation help the 3-layer CNN?
#
# Why this exists: dropout was tuned in R/tune_cnn.R on the 2-layer baseline, which
# barely overfits (training 60%, validation 51%). The 3-layer model is the one that
# overfits (training 92%, validation 59.5% at the kept epoch), so this checks
# regularisation on the model we actually use.
#
# 4 settings x 3 seeds = 12 fits, validation (session 4) only - no test set here:
#   dropout 0.3, no L2     the current final model, re-run so every column is comparable
#   dropout 0.5, no L2     stronger dropout before the output layer
#   dropout 0.3, L2 1e-4   L2 weight decay on every convolution and dense layer
#   dropout 0.5, L2 1e-4   both
# Same training rule as the final models: early stopping on validation accuracy,
# patience 10, at most 150 epochs, best weights restored.
#
# Decision rule, fixed BEFORE looking at the results:
#   switch the final model only if a setting beats the current one by more than
#   1 percentage point of mean validation accuracy AND by more than 2 standard
#   errors (seed noise). Otherwise keep dropout 0.3 with no L2.
#
# Results are appended to outputs/tuning_cnn_reg.csv after every fit; re-running
# the script skips finished fits. About 12-15 minutes on a laptop.

library(keras3)
use_backend("tensorflow")
source("R/00_data.R")
source("R/cnn_functions.R")

data  <- load_split_data()
train <- get_split("train", data)   # sessions 1-3
val   <- get_split("val", data)     # session 4

# ---- 1. The runs -----------------------------------------------------------

settings <- tibble(
  setting = c("dropout 0.3, no L2 (current)", "dropout 0.5, no L2",
              "dropout 0.3, L2 1e-4", "dropout 0.5, L2 1e-4"),
  dropout = c(0.3, 0.5, 0.3, 0.5),
  l2      = c(0,   0,   1e-4, 1e-4)
)
seeds <- c(2026, 2027, 2028)   # same seeds as the other tuning scripts

runs <- crossing(settings, seed = seeds) %>%
  mutate(run_id = paste("reg", dropout, l2, seed, sep = "_"))

out_file <- "outputs/tuning_cnn_reg.csv"
done_ids <- if (file.exists(out_file)) read_csv(out_file, show_col_types = FALSE)$run_id else character(0)

# ---- 2. Fit each run -------------------------------------------------------

for (i in seq_len(nrow(runs))) {
  r <- runs[i, ]
  if (r$run_id %in% done_ids) next

  start <- Sys.time()
  fit <- fit_cnn(train, val, seed = r$seed, n_conv = 3,
                 dropout = r$dropout, l2 = r$l2)   # weights from the best epoch
  scores <- fit$model %>% evaluate(val$x, val$y, verbose = 0)

  # Keras adds the L2 penalty to its loss, so loss is not comparable across
  # settings. Cross-entropy from the predicted probabilities is: it measures how
  # much probability the model gives the true writer (lower = less overconfident).
  probs  <- predict(fit$model, val$x, batch_size = 512, verbose = 0)
  p_true <- probs[cbind(seq_len(nrow(probs)), val$meta$person_id)]
  best   <- which.max(fit$history$metrics$val_accuracy)   # the epoch whose weights were kept

  result <- r %>% mutate(
    n_conv         = 3,
    val_accuracy   = scores$accuracy,
    val_top3       = scores$top3,
    val_xent       = mean(-log(pmax(p_true, 1e-7))),
    train_accuracy = fit$history$metrics$accuracy[best],    # with dropout on, as Keras reports it
    gap            = train_accuracy - val_accuracy,         # the overfitting gap
    epochs_run     = length(fit$history$metrics$loss),
    best_epoch     = best,
    minutes        = round(as.numeric(difftime(Sys.time(), start, units = "mins")), 2)
  )
  write_csv(result, out_file, append = file.exists(out_file))

  message(sprintf("[%2d/%d] %-28s seed=%d -> val accuracy %.3f, gap %.3f (best epoch %d of %d)",
                  i, nrow(runs), r$setting, r$seed, result$val_accuracy, result$gap,
                  result$best_epoch, result$epochs_run))
}

# ---- 3. Compare the settings ----------------------------------------------

reg <- read_csv(out_file, show_col_types = FALSE) %>%
  mutate(setting = factor(setting, levels = settings$setting))

# Every individual run, so a single odd run is easy to spot
print(reg %>% arrange(setting, seed) %>%
        select(setting, seed, val_accuracy, val_xent, train_accuracy, gap, best_epoch, epochs_run),
      n = Inf)

by_setting <- reg %>%
  group_by(setting) %>%
  summarise(mean_acc  = mean(val_accuracy), sd_acc = sd(val_accuracy),
            mean_xent = mean(val_xent),     mean_gap = mean(gap),
            mean_epochs = mean(epochs_run), n = n(), .groups = "drop")
print(by_setting)

# Each setting against the current one, with the standard error of the difference
current <- by_setting %>% filter(setting == settings$setting[1])
vs_current <- by_setting %>%
  filter(setting != settings$setting[1]) %>%
  mutate(gain       = mean_acc - current$mean_acc,
         se_gain    = sqrt(sd_acc^2 / n + current$sd_acc^2 / current$n),
         gain_over_se = gain / se_gain,
         passes_rule  = gain > 0.01 & gain_over_se > 2)
print(vs_current %>% select(setting, mean_acc, gain, se_gain, gain_over_se, passes_rule))

winner <- vs_current %>% filter(passes_rule) %>% slice_max(mean_acc, n = 1)
if (nrow(winner) == 0) {
  message("Decision: no setting passes the rule. Keep the current final model (dropout 0.3, no L2).")
} else {
  message("Decision: switch the final model to '", winner$setting, "'. ",
          "In R/cnn.R use fit_final(n_conv = 3, dropout = ..., l2 = ...) with these values, ",
          "and set `final` in R/cnn_digit_session.R to match.")
}

# Check: the current setting re-run should match the saved 3-layer runs in
# outputs/tuning_cnn_valacc_p10_e150.csv (same seeds, same rule)
if (file.exists("outputs/tuning_cnn_valacc_p10_e150.csv")) {
  old <- read_csv("outputs/tuning_cnn_valacc_p10_e150.csv", show_col_types = FALSE) %>%
    filter(n_conv == 3) %>% select(seed, old_val_accuracy = val_accuracy)
  print(reg %>% filter(setting == settings$setting[1]) %>%
          select(seed, val_accuracy) %>% left_join(old, by = "seed"))
}
