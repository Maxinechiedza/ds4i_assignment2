# R/tune_cnn.R
# Owner: Maxine (Role B) - Task 5: CNN hyperparameter sensitivity
# Design: one-at-a-time around the Day 1 baseline.
#   Baseline: n_conv = 2, dropout = 0.3, learning_rate = 0.001
#   Vary ONE hyperparameter at a time, keep the other two at baseline:
#     convolution layers: 1, 2, 3
#     learning rate:      0.0003, 0.001, 0.003
#     dropout:            0, 0.3, 0.5
#   7 configurations x 3 seeds = 21 fits 
# Results are appended to outputs/tuning_cnn.csv after every fit
library(keras3)
use_backend("tensorflow")
source("R/00_data.R")
source("R/cnn_functions.R")

data  <- load_split_data()
train <- get_split("train", data)   # sessions 1-3
val   <- get_split("val", data)     # session 4

# ---- 1. The runs -----------------------------------------------------------

configs <- bind_rows(
  tibble(varied = "baseline",      n_conv = 2,       dropout = 0.3,      learning_rate = 1e-3),
  tibble(varied = "n_conv",        n_conv = c(1, 3), dropout = 0.3,      learning_rate = 1e-3),
  tibble(varied = "learning_rate", n_conv = 2,       dropout = 0.3,      learning_rate = c(3e-4, 3e-3)),
  tibble(varied = "dropout",       n_conv = 2,       dropout = c(0, 0.5), learning_rate = 1e-3)
)
seeds <- c(2026, 2027, 2028)

runs <- crossing(configs, seed = seeds) %>%
  mutate(run_id = paste(varied, n_conv, dropout, learning_rate, seed, sep = "_"))

out_file <- "outputs/tuning_cnn.csv"
done_ids <- if (file.exists(out_file)) read_csv(out_file, show_col_types = FALSE)$run_id else character(0)

# ---- 2. Fit each run -------------------------------------------------------

for (i in seq_len(nrow(runs))) {
  r <- runs[i, ]
  if (r$run_id %in% done_ids) next

  clear_session()              # start each fit from a clean slate
  set_random_seed(r$seed)      # same seed -> same starting weights and batch order

  model <- build_cnn(n_conv = r$n_conv, dropout = r$dropout, learning_rate = r$learning_rate)

  start <- Sys.time()
  history <- model %>% fit(
    train$x, train$y,
    validation_data = list(val$x, val$y),
    epochs = 50, batch_size = 64, shuffle = TRUE,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 5,
                                             restore_best_weights = TRUE)),
    verbose = 0
  )
  scores <- model %>% evaluate(val$x, val$y, verbose = 0)   # best epoch's weights

  result <- r %>% mutate(
    val_accuracy = scores$accuracy,
    val_top3     = scores$top3,
    val_loss     = scores$loss,
    epochs_run   = length(history$metrics$loss),
    params       = count_params(model),
    minutes      = round(as.numeric(difftime(Sys.time(), start, units = "mins")), 2)
  )
  write_csv(result, out_file, append = file.exists(out_file))

  message(sprintf("[%2d/%d] %-13s n_conv=%d dropout=%.1f lr=%g seed=%d -> val accuracy %.3f (%d epochs)",
                  i, nrow(runs), r$varied, r$n_conv, r$dropout, r$learning_rate, r$seed,
                  scores$accuracy, result$epochs_run))
}

# ---- 3. Summarise ----------------------------------------------------------

tune <- read_csv(out_file, show_col_types = FALSE)

# The baseline runs are the middle value of every hyperparameter, so each
# hyperparameter's comparison = its own runs + the baseline runs.
tune_long <- bind_rows(
  tune %>% filter(varied %in% c("baseline", "n_conv"))        %>% mutate(hyperparameter = "Convolution layers", value = n_conv),
  tune %>% filter(varied %in% c("baseline", "learning_rate")) %>% mutate(hyperparameter = "Learning rate",      value = learning_rate),
  tune %>% filter(varied %in% c("baseline", "dropout"))       %>% mutate(hyperparameter = "Dropout rate",       value = dropout)
)

by_value <- tune_long %>%
  group_by(hyperparameter, value) %>%
  summarise(mean_acc = mean(val_accuracy), sd_acc = sd(val_accuracy),
            mean_epochs = mean(epochs_run), .groups = "drop")
print(by_value)

# Seed noise: the spread you get from re-running the SAME configuration.
# Pooled over all 7 configurations, so it rests on 21 fits rather than 3.
seed_sd <- tune %>%
  group_by(varied, n_conv, dropout, learning_rate) %>%
  summarise(v = var(val_accuracy), .groups = "drop") %>%
  summarise(sqrt(mean(v))) %>% pull()

sensitivity <- by_value %>%
  group_by(hyperparameter) %>%
  summarise(best_value  = value[which.max(mean_acc)],
            range_of_means = max(mean_acc) - min(mean_acc), .groups = "drop") %>%
  mutate(seed_sd = seed_sd, range_over_noise = range_of_means / seed_sd)
print(sensitivity)
