# R/cnn.R
# Owner: Maxine (Role B) - Task 2: CNN for writer identification

# The task: given ONE 28 x 28 image, predict which of the 20 known writers wrote it.
# Train on sessions 1-3, session 4 (validation) for early stopping and model choices,
# session 5 (test) used ONCE per final model.
#
# Final models (chosen in Task 5 on validation only - see R/tune_cnn.R and R/tune_cnn_valacc.R):
#   cnn3 - 3 convolution layers: the best CNN, used for the Task 2 results, and (agreed
#          with Nicole) the backbone shared with her Siamese network, so it is
#          also the CNN in Jesse's like-for-like comparison in Task 6
#   cnn2 - 2 convolution layers: the original kickoff backbone. Kept as a backup in case
#          the Siamese network goes back to 2 layers, and as the smaller model in Task 2
# Both stop on validation ACCURACY with patience 10, with up to 150 epochs: validation
# loss stopped the 3-layer model too early, patience 5 stopped one slow 2-layer run
# too early, and a 50-epoch cap cut the slow-learning 2-layer model off while it was
# still improving.

library(keras3)
use_backend("tensorflow")
library(dplyr)

# Loading the data
source("R/00_data.R")
source("R/cnn_functions.R")

set.seed(2026)
set_random_seed(2026)  # just in case for the python

# ---- 1. Data ---------------------------------------------------------------

data <- load_split_data()

train <- get_split("train", data)   # sessions 1-3
val   <- get_split("val", data)     # session 4
test  <- get_split("test", data)    # session 5

dim(train$x)       # 15000 28 28 1
colSums(train$y)   # 750 images per writer

# ---- 2. Learning-curve check (Task 5 follow-up) ----------------------------
# The 3-layer model trained for 30 epochs WITHOUT early stopping, to see how
# validation loss and accuracy behave. Already run; set to TRUE to repeat it.
# The saved history can be plotted in the report.
run_curve_check <- FALSE
if (run_curve_check) {
  set_random_seed(2026)
  m3 <- build_cnn(n_conv = 3)
  h3 <- m3 %>% fit(train$x, train$y, validation_data = list(val$x, val$y),
                   epochs = 30, batch_size = 64, verbose = 2)
  plot(h3)
  write_csv(as.data.frame(h3), "outputs/cnn3_curve_check.csv")
}

# ---- 3. Final models -------------------------------------------------------

fit_final <- function(n_conv, seed = 2026) {
  clear_session()
  set_random_seed(seed)
  model <- build_cnn(n_conv = n_conv)
  history <- model %>% fit(
    train$x, train$y,
    validation_data = list(val$x, val$y),
    epochs = 150, batch_size = 64, shuffle = TRUE,
    callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = 10,
                                             restore_best_weights = TRUE)),
    verbose = 2
  )
  list(model = model, history = history)
}

cnn3 <- fit_final(n_conv = 3)   # best CNN and shared backbone with Nicole's Siamese network
cnn2 <- fit_final(n_conv = 2)   # backup: the original 2-layer backbone

plot_history(as.data.frame(cnn3$history), "3 convolution layers")
plot_history(as.data.frame(cnn2$history), "2 convolution layers")

# ---- 4. Test set: ONE evaluation per model --------------------------------

test_scores <- bind_rows(
  as_tibble(evaluate(cnn3$model, test$x, test$y, verbose = 0)) %>% mutate(model = "cnn3", .before = 1),
  as_tibble(evaluate(cnn2$model, test$x, test$y, verbose = 0)) %>% mutate(model = "cnn2", .before = 1)
)
test_scores

# One row per test image: the true writer, the predicted writer, the probability
# given to the true writer, and whether the true writer is in the top 3.
# Column j of the softmax output is class j - 1, i.e. writer j.
predict_writers <- function(model, split) {
  probs <- predict(model, split$x, batch_size = 512, verbose = 0)
  top3  <- t(apply(probs, 1, function(p) order(p, decreasing = TRUE)[1:3]))
  true  <- split$meta$person_id
  tibble(
    image_id         = split$meta$image_id,
    digit            = split$meta$digit,
    true_writer      = true,
    predicted_writer = max.col(probs, ties.method = "first"),
    prob_true        = probs[cbind(seq_along(true), true)],
    top3_hit         = top3[, 1] == true | top3[, 2] == true | top3[, 3] == true
  )
}

# ---- 5. Save ---------------------------------------------------------------

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)

write_csv(test_scores, "outputs/cnn_test_scores.csv")
write_csv(predict_writers(cnn3$model, test), "outputs/cnn_test_predictions.csv")    # Task 2 results
write_csv(predict_writers(cnn2$model, test), "outputs/cnn2_test_predictions.csv")
write_csv(as.data.frame(cnn3$history), "outputs/cnn_history.csv")                  # read by index.qmd
write_csv(as.data.frame(cnn2$history), "outputs/cnn2_history.csv")

save_model(cnn3$model, "outputs/models/cnn3_final.keras", overwrite = TRUE)        # git-ignored
save_model(cnn2$model, "outputs/models/cnn2_final.keras", overwrite = TRUE)        # git-ignored

# ---- 6. Embeddings for Jesse (Task 6) -------------------------------------
# The 32 numbers from the layer named "embedding", for ALL 25 000 images
# One row per image, same order as data/splits.csv, columns image_id, e01 ... e32.

export_embeddings <- function(model, file) {
  embedder <- keras_model(inputs = model$input,
                          outputs = get_layer(model, "embedding")$output)
  emb <- predict(embedder, data$images, batch_size = 512, verbose = 0)
  colnames(emb) <- sprintf("e%02d", seq_len(ncol(emb)))
  out <- bind_cols(tibble(image_id = data$metadata$image_id), as_tibble(round(emb, 5)))
  write_csv(out, file)
  invisible(out)
}

emb3 <- export_embeddings(cnn3$model, "outputs/cnn3_embeddings.csv")   # shared backbone: compare with Siamese
emb2 <- export_embeddings(cnn2$model, "outputs/cnn2_embeddings.csv")   # backup, if Siamese goes back to 2 layers
dim(emb3)   # 25000 33
