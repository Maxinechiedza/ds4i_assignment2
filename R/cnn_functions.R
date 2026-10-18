# R/cnn_functions.R
# Owner: Maxine (Role B)
#
# Shared CNN helpers, sourced by R/cnn.R (Task 2), R/tune_cnn.R, R/tune_cnn_valacc.R,
# R/tune_cnn_reg.R (Task 5) and R/cnn_digit_session.R (Task 6), so every script
# builds exactly the same model.

library(keras3)

# Pull out one split: images (x), one-hot writer labels (y) and its metadata.
# Keras numbers classes 0-19, so writer 1-20 becomes 0-19.
get_split <- function(split_name, data) {
  idx <- which(data$metadata$split_main == split_name)
  list(
    x    = data$images[idx, , , , drop = FALSE],
    y    = to_categorical(data$metadata$person_id[idx] - 1L, num_classes = 20),
    meta = data$metadata[idx, ]
  )
}

# The same, for any set of images: `keep` is TRUE/FALSE for each of the 25,000
# images, e.g. data$metadata$session == 2 or data$metadata$digit == 0.
get_subset <- function(keep, data) {
  idx <- which(keep)
  list(
    x    = data$images[idx, , , , drop = FALSE],
    y    = to_categorical(data$metadata$person_id[idx] - 1L, num_classes = 20),
    meta = data$metadata[idx, ]
  )
}

# Build and compile the writer-identification CNN.
# Defaults (n_conv = 2, dropout = 0.3, learning_rate = 1e-3) reproduce the Day 1 model.
#   n_conv = 1: conv 16
#   n_conv = 2: conv 16 -> pool -> conv 32          (original kickoff backbone, kept as backup)
#   n_conv = 3: conv 16 -> pool -> conv 32 -> pool -> conv 64   (shared backbone with the Siamese net)
# l2 > 0 adds L2 weight decay to every convolution and dense layer (Task 5
# regularisation check). The default l2 = 0 adds nothing, so the saved models
# are rebuilt exactly as before.
build_cnn <- function(n_conv = 2, embedding_units = 32, dropout = 0.3, learning_rate = 1e-3,
                      l2 = 0) {
  reg <- if (l2 > 0) regularizer_l2(l2) else NULL
  input <- layer_input(shape = c(28, 28, 1))

  x <- input %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu",
                  kernel_regularizer = reg)
  if (n_conv >= 2) {
    x <- x %>%
      layer_max_pooling_2d(pool_size = c(2, 2)) %>%
      layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                    kernel_regularizer = reg)
  }
  if (n_conv >= 3) {
    x <- x %>%
      layer_max_pooling_2d(pool_size = c(2, 2)) %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                    kernel_regularizer = reg)
  }

  output <- x %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = embedding_units, activation = "relu", name = "embedding",
                kernel_regularizer = reg) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 20, activation = "softmax", kernel_regularizer = reg)

  model <- keras_model(inputs = input, outputs = output)
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list("accuracy", metric_top_k_categorical_accuracy(k = 3, name = "top3"))
  )
  model
}

# Fit one CNN with the final training rule: early stopping on validation
# accuracy, patience 10, at most 150 epochs, the best epoch's weights restored.
# `train` and `val` come from get_split() or get_subset(). Any other argument
# (n_conv, dropout, learning_rate, l2) goes to build_cnn(); the default n_conv = 3
# is the final model.
fit_cnn <- function(train, val, seed = 2026, n_conv = 3, ..., max_epochs = 150,
                    patience = 10, verbose = 0) {
  clear_session()
  set_random_seed(seed)
  model <- build_cnn(n_conv = n_conv, ...)
  history <- model %>% fit(
    train$x, train$y,
    validation_data = list(val$x, val$y),
    epochs = max_epochs, batch_size = 64, shuffle = TRUE,
    callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = patience,
                                             restore_best_weights = TRUE)),
    verbose = verbose
  )
  list(model = model, history = history)
}

# Learning curves from a training history, drawn only over the epochs that
# actually ran (Keras's own plot() stretches the x axis to the epoch cap).
# `history_df` is as.data.frame(history) or a saved history CSV read back in.
# The dashed line marks the best validation-accuracy epoch - the weights that
# early stopping kept.
plot_history <- function(history_df, title = NULL) {
  df <- dplyr::filter(history_df, !is.na(value))
  best <- df %>%
    dplyr::filter(metric == "accuracy", data == "validation") %>%
    dplyr::slice_max(value, n = 1, with_ties = FALSE) %>%
    dplyr::pull(epoch)
  ggplot2::ggplot(df, ggplot2::aes(epoch, value, colour = data)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 0.8) +
    ggplot2::geom_vline(xintercept = best, linetype = "dashed", colour = "grey50") +
    ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 1) +
    ggplot2::labs(x = "Epoch", y = NULL, colour = NULL, title = title) +
    ggplot2::theme_minimal()
}

# The encoder on its own: one image in, the 32-number embedding out.
# Same layers as build_cnn() up to the layer named "embedding" (no dropout, no
# softmax, not compiled).
# For the Siamese network, share ONE encoder between the two images of a pair:
#   encoder <- build_encoder()        # 3 conv layers: the agreed shared backbone
#   emb_a <- encoder(input_a)
#   emb_b <- encoder(input_b)
# build_encoder(n_conv = 2) gives the original 2-layer backbone (backup).
# l2 works as in build_cnn(); the default 0 adds nothing.
build_encoder <- function(n_conv = 3, embedding_units = 32, l2 = 0) {
  reg <- if (l2 > 0) regularizer_l2(l2) else NULL
  input <- layer_input(shape = c(28, 28, 1))

  x <- input %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu",
                  kernel_regularizer = reg)
  if (n_conv >= 2) {
    x <- x %>%
      layer_max_pooling_2d(pool_size = c(2, 2)) %>%
      layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                    kernel_regularizer = reg)
  }
  if (n_conv >= 3) {
    x <- x %>%
      layer_max_pooling_2d(pool_size = c(2, 2)) %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                    kernel_regularizer = reg)
  }

  output <- x %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = embedding_units, activation = "relu", name = "embedding",
                kernel_regularizer = reg)

  keras_model(inputs = input, outputs = output, name = "encoder")
}
