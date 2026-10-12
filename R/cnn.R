# R/cnn.R
# Owner: Maxine (Role B) - Task 2: CNN for writer identification

# Day 1: set-up, task definition and a first CNN trained on the main split.

# The task: given ONE 28 x 28 image, predict which of the 20 known writer wrote it, train on sessions 1-3,
# session 4 (validation) for early stopping and model choices
# session 5 (test) untouched

library(keras3)
use_backend("tensorflow")
library(dplyr)

# Loading the data
source("R/00_data.R") 

set.seed(2026)
set_random_seed(2026)  # just in case for the python

# ---- 1. Data ---------------------------------------------------------------

data <- load_split_data()
meta <- data$metadata

# Data preprocessing
# Pull out one split: images (x), one-hot writer labels (y) and its metadata.
# Keras classes numbered 0-19
get_split <- function(split_name) {
  idx <- which(meta$split_main == split_name)
  list(
    x    = data$images[idx, , , , drop = FALSE],
    y    = to_categorical(meta$person_id[idx] - 1L, num_classes = 20), # data preprocessing for target
    meta = meta[idx, ]
  )
}

train <- get_split("train")   # sessions 1-3
val   <- get_split("val")     # session 4
test  <- get_split("test")    # session 5

dim(train$x)       # 15000 28 28 1
colSums(train$y)   # 750 images per writer

# ---- 2. Model --------------------------------------------------------------

# Taken from the lecture's Siamese encoder
build_cnn <- function(embedding_units = 32, dropout = 0.3, learning_rate = 1e-3) {
  input <- layer_input(shape = c(28, 28, 1))

  output <- input %>%
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = embedding_units, activation = "relu", name = "embedding") %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 20, activation = "softmax")

  model <- keras_model(inputs = input, outputs = output)

  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list("accuracy", metric_top_k_categorical_accuracy(k = 3, name = "top3"))
  )
  model
}

model <- build_cnn()
summary(model)

# ---- 3. Train --------------------------------------------------------------

history <- model %>% fit(
  train$x, train$y,
  validation_data = list(val$x, val$y), 
  epochs = 50, batch_size = 64, shuffle = TRUE,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5,
                            restore_best_weights = TRUE)
  ),
  verbose = 2
)

plot(history)

# Validation performance 
model %>% evaluate(val$x, val$y, verbose = 0)

# ---- 4. Save ---------------------------------------------------------------

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)
write_csv(as.data.frame(history), "outputs/cnn_history.csv")              # committed
save_model(model, "outputs/models/cnn_day1.keras", overwrite = TRUE)      # git-ignored
