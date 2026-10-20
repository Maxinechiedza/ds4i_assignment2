### TASK 5: SIAMESE NETWORK HYPERPARAMETER TUNING

#Setup
library(reticulate)
use_virtualenv("siamese-env", required = TRUE)

library(keras3)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(pROC)

use_backend("tensorflow")

#for reproducibility
set.seed(2026)
keras3::set_random_seed(2026)

#Loading the data
source("R/00_data.R")
data <- load_split_data()


#Using the same pair generating code as in the main Siamese network code, so that the same objects are being generated

#identifying the train and validation images that Jess did
x <- data$images
meta <- data$metadata

train_ids <- which(meta$split_main == "train")
valid_ids <- which(meta$split_main == "val")
test_ids  <- which(meta$split_main == "test")

length(train_ids)
length(valid_ids)
length(test_ids)



#creating the pair-generation function by adapting it from the slides
make_pairs <- function(ids, n_pairs, meta, cross_session_positive = TRUE) {
  
  pairs <- matrix(NA_integer_, nrow = n_pairs, ncol = 2)
  
  #exactly half positive and half negative
  same <- rep(c(TRUE, FALSE), length.out = n_pairs)
  same <- sample(same)
  
  for (k in seq_len(n_pairs)) {
    
    #random first image
    i <- sample(ids, 1)
    
    #restricting the candidates to the same digit
    pool <- ids[meta$digit[ids] == meta$digit[i]]
    
    if (same[k]) {
      
      #positive: the same writer
      pool <- pool[meta$person_id[pool] == meta$person_id[i]]
      
      #prefering a different session when possible
      if (cross_session_positive) {
        cross_pool <- pool[meta$session[pool] != meta$session[i]]
        
        if (length(cross_pool) > 0) {
          pool <- cross_pool
        }
      }
      
    } else {
      
      #negative: a different writer
      pool <- pool[meta$person_id[pool] != meta$person_id[i]]
    }
    
    #we can never pair an image with itself, so force a stop
    pool <- setdiff(pool, i)
    
    if (length(pool) == 0) {
      stop("No eligible partner for image ", i)
    }
    
    j <- sample(pool, 1)
    
    pairs[k, ] <- c(i, j)
  }
  
  colnames(pairs) <- c("id_1", "id_2")
  
  pairs
}



#generating the training and validation pairs
train_pairs <- make_pairs(ids = train_ids, n_pairs = 12000, meta = meta)

valid_pairs <- make_pairs(ids = valid_ids, n_pairs = 3000, meta = meta)


#then creating a function to verify if the pairs were constructed correctly
check_pairs <- function(pairs, meta) {
  
  same_writer <- meta$person_id[pairs[, 1]] == meta$person_id[pairs[, 2]]
  
  same_digit <- meta$digit[pairs[, 1]] == meta$digit[pairs[, 2]]
  
  table(same_writer)
  table(same_digit)
  
  stopifnot(all(same_digit))
  stopifnot(!any(pairs[, 1] == pairs[, 2]))
}

check_pairs(train_pairs, meta)
check_pairs(valid_pairs, meta)


#creating a function to convert the pairs into keras inputs, adapted from the lecture slides
make_batch <- function(pairs, images, meta) {
  
  y <- matrix(as.numeric(meta$person_id[pairs[, 1]] == meta$person_id[pairs[, 2]]), ncol = 1)
  
  list(left = images[pairs[, 1], , , , drop = FALSE],
       right = images[pairs[, 2], , , , drop = FALSE],
       y = y)
}

train <- make_batch(train_pairs, x, meta)
valid <- make_batch(valid_pairs, x, meta)


#checking the training and validation pairs
stopifnot(all(c("left", "right", "y") %in% names(train)), all(c("left", "right", "y") %in% names(valid)))

#number of training and validation pairs
dim(train$left)[1]
dim(valid$left)[1]

#distribution of training labels
table(as.vector(train$y))

#distribution of validation labels
table(as.vector(valid$y))



#Create the configurable 3-layer CNN encoder
#the architecture will stay the same but we can allow the embedding dimension to change

build_siamese_encoder <- function(embedding_units = 32) {
  
  image <- layer_input(shape = c(28, 28, 1))
  
  z <- image |>
    layer_conv_2d(
      filters = 16,
      kernel_size = 3,
      activation = "relu"
    ) |>
    layer_max_pooling_2d(pool_size = 2) |>
    layer_conv_2d(
      filters = 32,
      kernel_size = 3,
      activation = "relu"
    ) |>
    layer_max_pooling_2d(pool_size = 2) |>
    layer_conv_2d(
      filters = 64,
      kernel_size = 3,
      activation = "relu"
    ) |>
    layer_global_average_pooling_2d() |>
    layer_dense(
      units = embedding_units,
      activation = "relu",
      name = "embedding"
    )
  
  keras_model(
    inputs = image,
    outputs = z,
    name = "encoder"
  )
}



#Configurable contrastive loss
#created such that each time we call the function it creates a new Siamese network with freshly initialised weights. 

make_contrastive_loss <- function(margin = 1) {
  
  force(margin)
  
  function(y, d) {
    
    op_mean(
      y * op_square(d) +
        (1 - y) * op_square(
          op_maximum(margin - d, 0)
        )
    )
  }
}


#Configurable Siamese model
build_siamese_model <- function(
    embedding_units = 32,
    margin = 1,
    learning_rate = 0.001) {
  
  encoder <- build_siamese_encoder(
    embedding_units = embedding_units
  )
  
  a <- layer_input(shape = c(28, 28, 1))
  b <- layer_input(shape = c(28, 28, 1))
  
  distance <- list(encoder(a), encoder(b)) |>
    layer_lambda(f = function(z) {
      
      op_sqrt(
        op_sum(
          op_square(z[[1]] - z[[2]]),
          axis = 2,
          keepdims = TRUE
        ) + 1e-7
      )
      
    })
  
  model <- keras_model(
    inputs = list(a, b),
    outputs = distance
  )
  
  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = learning_rate
    ),
    loss = make_contrastive_loss(
      margin = margin
    )
  )
  
  list(
    model = model,
    encoder = encoder
  )
}



#Hyperparameter configurations
tuning_grid <- data.frame(
  model_id = 1:9,
  
  learning_rate = c(
    0.001, 0.0005, 0.002,
    0.001, 0.001, 0.001,
    0.001, 0.0005, 0.0005
  ),
  
  margin = c(
    1, 1, 1,
    0.5, 1.5, 1,
    1, 1.5, 1
  ),
  
  embedding_units = c(
    32, 32, 32,
    32, 32, 16,
    64, 32, 64
  )
)

seeds <- c(2026, 789)

tuning_grid




#Validation ROC AUC
calculate_auc <- function(y, distances) {
  
  y <- as.integer(as.vector(y))
  
  roc_obj <- pROC::roc(
    response = y,
    predictor = -as.numeric(distances),
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  as.numeric(pROC::auc(roc_obj))
}


#Validation balanced accuracy and threshold

find_best_threshold <- function(y, distances) {
  
  y <- as.integer(as.vector(y))
  distances <- as.numeric(distances)
  
  thresholds <- sort(unique(distances))
  
  scores <- sapply(thresholds, function(t) {
    
    predictions <- as.integer(distances <= t)
    
    sensitivity <- mean(predictions[y == 1] == 1)
    specificity <- mean(predictions[y == 0] == 0)
    
    (sensitivity + specificity) / 2
  })
  
  best_index <- which.max(scores)
  
  list(
    threshold = thresholds[best_index],
    balanced_accuracy = scores[best_index]
  )
}







#Training and evaluating

run_tuning_model <- function(model_id, learning_rate, margin, embedding_units, seed) {
  
  #clear previous model state
  keras3::clear_session()
  
  #set seed before creating the model
  keras3::set_random_seed(as.integer(seed))
  
  #build a fresh model
  networks <- build_siamese_model(embedding_units = embedding_units, margin = margin, learning_rate = learning_rate)
  
  model <- networks$model
  
  #train
  history <- model |> fit(x = list(train$left, train$right), y = train$y, validation_data = list(
      list(valid$left, valid$right), valid$y), epochs = 20, batch_size = 128,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 4, min_delta = 0.0005, restore_best_weights = TRUE)), verbose = 0)
  
  #predict the validation distances
  valid_dist <- as.numeric(predict(model, list(valid$left, valid$right), verbose = 0))
  
  #calculate the validation AUC
  valid_auc <- calculate_auc(valid$y, valid_dist)
  
  #calculate the validation threshold and balanced accuracy
  threshold_result <- find_best_threshold(valid$y, valid_dist)
  
  #record the training information
  history_df <- as.data.frame(history)
  
  epochs_run <- max(history_df$epoch)
  
  #return the results as a data frame
  data.frame(
    model_id = model_id,
    learning_rate = learning_rate,
    margin = margin,
    embedding_units = embedding_units,
    seed = seed,
    epochs_run = epochs_run,
    val_auc = valid_auc,
    val_balanced_accuracy =
      threshold_result$balanced_accuracy,
    val_threshold = threshold_result$threshold
  )
}

#Full hyperparameter tuning
dir.create("outputs", showWarnings = FALSE)

tuning_results <- data.frame()

for (i in seq_len(nrow(tuning_grid))) {
  
  config <- tuning_grid[i, ]
  
  for (seed in seeds) {
    
    #to keep track of model and seeds
    cat("\nTraining model", config$model_id, "| seed:", seed, "\n")
    
    result <- run_tuning_model(
      model_id = config$model_id,
      learning_rate = config$learning_rate,
      margin = config$margin,
      embedding_units = config$embedding_units,
      seed = seed
    )
    
    tuning_results <- rbind(tuning_results, result)
    
    write.csv(tuning_results, "outputs/siamese_tuning_results.csv", row.names = FALSE)
    
    result
  }
}

tuning_results


#Summarising tuning results

tuning_summary <- tuning_results |> group_by(model_id, learning_rate, margin, embedding_units) |>
  summarise(mean_auc = mean(val_auc), sd_auc = sd(val_auc), mean_balanced_accuracy = mean(val_balanced_accuracy), mean_epochs = mean(epochs_run), .groups = "drop") |> arrange(desc(mean_auc))

tuning_summary

write.csv(tuning_summary, "outputs/siamese_tuning_summary.csv", row.names = FALSE)

best_config <- tuning_summary[1, ]

best_config



#Hyperparameter sensitivity plots

#Learning rate
p_lr <- tuning_results |> filter(model_id %in% c(1, 2, 3)) |>
  ggplot(aes(x = factor(learning_rate), y = val_auc, group = factor(seed), colour = factor(seed))) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Sensitivity to learning rate",
    x = "Learning rate",
    y = "Validation ROC AUC",
    colour = "Seed"
  ) +
  theme_minimal()

p_lr

#Margin
p_margin <- tuning_results |> filter(model_id %in% c(1, 4, 5)) |>
  ggplot(aes(x = factor(margin), y = val_auc, group = factor(seed), colour = factor(seed))) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Sensitivity to contrastive margin",
    x = "Margin",
    y = "Validation ROC AUC",
    colour = "Seed"
  ) +
  theme_minimal()

p_margin

#Embedding size
p_embedding <- tuning_results |> filter(model_id %in% c(1, 6, 7)) |>
  ggplot(aes(x = factor(embedding_units), y = val_auc, group = factor(seed), colour = factor(seed))) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    title = "Sensitivity to embedding dimension",
    x = "Embedding dimension",
    y = "Validation ROC AUC",
    colour = "Seed"
  ) +
  theme_minimal()

p_embedding


ggsave("outputs/siamese_tuning_learning_rate.png", p_lr, width = 7, height = 4, dpi = 300)

ggsave("outputs/siamese_tuning_margin.png", p_margin, width = 7, height = 4, dpi = 300)

ggsave("outputs/siamese_tuning_embedding.png", p_embedding, width = 7, height = 4, dpi = 300)




#Training the  final selected Siamese model

keras3::clear_session()
keras3::set_random_seed(2026)

final_networks <- build_siamese_model(embedding_units = as.integer(best_config$embedding_units), margin = as.numeric(best_config$margin), learning_rate = as.numeric(best_config$learning_rate))

final_model <- final_networks$model
final_encoder <- final_networks$encoder

final_history <- final_model |> 
  fit(x = list(train$left, train$right), y = train$y, 
      validation_data = list(list(valid$left, valid$right), valid$y), epochs = 20, batch_size = 128, 
      callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 4, min_delta = 0.0005, restore_best_weights = TRUE)), verbose = 1)

plot(final_history)


#Validation performance and threshold

final_valid_dist <- as.numeric(predict(final_model, list(valid$left, valid$right), verbose = 0))

final_valid_auc <- calculate_auc(valid$y, final_valid_dist)

final_threshold_result <- find_best_threshold(valid$y, final_valid_dist)

final_threshold <- final_threshold_result$threshold

final_valid_auc
final_threshold_result$balanced_accuracy
final_threshold


#Held-out test evaluation

test_pairs <- read.csv("data/test_pairs.csv")

test_pairs$row_1 <- match(test_pairs$id_1, meta$image_id)

test_pairs$row_2 <- match(test_pairs$id_2, meta$image_id)

stopifnot(!anyNA(test_pairs$row_1), !anyNA(test_pairs$row_2), 
      all(meta$split_main[test_pairs$row_1] == "test"), all(meta$split_main[test_pairs$row_2] == "test"))

test_batch <- make_batch(as.matrix(test_pairs[, c("row_1", "row_2")]), x, meta)

final_test_dist <- as.numeric(predict(final_model, list(test_batch$left, test_batch$right), verbose = 0))

final_test_y <- as.integer(test_pairs$same_writer)

#Testing ROC AUC
final_test_auc <- calculate_auc(final_test_y, final_test_dist)

#Testing balanced accuracy using validation threshold
final_test_pred <- as.integer(final_test_dist <= final_threshold)

final_sensitivity <- mean(final_test_pred[final_test_y == 1] == 1)

final_specificity <- mean(final_test_pred[final_test_y == 0] == 0)

final_test_bal_acc <- (final_sensitivity + final_specificity) / 2

#APproximate equal error rate
eer_thresholds <- sort(unique(final_test_dist))

error_rates <- sapply(eer_thresholds, function(t) {
  
  pred <- as.integer(final_test_dist <= t)
  
  fpr <- mean(pred[final_test_y == 0] == 1)
  fnr <- mean(pred[final_test_y == 1] == 0)
  
  c(fpr = fpr, fnr = fnr)
})

eer_idx <- which.min(abs(error_rates["fpr", ] - error_rates["fnr", ]))

final_eer <- mean(error_rates[, eer_idx])

#Final result
task5_results <- data.frame(
  Model = paste0("Tuned Siamese (Model ", best_config$model_id, ")"),
  Learning_Rate = as.numeric(best_config$learning_rate),
  Margin = as.numeric(best_config$margin),
  Embedding_Units = as.integer(best_config$embedding_units),
  Validation_AUC = final_valid_auc,
  Test_AUC = final_test_auc,
  Test_Balanced_Accuracy = final_test_bal_acc,
  Approx_EER = final_eer,
  Validation_Threshold = final_threshold
)

task5_results

write.csv(task5_results, "outputs/siamese_task5_final_results.csv", row.names = FALSE)


#Performance by digit matching

test_analysis <- test_pairs |> mutate(distance = final_test_dist, predicted_same = final_test_pred)

digit_results <- test_analysis |> group_by(same_digit) |> group_modify(~ {
    labels <- as.integer(.x$same_writer)
    data.frame(n_pairs = nrow(.x), auc = calculate_auc(labels, .x$distance),
  balanced_accuracy = (mean(.x$predicted_same[labels == 1] == 1) + mean(.x$predicted_same[labels == 0] == 0)) / 2)
  }) |>
  ungroup()

digit_results

write.csv(digit_results, "outputs/siamese_task5_digit_results.csv", row.names = FALSE)


#Export the model outputs

#Test distances
final_test_results <- test_pairs |> select(id_1, id_2, same_writer, same_digit, same_session) |>
  mutate(distance = final_test_dist)

write.csv(final_test_results, "outputs/siamese_task5_test_distances.csv", row.names = FALSE)

#Embeddings
final_embeddings <- predict(final_encoder, x, batch_size = 256, verbose = 0)

final_embeddings_df <- data.frame(image_id = meta$image_id, final_embeddings)

names(final_embeddings_df) <- c("image_id", sprintf("e%02d", 1:32))

write.csv(final_embeddings_df, "outputs/siamese_task5_embeddings.csv", row.names = FALSE)
