#Initial set up is the same as in main Siamese.R

library(reticulate)
use_virtualenv("siamese-env", required = TRUE) #I created a new environment

library(keras3)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(pROC)

use_backend("tensorflow")

set.seed(2026)
tf$random$set_seed(2026L)

source("R/00_data.R") #loading the data Jesse gave
data <- load_split_data()

#just inspecting the data
names(data)
str(data)
dim(data$images)
head(data$metadata)


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
  
  print(table(same_writer))
  print(table(same_digit))
  
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



#Writer-held-out pair generation

set.seed(2026)

writer_train_ids <- which(meta$split_writer == "train")
writer_valid_ids <- which(meta$split_writer == "val")
writer_test_ids  <- which(meta$split_writer == "test")

# Confirm no writer overlap
writer_train_people <- unique(meta$person_id[writer_train_ids])
writer_valid_people <- unique(meta$person_id[writer_valid_ids])
writer_test_people  <- unique(meta$person_id[writer_test_ids])

stopifnot(
  length(intersect(writer_train_people, writer_valid_people)) == 0,
  length(intersect(writer_train_people, writer_test_people)) == 0,
  length(intersect(writer_valid_people, writer_test_people)) == 0
)

# Generate same-digit pairs
writer_train_pairs <- make_pairs(
  ids = writer_train_ids,
  n_pairs = 12000,
  meta = meta
)

writer_valid_pairs <- make_pairs(
  ids = writer_valid_ids,
  n_pairs = 3000,
  meta = meta
)

writer_test_pairs <- make_pairs(
  ids = writer_test_ids,
  n_pairs = 3000,
  meta = meta
)

# Verify pair construction
check_pairs(writer_train_pairs, meta)
check_pairs(writer_valid_pairs, meta)
check_pairs(writer_test_pairs, meta)

# Convert to neural-network inputs
writer_train <- make_batch(writer_train_pairs, x, meta)
writer_valid <- make_batch(writer_valid_pairs, x, meta)
writer_test  <- make_batch(writer_test_pairs, x, meta)

# Check class balance
table(as.vector(writer_train$y))
table(as.vector(writer_valid$y))
table(as.vector(writer_test$y))




#Building the unseen-writer Siamese network

keras3::clear_session()
keras3::set_random_seed(2026)

image <- layer_input(shape = c(28, 28, 1))

z <- image |>
  layer_conv_2d(filters = 16, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(units = 32, activation = "relu")

writer_encoder <- keras_model(inputs = image, outputs = z)

a <- layer_input(shape = c(28, 28, 1))
b <- layer_input(shape = c(28, 28, 1))

distance <- list(writer_encoder(a), writer_encoder(b)) |>
  layer_lambda(f = function(z) {
    op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2, keepdims = TRUE) + 1e-7)
  })

writer_model <- keras_model(inputs = list(a, b), outputs = distance)

writer_contrastive_loss <- function(y, d) {
  op_mean(y * op_square(d) + (1 - y) * op_square(op_maximum(1 - d, 0)))
}

writer_model |> compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = writer_contrastive_loss)

summary(writer_model)


#Training the unseen-writer model

writer_history <- writer_model |> fit(x = list(writer_train$left, writer_train$right), y = writer_train$y,
  validation_data = list(list(writer_valid$left, writer_valid$right), writer_valid$y),
  epochs = 20, batch_size = 128,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 4, min_delta = 0.0005, restore_best_weights = TRUE)), verbose = 1)

plot(writer_history)


#Validation-selected threshold
writer_valid_dist <- as.numeric(predict(writer_model, list(writer_valid$left, writer_valid$right), verbose = 0))

writer_balanced_accuracy <- function(y, distance, threshold) {
  
  y <- as.integer(as.vector(y))
  
  predicted <- as.integer(distance <= threshold)
  
  sensitivity <- mean(predicted[y == 1] == 1)
  specificity <- mean(predicted[y == 0] == 0)
  
  (sensitivity + specificity) / 2
}

writer_thresholds <- sort(unique(writer_valid_dist))

writer_scores <- sapply(writer_thresholds, function(t) {
  writer_balanced_accuracy(writer_valid$y, writer_valid_dist, t)
})

writer_best_threshold <- writer_thresholds[which.max(writer_scores)]

writer_valid_auc <- as.numeric(pROC::auc(pROC::roc(
      response = as.integer(as.vector(writer_valid$y)),
      predictor = -writer_valid_dist,
      levels = c(0, 1),
      direction = "<",
      quiet = TRUE
    )
  )
)

writer_valid_auc
max(writer_scores)
writer_best_threshold



#Unseen-writer test evaluation

writer_test_dist <- as.numeric(predict(writer_model, list(writer_test$left, writer_test$right), verbose = 0))

writer_test_y <- as.integer(as.vector(writer_test$y))

writer_roc <- pROC::roc(
  response = writer_test_y,
  predictor = -writer_test_dist,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

writer_test_auc <- as.numeric(pROC::auc(writer_roc))

writer_test_bal_acc <- writer_balanced_accuracy(writer_test_y, writer_test_dist, writer_best_threshold)

# Approximate EER
writer_eer_thresholds <- sort(unique(writer_test_dist))

writer_error_rates <- sapply(writer_eer_thresholds, function(t) {
  
  predicted <- as.integer(writer_test_dist <= t)
  
  fpr <- mean(predicted[writer_test_y == 0] == 1)
  fnr <- mean(predicted[writer_test_y == 1] == 0)
  
  c(fpr = fpr, fnr = fnr)
})

writer_eer_idx <- which.min(abs(writer_error_rates["fpr", ] - writer_error_rates["fnr", ]))

writer_eer <- mean(writer_error_rates[, writer_eer_idx])

#Results
unseen_writer_results <- data.frame(
  Experiment = "Unseen writers",
  Train_Writers = 14,
  Validation_Writers = 3,
  Test_Writers = 3,
  Validation_AUC = writer_valid_auc,
  Test_AUC = writer_test_auc,
  Test_Balanced_Accuracy = writer_test_bal_acc,
  Approx_EER = writer_eer,
  Validation_Threshold = writer_best_threshold
)

unseen_writer_results

dir.create("outputs", showWarnings = FALSE)

write.csv(unseen_writer_results,"outputs/siamese_unseen_writer_results.csv", row.names = FALSE)


#Unseen-writer distance distribution

writer_plot_data <- data.frame(distance = writer_test_dist, pair_type = factor(writer_test_y, levels = c(0, 1), labels = c("Different writer", "Same writer")))

p_unseen <- ggplot(writer_plot_data, aes(x = distance, fill = pair_type)) + 
  geom_histogram(bins = 40, alpha = 0.5, position = "identity") +
  geom_vline(xintercept = writer_best_threshold, linetype = "dashed") +
  labs(
    title = "Siamese network unseen-writer verification",
    x = "Euclidean distance",
    y = "Number of pairs",
    fill = "Pair type"
  ) +
  theme_minimal()

p_unseen

ggsave("outputs/siamese_unseen_writer_distances.png", p_unseen, width = 8, height = 5, dpi = 300)


unseen_test_export <- data.frame(
  id_1 = meta$image_id[writer_test_pairs[, 1]],
  id_2 = meta$image_id[writer_test_pairs[, 2]],
  same_writer = writer_test_y,
  same_digit = (meta$digit[writer_test_pairs[, 1]] == meta$digit[writer_test_pairs[, 2]]),
  same_session = (meta$session[writer_test_pairs[, 1]] == meta$session[writer_test_pairs[, 2]]),
  distance = writer_test_dist
)

write.csv(unseen_test_export, "outputs/siamese_unseen_writer_test_distances.csv", row.names = FALSE)
