### Siamese Network

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

train <- make_batch(train_pairs, x, meta)
valid <- make_batch(valid_pairs, x, meta)

dim(train$left)
dim(train$right)
dim(train$y)



#now building the actual Siamese network

#starting with creating the shared CNN encoder, which is a 3-layer architecture
image <- layer_input(shape = c(28, 28, 1))

z <- image |>
  layer_conv_2d(filters = 16, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(units = 32, activation = "relu")

encoder <- keras_model(inputs = image, outputs = z)

summary(encoder)

#building the siamese model
a <- layer_input(shape = c(28, 28, 1))
b <- layer_input(shape = c(28, 28, 1))

#encoding both images using the same CNN
distance <- list(encoder(a), encoder(b)) |>
  layer_lambda(f = function(z) {
    op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2, keepdims = TRUE) + 1e-7)
  })

model <- keras_model(inputs = list(a, b), outputs = distance)

summary(model)


#defining the contrastive loss
contrastive_loss <- function(y, d) {
  op_mean(y * op_square(d) + (1 - y) * op_square(op_maximum(1 - d, 0)))
}


#compiling and training the model
model |> compile(optimizer = optimizer_adam(learning_rate = 1e-3), loss = contrastive_loss)

history <- model |> fit(
  x = list(train$left, train$right),
  y = train$y,
  validation_data = list(
    list(valid$left, valid$right),
    valid$y),
    epochs = 20,
    batch_size = 128,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3, min_delta = 0.0005, restore_best_weights = TRUE)),
    verbose = 1
  )

plot(history)



#choosing the threshold using validation pairs
valid_dist <- as.numeric(predict(model, list(valid$left, valid$right), verbose = 0))

balanced_accuracy <- function(y, distance, threshold) {
  
  predicted_same <- as.integer(distance <= threshold)
  
  sensitivity <- mean(predicted_same[y == 1] == 1)
  specificity <- mean(predicted_same[y == 0] == 0)
  
  (sensitivity + specificity) / 2
}

thresholds <- sort(unique(valid_dist))

scores <- sapply(thresholds, function(t) {
  balanced_accuracy(valid$y[, 1], valid_dist, t)
})

best_threshold <- thresholds[which.max(scores)]

best_threshold
max(scores)


#loading the test pairs to evaluate on
test_pairs <- read.csv("data/test_pairs.csv")

head(test_pairs)

#matching image IDs to row positions
test_pairs$row_1 <- match(test_pairs$id_1, meta$image_id)

test_pairs$row_2 <- match(test_pairs$id_2, meta$image_id)

#preparing test inputs
test_batch <- make_batch(as.matrix(test_pairs[, c("row_1", "row_2")]), x, meta)

#predicting test distances
test_dist <- as.numeric(predict(model, list(test_batch$left, test_batch$right), verbose = 0))



#evaluating test performance
test_y <- as.integer(test_pairs$same_writer)

roc_obj <- pROC::roc(response = test_y, predictor = -test_dist, levels = c(0, 1), direction = "<", quiet = TRUE)

test_auc <- as.numeric(pROC::auc(roc_obj))

test_bal_acc <- balanced_accuracy(test_y, test_dist, best_threshold)

test_auc
test_bal_acc



#calculating the equal error rate (EER)

eer_thresholds <- sort(unique(test_dist))

error_rates <- sapply(eer_thresholds, function(t) {
  
  pred <- as.integer(test_dist <= t)
  
  fpr <- mean(pred[test_y == 0] == 1)
  fnr <- mean(pred[test_y == 1] == 0)
  
  c(fpr = fpr, fnr = fnr)
})

idx <- which.min(abs(error_rates["fpr", ] - error_rates["fnr", ]))

eer <- mean(error_rates[, idx])

eer


#sumarising the results
task3_results <- data.frame(
  Model = "3-layer Siamese",
  ROC_AUC = test_auc,
  Balanced_Accuracy = test_bal_acc,
  Approx_EER = eer,
  Threshold = best_threshold
)

task3_results



##visualising the matching performance

results <- data.frame(
  distance = test_dist,
  same_writer = factor(
    test_y,
    levels = c(0, 1),
    labels = c("Different writer", "Same writer")
  )
)

ggplot(results, aes(x = distance, fill = same_writer)) +
  geom_histogram(bins = 40, alpha = 0.5, position = "identity") +
  geom_vline(xintercept = best_threshold, linetype = "dashed") +
  labs(
    title = "Siamese network: test pair distances",
    subtitle = "Dashed line indicates validation-selected threshold",
    x = "Euclidean distance",
    y = "Number of pairs",
    fill = "Pair type"
  ) +
  theme_minimal()


#predictions using validation-selected threshold
test_pred <- as.integer(test_dist <= best_threshold)

#false acceptance rate
FAR <- mean(test_pred[test_y == 0] == 1)

#false rejection rate
FRR <- mean(test_pred[test_y == 1] == 0)

#true acceptance and rejection rates
TAR <- 1 - FRR
TRR <- 1 - FAR

data.frame(FAR = FAR, FRR = FRR, TAR = TAR, TRR = TRR)




## exporting the results for Jess

#embeddings
dir.create("outputs", showWarnings = FALSE)

embeddings <- predict(encoder, x, batch_size = 256, verbose = 0)

embeddings_df <- data.frame(image_id = meta$image_id, embeddings)

names(embeddings_df) <- c("image_id", sprintf("e%02d", 1:32))

write.csv(embeddings_df, "outputs/siamese_embeddings.csv", row.names = FALSE)


#test distances
test_results <- test_pairs |> select(id_1, id_2, same_writer, same_digit, same_session) |>
  mutate(distance = test_dist)

write.csv(test_results,"outputs/siamese_test_distances.csv", row.names = FALSE)
