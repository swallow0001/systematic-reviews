# Adapted from IMDB LSTM example:
# https://keras.rstudio.com/articles/examples/imdb_lstm.html
#
# Author: Kees van Eijden
rm(list=ls())
set.seed(20180301)

library(keras)
library(tidyverse)
library(caret)
library(PRROC)

# Parameters for transforming raw text into tokenized vectors
max_features     <- 8000  # First max_features words (ordered by frequency) are used
quantile_prob    <- 0.7   #  Used for truncating and padding input sequences with
                          # quantile_probs*100 % quantile as length
validation_split = 0.2 # Fraction of the train data used for validation

# The target variable containing the classes to be predicted in full_data.
#    (either included_final (second stage) or included_ats (first stage))
target_variable <- "included_final"


##################################################################
# Reading data and creating train and test sets                  #
##################################################################

# Read input data generated from read_and_clean_schoot_lgmm_ptsd.R
#     (if this data is unavailable run read_and_clean_schoot_lgmm_ptsd.R first)
cat('Loading data..schoot-lgmm-ptsd.\n')
full_data <- read.csv(file = "./data/example_dataset_1/csv/schoot-lgmm-ptsd-traindata.csv",
                            header= TRUE, sep= ",", stringsAsFactors = FALSE)

# Create test and train sets (seed set above)
# Due to extreme sparseness, test and train are stratified with pps

# Create test set
test_set_size <- floor(validation_split * nrow(full_data)) # Number of test cases
positive_proportion <- mean(full_data[, target_variable]) # Proportion of included cases
all_idx <- 1:nrow(full_data)
positive_test_size <- floor(test_set_size * positive_proportion)

# Sample at random within class strata
test_set_idx <-
  sample(all_idx[full_data[, target_variable] == 1], size = positive_test_size) %>%
  c(sample(all_idx[full_data[, target_variable] == 0],
           size = test_set_size - positive_test_size))

test_set_idx <- test_set_idx %>% sample # Randomize resulting test set

# Sanity checks on test set generation
stopifnot(length(test_set_idx) == test_set_size)
# Test data should have abou the same proportion of included papers as full data:
stopifnot(abs(mean(full_data[, target_variable][test_set_idx]) - positive_proportion) < 1e-3)

# Create training set
train_set_idx <- all_idx[!all_idx %in% test_set_idx]
train_set_idx <- train_set_idx %>% sample

# Proportion of included papers (used below)
train_set_positive_proportion <- mean(full_data[, target_variable][train_set_idx])

# Training will be performed with class weights inversely proportional to
#   class sizes.
class_weights <- c("0" = 1/(1 - train_set_positive_proportion),
                   "1" = 1/train_set_positive_proportion)

# Sanity checks on training data and class weights
# Weighted class size should be balanced:
stopifnot(abs(weighted.mean(full_data[train_set_idx, 'included_final'], class_weights[as.character(full_data[train_set_idx, 'included_final'])]) - 0.5) < 1e-2)

# All cases should be included in either train or test set
stopifnot(length(intersect(train_set_idx, test_set_idx)) == 0)


#####################################
# Turn raw text into word sequences #
#####################################

## Contrary to the IMDB example, we don't have a vocabulary yet
##   we use the keras tokenizer to create a vocabulary from the words in the abstracts and the titles
##   every word is assigned an integer token
##   the abstracts and titles are transformed into sequences (vectors of tokens)
tokenizer     <- keras::text_tokenizer(num_words = max_features)
tokenizer %>% keras::fit_text_tokenizer(full_data$text)

# Following outputs different object (proto class or array) depending on Keras v
sequences <- keras::texts_to_sequences(tokenizer, full_data$text)

# We need sequences with fixed length.
#    the quantile_prob quantile of lengths of all tokenized text is used.
#    Shorter sequences are padded with 0 this will favors titles because
#    they are put in front of the abstracts.

sequence_len  <- quantile(sapply(sequences, length), probs = quantile_prob)
sequences     <- keras::pad_sequences(sequences,
                       maxlen = sequence_len,
                       padding = "post",
                       truncating = "post")

# Final train and test sets are obtained from generated word sequences and
#    previously created training and test indices (_idx variables)
x_train       <- sequences[train_set_idx, ]
y_train       <- as.integer(full_data[, target_variable][train_set_idx])

x_test        <- sequences[test_set_idx, ]
y_test        <- as.integer(full_data[, target_variable][test_set_idx])

cat("Overview of train and test tensors ......................\n")
cat('x_train tensor: ', length(dim(x_train)), ' shape:', dim(x_train), '\n')
cat('x_test  tensor: ', length(dim(x_test)), ' shape:', dim(x_test), '\n')


#####################
# Build Keras model #
#####################

cat("Building model...\n")

# Paramter settings

# Embedding
max_features = max_features # Dictionary size
maxlen = sequence_len # Maximum sequence length
embedding_size = 128

# Convolution
kernel_size <- 20
filters <- 128
pool_size <- 8

# LSTM
lstm_output_size <- 128

# Training
loss             <-  'binary_crossentropy'
optimizer        <-  'adam'
batch_size       <-  30
epochs           <-  1

# Custom metrics are loaded from separate file:
source("r/custom_metrics.R")
metrics          <-  c('recall' = metric_recall, 'precision' = metric_precision)

# Model generation
model <- keras::keras_model_sequential()

# model %>%
#     keras::layer_embedding(input_dim = max_features, output_dim = 64) %>%
#     keras::layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2, activation = "relu") %>%
#     keras::layer_dense(units = 1, activation = "sigmoid")

model %>%
  layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
  layer_dropout(0.25) %>%
  layer_conv_1d(
    filters,
    kernel_size,
    padding = "valid",
    activation = "relu",
    strides = 1
  ) %>%
  layer_max_pooling_1d(pool_size) %>%
  layer_lstm(lstm_output_size) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model %>% keras::compile(loss      = loss,
                         optimizer = optimizer,
                         metrics   = metrics)

cat("Training the model...\n")
history <- model %>% keras::fit(
  x_train,
  y_train,
  batch_size = batch_size,
  shuffle = TRUE,
  view_metrics = FALSE,
  class_weight = as.list(class_weights),
  epochs = epochs,
  # validation_split = validation_split
  validation_data = list(x_test, y_test)
)

####################
# Model evaluation #
####################

cat("Evaluate model on test set...\n")

# Use Keras vanilla test set evaluation
scores <- model %>% keras::evaluate(x_test, y_test, batch_size = batch_size)
print(scores)

# Predicted classes (default threshold):
y_test_pred <- model %>% keras::predict_classes(x_test)
# Raw network output (predicted probability for sigmoid)
y_test_pred_prob <- model %>% keras::predict_proba(x_test)

# Turn prediction and y_test into factors for easier interpretation
y_test_pred <- factor(y_test_pred, levels = c("1", "0"))
y_test  <- factor(y_test, levels = c("1", "0"))

# Calculate confusion matrix and bunch of other metrics:
conf_mod    <-
  caret::confusionMatrix(y_test_pred,
                         y_test,
                         mode = "everything",
                         dnn = c("Prediction", "Included"))
print(conf_mod)

# Calibration plot:
caret::calibration(y_test ~ y_test_pred_prob[,1]) %>% plot

# Precision-recall curve

# use PRROC package to calculate P-R curve and AUC:
pr_score <- PRROC::pr.curve(scores.class0 = y_test_pred_prob[,1],
                            weights.class0 = ifelse(y_test=='0', 0, 1),
                            curve = TRUE)
pr_curve <- as.data.frame(pr_score$curve)
names(pr_curve) <- c("Recall", "Precision", "Cutpoint")

# We will want to include all included papers, so we need to look at
#    precision and cutpoint for some acceptable level of recall, e.g. 0.99
acceptable_recall <- 0.99

# The precision and cutpoint where recall is acceptable
results_acceptable_recall <- pr_curve %>%
  filter(Recall >= 1) %>%
  summarize(max(Precision), Cutpoint[which.max(Precision)])

# Also get ROC curve AUC just for reference
roc_curve <- PRROC::roc.curve(scores.class0 = y_test_pred_prob[,1],
                 weights.class0 = ifelse(y_test=='0', 0, 1),
                 curve = TRUE)

# Plot P-R curve with ggplot. Lines showing the chosen "acceptable" recall
#   level and the corresponding precision are also shown.
pr_curve %>%
  ggplot(aes(Recall, Precision, colour = Cutpoint)) +
  ggtitle(sprintf("Area under P-R curve: %1.3f (area under ROC: %1.3f)", pr_score$auc.integral, roc_curve$auc)) +
  geom_point() + geom_smooth(se = FALSE) +
  ylim(positive_proportion, 1) + xlim(0,1)+
  scale_y_continuous(name = "Precision", breaks = c(results_acceptable_recall$`max(Precision)`, positive_proportion, 0, 0.3, 0.6, 0.9, 1) %>% sort %>% round(2)) +
  scale_x_continuous(name = "Recall", breaks = seq(0,1,by = 0.1)) +
    geom_segment(x = 0, y = results_acceptable_recall$`max(Precision)`, xend = acceptable_recall, yend = results_acceptable_recall$`max(Precision)`, lty = 2) +
    geom_segment(x = acceptable_recall, xend = acceptable_recall, y = 0, yend = results_acceptable_recall$`max(Precision)`, lty = 2)

# The most important metric is the number of non-relevant papers we can filter
#   out in the test data by using the cutpoint corresponding to acceptable_recall

# Show test class distribution before filtering on predicted probability
cat("Test set included and excluded papers:")
print(y_test %>% table)
# Show test class distribution after filtering on predicted probability
cat(sprintf("Test set included and excluded papers, after filtering rule applied (recall>=%1.3f):", acceptable_recall))
print(y_test[y_test_pred_prob >= results_acceptable_recall$`Cutpoint[which.max(Precision)]`] %>% table)

cat("End of model 1.\n")

# export_savedmodel(model, "./saved_models/lstm_cnn")
