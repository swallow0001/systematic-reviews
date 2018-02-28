#Model 1. Copy of de Imdb sentiment example: https://keras.rstudio.com/articles/examples/imdb_lstm.html
#
#author: Kees van Eijden
#version/date:  0.3/21-feb-2018

library(stringr)
library(dplyr)
library(keras)
library(caret)

#metric functions precision and recall for training model
#used to be in keras package but have been left out as of keras 2.0

metric_precision <- function(y_true, y_pred) {
    #Only computes a batch-wise average of precision.
    
    #Computes the precision, a metric for multi-label classification of
    #how many selected items are relevant.
    
    true_positives      <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
    predicted_positives <- k_sum(k_round(k_clip(y_pred, 0, 1)))
    precision           <- true_positives / (predicted_positives + k_epsilon())
    return(precision)
}

metric_recall <- function(y_true, y_pred) {
    
    #Only computes a batch-wise average of recall.
    #Computes the recall, a metric for multi-label classification of
    #how many relevant items are selected.
    #
    
    true_positives     <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
    possible_positives <- k_sum(k_round(k_clip(y_true, 0, 1)))
    recall             <- true_positives / (possible_positives + k_epsilon())
    return(recall)
}

#parameter settings
#parameters of the input data 
max_features     <- 20000
quantile_prob    <- 0.75    #used for truncating and padding input sequences with 
                            #quantile_probs*100 % quantile as length 
test_ratio       <- 0.1     #fraction of input data set apart for testing the fitted model 

#model parameters
word_dim         =  128      #first layer maps word tokens to vectors in a 'word_dim' dimensional space
activation       = 'sigmoid'

#parameters for compiling the model
loss             = 'binary_crossentropy'
optimizer        = 'adam'
metrics          = c('recall' = metric_recall, 'precision' = metric_precision)

#parameters for fitting process
batch_size       = 128
epochs           = 100
validation_split = 0.2 #fraction of the train data used for validation after epoch

#reading traindata schoot-lgmm-ptsd
cat('Loading data..schoot-lgmm-ptsd.\n')
train_data      <- read.csv(file = "./data/example_dataset_1/csv/schoot-lgmm-ptsd-traindata.csv",
                            header= TRUE, sep= ",", stringsAsFactors = FALSE)

#the model will train on the title and abstract together as one text string
train_data <- train_data %>%
                mutate(
                  text = str_c( str_replace_na(title, ""),
                                str_replace_na(abstract, ""),
                                sep=" "
                              )
                ) %>% select(text, included_ats, included_final)



##contrary to the idm example, we don't have a vocabulary yet
##we use the keras tokenizer to create a vocabulary from the words in the abstracts and the titles
##every word is assigned a integer token
##the abstracts and titles are are transformed into sequences (vectors of tokens)
tokenizer     <- keras::text_tokenizer(num_words = max_features)
tokenizer    %>% keras::fit_text_tokenizer(train_data$text)




sequences <- keras::texts_to_sequences(tokenizer, train_data$text)

#we need sequences with fixed length. 
#the 75% quantile of lengths of all tokenized text is used. Shorter sequences are padded with 0
#this will favors titles because they are put in front of the abstracts
cat('Pad sequences................\n')
sequence_len  <- quantile(sapply(sequences, length), probs= quantile_prob)

sequences     <- keras::pad_sequences(sequences, maxlen= sequence_len, padding = "post", truncating = "post")



#make train en test sets
cat("Make train en test set .......\n")
set.seed(12345)
no_samples    <- dim(sequences)[1]
test_size     <- as.integer(no_samples * test_ratio)
TESTSET       <- sample(1:no_samples, test_size)

x_train       <- sequences[-TESTSET, ]
y_train_ats   <- as.integer(train_data$included_ats[-TESTSET])
y_train_final <- as.integer(train_data$included_final[-TESTSET])

x_test        <- sequences[TESTSET, ]
y_test_ats    <- as.integer(train_data$included_ats[TESTSET])
y_test_final  <- as.integer(train_data$included_final[TESTSET])

cat("Overview of train en test tensors ......................\n")
cat('x_train tensor: ', length(dim(x_train)), ' shape:', dim(x_train), '\n')
cat('x_test  tensor: ', length(dim(x_test)), ' shape:', dim(x_test), '\n')


cat("Building model ........\n")
model <- keras::keras_model_sequential()
model %>%
    keras::layer_embedding(input_dim = max_features, output_dim = word_dim) %>% 
    keras::layer_lstm(units = word_dim, dropout = 0.2, recurrent_dropout = 0.2) %>% 
    keras::layer_dense(units = 1, activation = activation)

# Try using different optimizers and different optimizer configs
model %>% keras::compile(
    loss      = loss,
    optimizer = optimizer,
    metrics   = metrics
)

cat("Train the model on abstracts as features and included after title screening as labels .......\n")
history <- model %>% keras::fit(
    x_train, y_train_ats,
    batch_size       = batch_size,
    epochs           = epochs,
    validation_split = validation_split 
    #validation_data = list(x_test, y_test)
)

cat("Evaluate model on test set: ........\n")
scores <- model %>% keras::evaluate(
    x_test, y_test_ats,
    batch_size = batch_size
)

cat('Loss on test set:', scores[[1]])
cat('Accuracy on test set', scores[[2]])

y_test_pred <- model %>% keras::predict_classes(x_test)

y_test_pred <- factor(x=y_test_pred, levels= c("1", "0"))
y_test_ats  <- factor(x=y_test_ats, levels= c("1", "0"))
conf_mod    <- caret::confusionMatrix(y_test_pred, y_test_ats, mode= "everything", dnn= c("Prediction", "ATS"))

cat('Confusion matrix based on test set:\n')
print(conf_mod$table)

cat("Evaluation of model performance based on test set\n")
cat("Recall:    ", conf_mod$byClass["Recall"], "\n")
cat("Precision: ", conf_mod$byClass["Precision"], "\n")


cat("End of model 1 .................\n")

