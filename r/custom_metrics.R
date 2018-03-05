# Metric functions precision and recall for training model
# used to be in keras package but have been left out as of keras 2.0

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
