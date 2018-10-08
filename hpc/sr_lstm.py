# Systematic Review with LSTM
#
# Arguments:
#     -t: Task number
#     -training_size: Size of training dataset
#     -allowed_FN : Number of allowed False Negative cases
#     -init_included_papers: Initial number of included papers
#     -dataset: Name of dataset
#
# Authors: Parisa Zahedi
#
# License: BSD-3-Clause
#
# pylint: disable=C0321

# CPython dependencies
import os
import sys
import argparse
import json
from datetime import datetime

# project dependencies
sys.path.insert(0, 'python')  # path to the module.

from models.lstm import LSTM_Model
from utils import split_data
from config import *

# parse the arguments
parser = argparse.ArgumentParser(description='Systematic Review options')
parser.add_argument("-T", default=1, type=int, help='Task number.')
parser.add_argument(
    "-training_size", default=50, type=int, help='Size of training dataset')
parser.add_argument(
    "-allowed_FN",
    default=1,
    type=int,
    help='Number of allowed False Negative cases')
parser.add_argument(
    "-init_included_papers",
    default=10,
    type=int,
    help='Initial number of included papers')
parser.add_argument(
    "-dataset", default='ptsd', type=str, help='Name of dataset')
parser.add_argument(
    "-dropout", default=0, type=float, help='dropout')
sr_args = parser.parse_args()
print(sr_args)

# Read dataset, labels and embedding layer from pickle file.
pickle_fp = os.path.join(TEMP_DATA_DIR, args.dataset + '_pickle.pickle')
with open(pickle_fp, 'rb') as f:
    data, labels, embedding_layer = pickle.load(f)

# Split dataset to train and test
x_train, x_val, y_train, y_val = split_data(
    data, labels, sr_args.training_size, sr_args.init_included_papers,
    2017 + sr_args.T)
print("x_train shape:", x_train.shape, ", x_val shape:", x_val.shape)
print("y_train shape:", y_train.shape, ", y_val shape:", y_val.shape)
print("included in train", (y_train[:, 1] == 1).sum())
print("included in test", (y_val[:, 1] == 1).sum())

# make a lstm model
deep_model = LSTM_Model
args_model = {
    'backwards': True,
    'dropout': sr_args.dropout,
    'optimizer': 'rmsprop',
    'max_sequence_length': 1000,
    'embedding_layer': embedding_layer
}
start = datetime.now()
model = deep_model(**args_model)
model.train(x_train, y_train, x_val, y_val)
tn, fp, fn, tp, pred = model.score(x_val, y_val, sr_args.allowed_FN)
runtime = datetime.now() - start
print('training model takes', runtime.total_seconds())

print('best_tn', tn, 'best_fp', fp, 'best_fn', fn, 'best_tp', tp)

seed_val = 2017 + int(sr_args.T)
lstm_scores = {
    'T': sr_args.T,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'tp': tp,
    'allowed_FN': sr_args.allowed_FN,
    'init_included_papers': sr_args.init_included_papers,
    'dataset': sr_args.dataset,
    'seed': seed_val,
    'training_size': sr_args.training_size,
    'dropout': sr_args.dropout,
    'pred': [pred]
}

# save the result to a file
output_dir = os.path.join(PASSIVE_OUTPUT_DIR, args.dataset)
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
export_path = os.path.join(output_dir, 'dataset_{}_sr_lstm{}.json'.format(sr_args.dataset, sr_args.T))
with open(export_path, 'w') as outfile:
    json.dump(lstm_scores, outfile)
