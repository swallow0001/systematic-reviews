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

import os
import argparse
import numpy as np

from utils import *
from model.textManager import TextManager
from model.embedding import Embedding_Layer
from model.lstm import LSTM_Model
import pickle
import json
from datetime import datetime



# parse the arguments
parser = argparse.ArgumentParser(description='Systematic Review options')
parser.add_argument("-T", default=1, type=int, help='Task number.')
parser.add_argument("-training_size", default=50, type=int, help='Size of training dataset')
parser.add_argument("-allowed_FN", default=1, type=int, help='Number of allowed False Negative cases')
parser.add_argument("-init_included_papers", default=10, type=int, help='Initial number of included papers')
parser.add_argument("-dataset", default='ptsd', type=str, help='Name of dataset')
sr_args = parser.parse_args()
print(sr_args)


pickle_file_name = sr_args.dataset+'_pickle.pickle'
PICKLE_File = path.join(PICKLE_PATH,pickle_file_name)


start = datetime.now()
# Load dataset
# get the texts and their corresponding labels
if sr_args.dataset == 'ptsd':
    texts, labels = load_ptsd_data()
else: 
    texts, labels = load_drug_data(sr_args.dataset)

textManager = TextManager()
data, labels,word_index = textManager.sequence_maker(texts,labels)
max_num_words = textManager.max_num_words
max_sequence_length = textManager.max_sequence_length

#Split dataset to train and test
x_train, x_val, y_train, y_val = split_data(data, labels,sr_args.training_size, sr_args.init_included_papers, 2017+sr_args.T)
print("x_train shape:", x_train.shape, ", x_val shape:", x_val.shape)
print("y_train shape:", y_train.shape, ", y_val shape:", y_val.shape)
print("included in train",(y_train[:,1]==1).sum())
print("included in test",(y_val[:,1]==1).sum())


runtime = datetime.now()-start
print('loading dataset takes', runtime.total_seconds())


if os.path.isfile(PICKLE_File):
   start = datetime.now() 
   embedding_layer = load_pickle(PICKLE_File)
   runtime = datetime.now()-start
   print('loading pickle takes', runtime.total_seconds())

   
else:     
    start = datetime.now() 

    # make an embedding layer
    embedding = Embedding_Layer(word_index,max_num_words,max_sequence_length)
    embedding.load_word2vec_data(WORD2VEC_PATH)
    embedding_layer = embedding.build_embedding()
    dump_pickle(embedding_layer,PICKLE_File)
    runtime = datetime.now()-start
    print('loading w2vec takes', runtime.total_seconds())


# make a lstm model
deep_model = LSTM_Model
args_model = {
    'backwards':False,
    'dropout':0.3,
    'optimizer':'rmsprop',
    'max_sequence_length':max_sequence_length,
    'embedding_layer':embedding_layer
}
start = datetime.now()
model = deep_model(**args_model)
model.train(x_train, y_train, x_val, y_val)
tn, fp, fn, tp, pred = model.score(x_val, y_val,sr_args.allowed_FN)
runtime = datetime.now()-start
print('training model takes', runtime.total_seconds())

print( 'best_tn' ,tn,
        'best_fp' ,fp,
        'best_fn' ,fn, 
        'best_tp' ,tp)
		
seed_val = 2017+int(sr_args.T)
lstm_scores = {  'T': sr_args.T,
				 'tn': tn,
				 'fp': fp,
				 'fn': fn,
				 'tp': tp,
                 'allowed_FN': sr_args.allowed_FN,
                 'init_included_papers': sr_args.init_included_papers,
				 'dataset': sr_args.dataset,
				 'seed': seed_val,
				 'training_size':sr_args.training_size,
                 'pred': [pred]}

# save the result to a file
if not os.path.exists('output'): os.makedirs('output')
export_path = os.path.join('output', 'sr_lstm{}.json'.format(sr_args.T))
with open(export_path,'w') as outfile:
    json.dump(lstm_scores,outfile)