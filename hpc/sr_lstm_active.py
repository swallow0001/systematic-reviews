#!/usr/bin/env python3
"""
This script simulates real world use of active learning algorithms. Which in the
start, there are only a small fraction of samples are labeled. During active
learing process active learning algorithm (QueryStrategy) will choose a sample
from unlabeled samples to ask the oracle to give this sample a label (Labeler).

In this example, ther dataset are from the digits dataset from sklearn. User
would have to label each sample choosed by QueryStrategy by hand. Human would
label each selected sample through InteractiveLabeler. Then we will compare the
performance of using UncertaintySampling and RandomSampling under
LogisticRegression.
"""

import copy
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# project dependencies
sys.path.insert(0, 'python')  # path to the module.

from models.textmanager import TextManager
from models.embedding import Word2VecEmbedding
from models.lstm_libact import LSTM_Libact

from sklearn.model_selection import StratifiedKFold

# libact classes
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import InteractiveLabeler, IdealLabeler
from libact_utils.labeler import InteractivePaperLabeler

# demo utils
from utils import *
from config import *

# parse arguments if available
parser = argparse.ArgumentParser(description='Active learning parameters')
parser.add_argument("-T", default=1, type=int, help='Task number.')
parser.add_argument("-dataset", type=str, default='ptsd', help="The dataset to use for training.")

# the number of iteration
parser.add_argument("-quota", type=int, default=10, help="The number of queries")
parser.add_argument('-interactive', dest='interactive', action='store_true',help="Interactive or not?")
parser.add_argument('-no-interactive', dest='interactive', action='store_false')
parser.set_defaults(interactive=False)

parser.add_argument("-init_included_papers",default=10, type=int, help='Initial number of included papers')

parser.add_argument('-model', type=str, default='LSTM', help="A deep learning model to use for classification.")
parser.add_argument("-query_strategy", type=str, default='lc', help="The query strategy")

def get_indices_labeled_entries(dataset):
    """Get labeled indices"""
    return [idx for idx, entry in enumerate(dataset.data) if entry[1] is not None]
    
def select_prelabeled(labels, init_included, seed):
    included_indexes = np.where(labels[:, 1] == 1)[0]
    excluded_indexes = np.where(labels[:, 1] == 0)[0]
    included_number = min(init_included,len(included_indexes))
    np.random.seed(seed)
    to_add_indx = np.random.choice(included_indexes, included_number, replace=False)
    to_add_indx = np.append(to_add_indx,
              np.random.choice(excluded_indexes, included_number, replace=False))
    return to_add_indx

def make_pool(X, y, prelabeled=np.arange(5)):
    """Function to split dataset into train and test dataset.

    Arguments
    ------

    prelabeled: list
        List of indices for which the label is already available.

    """
    y = y.argmax(axis=1)
    # a set of labels is already labeled by the oracle
    y_train_labeled = np.array([None] * len(y))
    #y_train_labeled =np.empty((len(y), 2))* np.nan
    y_train_labeled[prelabeled] = y[prelabeled]

    # we are making a pool of the train data
    # the 'prelabeled' labels of the dataset are already labeled.
    return Dataset(X, y_train_labeled), Dataset(X, y)

# For cross validation
def cross_validation(model,pool,split_no,seed):

    np.random.seed(seed)
    
    X,Y = pool.format_sklearn()

    kfold = StratifiedKFold(n_splits=split_no, shuffle = True, random_state=seed)
    cvscores =[]
    for train_idx, val_idx in kfold.split(X, Y):
        print('train_idx:',train_idx,' val_idx:',val_idx)
        model.train(pool,
            list(X[idx] for idx in train_idx), 
            list(Y[idx] for idx in train_idx),
            list(X[idx] for idx in val_idx),
            list(Y[idx] for idx in val_idx))
        tn, fp, fn, tp, pred = model.score(
            list(X[idx] for idx in val_idx),
            list(Y[idx] for idx in val_idx),1 )
        to_read_rate = (tp + fp)/(tp + fp + tn + fn)
        print('to_read_rate:',to_read_rate)
#     #     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # add recalls
        cvscores.append(to_read_rate)
#     # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))    
    return (np.mean(cvscores), np.std(cvscores))
def main(args):
    pickle_file_name = args.dataset + '_pickle.pickle'
    pickle_file_path = os.path.join(TEMP_DATA_DIR, pickle_file_name)

    seed = 2018* args.T
    if args.dataset == 'ptsd':
        texts, lbls = load_ptsd_data()
    else: 
        texts, lbls = load_drug_data(args.dataset)

    # get the texts and their corresponding labels
    textManager = TextManager()
    data, labels,word_index = textManager.sequence_maker(texts,lbls)
    max_num_words = textManager.max_num_words
    max_sequence_length = textManager.max_sequence_length

    prelabeled_index = select_prelabeled(labels, args.init_included_papers, seed)
    # [1, 2, 3, 4, 5, 218, 260, 466, 532, 564]
    print('prelabeled_index',prelabeled_index)
    pool, pool_ideal = make_pool(
        data, labels,
        prelabeled=prelabeled_index
    )

    if os.path.isfile(pickle_file_path):
        embedding_layer = load_pickle(pickle_file_path)
    else:
        if not os.path.exists(TEMP_DATA_DIR):
            os.makedirs(TEMP_DATA_DIR)

        embedding = Word2VecEmbedding(word_index, max_num_words, max_sequence_length)
        embedding.load_word2vec_data(GLOVE_PATH)
        embedding_layer = embedding.build_embedding()
        dump_pickle(embedding_layer, pickle_file_path)
    # get the model
    if args.model.lower() == 'lstm':
        deep_model = LSTM_Libact
        kwargs_model = {
        'backwards':True,
        'dropout':0.4,
        'optimizer':'rmsprop',
        'max_sequence_length':max_sequence_length,
        'embedding_layer':embedding_layer
        }
    else:
        raise ValueError('Model not found.')

    model = deep_model(**kwargs_model)

#     # query strategy
#     # https://libact.readthedocs.io/en/latest/libact.query_strategies.html
#     # #libact-query-strategies-uncertainty-sampling-module
#     #
#     # least confidence (lc), it queries the instance whose posterior
#     # probability of being positive is nearest 0.5 (for binary
#     # classification); smallest margin (sm), it queries the instance whose
#     # posterior probability gap between the most and the second probable
#     # labels is minimal
#     qs = UncertaintySampling(
#         pool, method='lc', model=SklearnProbaAdapter(sklearn_model(**kwargs_model)))

#Todo: check if 'lc' works correctly/ add random as well
    qs = UncertaintySampling(
        pool, method='lc', model=deep_model(**kwargs_model))

    # Give each label its name (labels are from 0 to n_classes-1)
    if args.interactive:
        lbr = InteractivePaperLabeler(label_name=["0", "1"])
    else:
        lbr = IdealLabeler(dataset=pool_ideal)

    result_df = pd.DataFrame({'label': [x[1] for x in pool_ideal.data]})
    query_i = 1
##Todo: add multiple papers to labeled dataset with size of batch_size
    while query_i <= args.quota:

        # make a query from the pool
        print("Asking sample from pool with Uncertainty Sampling")
        # unlabeled_entry = pool.get_unlabeled_entries()

        ask_id = qs.make_query()
        print("Index {} returned. True label is {}.".format(
            ask_id, pool_ideal.data[ask_id][1]))

        # get the paper
        data_point = pool.data[ask_id][0]
        lb = lbr.label(data_point)

        # update the label in the train dataset
        pool.update(ask_id, lb)
        # train the model again
        # to_read_mean, to_read_std = cross_validation(model,pool,split_no=3,seed =query_i)
        model.train(pool)

        idx_features = pool.get_unlabeled_entries()
        idx= [x[0] for x in idx_features]
        features = [x[1] for x in idx_features]
        pred = model.predict(features)

        c_name = str(query_i)
        result_df[c_name]=-1
        result_df.loc[idx,c_name]= pred[:,1]

        # update the query counter
        query_i += 1


    # save the result to a file
    output_dir = os.path.join(ACTIVE_DIR,args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    export_path = os.path.join(output_dir,'sr_lstm_active{}.csv'.format(args.T))

    result_df.to_csv(export_path)
    input("Press any key to continue...")


if __name__ == '__main__':

    # parse all the arguments
    args = parser.parse_args()

    try:
        # start the active learning algorithm
        main(args)
    except KeyboardInterrupt:
        print('Closing down.')