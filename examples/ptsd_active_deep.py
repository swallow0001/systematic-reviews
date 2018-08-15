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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from model.textManager import TextManager
from model.embedding import Embedding_Layer
from model.lstm import LSTM_Model

# keras
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical

# libact classes
from libact.base.dataset import Dataset
# from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import InteractiveLabeler, IdealLabeler

# demo utils
from utils import load_ptsd_data
from utils import word2vec_filePath
from labeler import InteractivePaperLabeler


# parse arguments if available
parser = argparse.ArgumentParser(description='Active learning parameters')

# the number of iterations
parser.add_argument(
    "--quota", type=int, default=10, help="The number of queries")

# interactive or not
parser.add_argument('--interactive', dest='interactive', action='store_true',
                    help="Interactive or not?")
parser.add_argument('--no-interactive', dest='interactive',
                    action='store_false')
parser.set_defaults(interactive=False)

# type of model
parser.add_argument('--model', type=str,
                    default='LSTM',
                    help="A deep learning model to use for classification.")


def get_indices_labeled_entries(dataset):
    """Get labeled indices"""

    return [idx for idx, entry in enumerate(dataset.data) if entry[1] is not None]


# def tranform_text_data(texts):
# 
#     # tokenizer
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(texts)
# 
#     sequences = tokenizer.texts_to_sequences(texts)
#     data = pad_sequences(sequences,
#                          maxlen=maxlen,
#                          padding='post', truncating='post')
# 
#     return data, tokenizer.word_index
# 

def make_pool(X, y, prelabeled=np.arange(5)):
    """Function to split dataset into train and test dataset.

    Arguments
    ---------

    prelabeled: list
        List of indices for which the label is already available.

    """

    print(y.shape)
    # a set of labels is already labeled by the oracle
    # y_train_labeled = np.array([None] * len(y))
    y_train_labeled =np.empty((len(y), 2))* np.nan
    y_train_labeled[prelabeled] = y[prelabeled]

    # we are making a pool of the train data
    # the 'prelabeled' labels of the dataset are already labeled.
    return Dataset(X, y_train_labeled), Dataset(X, y)


def main(args):

    acc_pool = []
    maxlen = 100
        
    # get the texts and their corresponding labels
    textManager = TextManager()
    texts, labels = load_ptsd_data()
    data, labels,word_index = textManager.sequence_maker(texts,labels)
    max_num_words = textManager.max_num_words
    max_sequence_length = textManager.max_sequence_length

    embedding = Embedding_Layer(word_index,max_num_words,max_sequence_length)
    embedding.load_word2vec_data(word2vec_filePath)
    embedding_layer = embedding.build_embedding()

        
   
#     from sklearn.feature_extraction.text import CountVectorizer
#     from sklearn.feature_extraction.text import TfidfTransformer
#     from libact.models import SklearnProbaAdapter, SklearnAdapter
# 
#     from sklearn.naive_bayes import MultinomialNB
#     from sklearn.svm import SVC
#     from sklearn.linear_model import LogisticRegression

    # # count words
    # count_vect = CountVectorizer(max_features=5000, stop_words='english')
    # features = count_vect.fit_transform(texts).todense().tolist()

        
    pool, pool_ideal = make_pool(
        data, labels,
        prelabeled=[1, 2, 3, 4, 5, 218, 260, 466, 532, 564]
    )

    # get the model
    if args.model.lower() == 'lstm':
        deep_model = LSTM_Model
        kwargs_model = {
        'backwards':False,
        'dropout':0,
        'optimizer':'rmsprop',
        'max_sequence_length':max_sequence_length,
        'embedding_layer':embedding_layer
        }
    elif args.model.lower() == 'cnn':
        deep_model = LSTM_Model
        kwargs_model = {
        }
    else:
        raise ValueError('Model not found.')
# 
#     # initialize the model through the adapter
#     model = SklearnProbaAdapter(sklearn_model(**kwargs_model))
    model = deep_model(**kwargs_model)

    print(model)
        
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

    qs = UncertaintySampling(
        pool, method='lc', model=model)
 
#     # The passive learning model. The model given in the query strategy is not
#     # the same. Have a look at this one.
#     # model = LogisticRegression()
# 
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Value')

    # Train the model on the train dataset.
    model.train(pool)
# 
    # the accuracy of the entire pool
    acc_pool = np.append(
        acc_pool,
        model._model.score([x[0] for x in pool.get_entries()], labels)
    )
# 
#     # make plot
    query_num = np.arange(0, 1)
    p2, = ax.plot(query_num, acc_pool, 'r', label='Accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
                shadow=True, ncol=5)
    plt.show(block=False)
# 
    # Give each label its name (labels are from 0 to n_classes-1)
    if args.interactive:
        lbr = InteractivePaperLabeler(label_name=["0", "1"])
    else:
        lbr = IdealLabeler(dataset=pool_ideal)
# 
    query_i = 1

    while query_i <= args.quota:

        # make a query from the pool
        print("Asking sample from pool with Uncertainty Sampling")
        ask_id = qs.make_query()
        print("Index {} returned. True label is {}.".format(
            ask_id, pool_ideal.data[ask_id][1]))
# 
#         # get the paper
        data_point = pool.data[ask_id][0]
        lb = lbr.label(data_point)

        # update the label in the train dataset
        pool.update(ask_id, lb)

        # train the model again
        model.train(pool)

#         # append the score to the model
        acc_pool = np.append(
            acc_pool,
            model._model.score([x[0] for x in pool.get_entries()], labels)
         )
# 
        # additional evaluations
        pred = model.predict([x[0] for x in pool.get_entries()])
        print(confusion_matrix(labels, pred))
        print(recall_score(labels, pred))

        if args.interactive:
            # update plot
            ax.set_xlim((0, query_i))
            ax.set_ylim((0, max(acc_pool) + 0.2))
            p2.set_xdata(np.arange(0, query_i + 1))
            p2.set_ydata(acc_pool)
            plt.draw()

        # update the query counter
        query_i += 1

    if not args.interactive:
        # update plot
        ax.set_xlim((0, query_i - 1))
        ax.set_ylim((0, max(acc_pool) + 0.2))
        p2.set_xdata(np.arange(0, query_i))
        p2.set_ydata(acc_pool)
        plt.draw()

    print(acc_pool)

    input("Press any key to continue...")


if __name__ == '__main__':

    # parse all the arguments
    args = parser.parse_args()

    try:
        # start the active learning algorithm
        main(args)
    except KeyboardInterrupt:
        print('Closing down.')
