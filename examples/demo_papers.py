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
from sklearn.metrics import accuracy_score

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from labeler import InteractivePaperLabeler
from libact.labelers import InteractiveLabeler


# parse arguments if available
parser = argparse.ArgumentParser(description='Active learning parameters')

parser.add_argument(
    "--quota", type=int, default=10, help="The number of queries")


def get_indices_labeled_entries(dataset):
    """Get labeled indices"""

    return [idx for idx, entry in enumerate(dataset.data) if entry[1] is not None]


def load_ptsd_data(fp="../data/example_dataset_1/csv/schoot-lgmm-ptsd-traindata.csv"):
    """Load ptsd papers and their labels."""

    # read the data of the file location given as argument to this function
    df = pd.read_csv(fp)

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstract'].fillna(''))
    labels = df["included_final"]

    return texts, labels


def tranform_text_data(texts):

    # tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences,
                         maxlen=maxlen,
                         padding='post', truncating='post')

    return data, tokenizer.word_index



def split_train_test(prelabeled=np.arange(5)):
    """Function to split dataset into train and test dataset.

    Arguments
    ---------

    prelabeled: list
        List of indices for which the label is already available.

    """

    from sklearn.datasets import load_digits

    digits = load_digits(n_class=2)  # consider binary case
    X = digits.data
    y = digits.target
    print(np.shape(X))

    # make a split with the built-in function of sklearn. The Google approach
    # is similar.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("Number of samples in train pool", X_train.shape[0])
    print("Number of samples in test set", X_test.shape[0])

    # All unique labels need to be in the training dataset. If this is not the
    # case, we are splitting the dataset again.
    while len(np.unique(y_train[prelabeled])) < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)

    # a set of labels is already labeled by the oracle
    y_train_labeled = np.array([None] * len(y_train))
    y_train_labeled[prelabeled] = y_train[prelabeled]

    # we are making a pool of the train data
    # the 'prelabeled' labels of the dataset are already labeled.
    trn_ds = Dataset(
        X_train,  # numpy array with train data (features)
        y_train_labeled  # numpy array with the labels
    )
    tst_ds = Dataset(X_test, y_test)

    return trn_ds, tst_ds, y_train


def main(args):

    acc_reviewer, acc_train, acc_test = [], [], []

    # get the data
    data = load_ptsd_data()


    trn_ds, tst_ds, y_train = split_train_test()

    # query strategy
    # https://libact.readthedocs.io/en/latest/libact.query_strategies.html
    # #libact-query-strategies-uncertainty-sampling-module
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())

    # The passive learning model. The model given in the query strategy is not
    # the same. Have a look at this one.
    model = LogisticRegression()

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Error')

    oracle = y_train[get_indices_labeled_entries(trn_ds)]
    review = [label for feat, label in trn_ds.get_labeled_entries()]
    reviewer_acc = accuracy_score(oracle, review)

    # # Train the model on the train dataset.
    # # Append the score (error).
    # model.train(trn_ds)
    # acc_reviewer = np.append(acc_reviewer, reviewer_acc)
    # acc_train = np.append(
    #     acc_train,
    #     model.model.score([x[0] for x in trn_ds.get_entries()], y_train)
    # )
    acc_train_tst = np.append(acc_test, model.score(tst_ds))

    query_num = np.arange(0, 1)
    p2, = ax.plot(query_num, acc_train_tst, 'r', label='Accuracy')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
               shadow=True, ncol=5)
    plt.show(block=False)

    # Give each label its name (labels are from 0 to n_classes-1)
    lbr = InteractivePaperLabeler(label_name=["0", "1"])

    for i in range(args.quota):

        # make a query from the pool
        ask_id = qs.make_query()
        print("asking sample from Uncertainty Sampling")

        # reshape the image to its width and height
        data_point = trn_ds.data[ask_id][0].reshape(8, 8)
        lb = lbr.label(data_point)

        # update the label in the train dataset
        trn_ds.update(ask_id, lb)

        # train the model again
        model.train(trn_ds)

        # compute accuracy of the reviewer
        oracle = y_train[get_indices_labeled_entries(trn_ds)]
        review = [label for feat, label in trn_ds.get_labeled_entries()]
        reviewer_acc = accuracy_score(oracle, review)

        # append the score to the model
        acc_reviewer = np.append(acc_reviewer, reviewer_acc)
        # acc_train = np.append(acc_train, model.model.score([x[0] for x in trn_ds.get_entries()], y_train))
        acc_test = np.append(acc_test, model.score(tst_ds))

        # adjust the limits of the axes
        ax.set_xlim((0, i + 1))
        ax.set_ylim((0, max(acc_test) + 0.2))

        query_num = np.arange(0, i + 2)
        # p0.set_xdata(query_num)
        # p0.set_ydata(acc_reviewer)
        # p1.set_xdata(query_num)
        # p1.set_ydata(acc_train)
        p2.set_xdata(query_num)
        p2.set_ydata(acc_test)

        plt.draw()

    input("Press any key to continue...")


if __name__ == '__main__':

    # parse all the arguments
    args = parser.parse_args()

    # start the active learning algorithm
    main(args)
