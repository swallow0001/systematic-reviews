#!/usr/bin/env python3

import os
import sys
import argparse

# project dependencies
sys.path.insert(0, os.path.join('src', 'python'))  # path to the module.

from models.textmanager import TextManager
from models.embedding import Word2VecEmbedding


# demo utils
from utils import *
from config import *

# parse arguments if available
parser = argparse.ArgumentParser(description='Dataset preparation')
parser.add_argument(
    "--dataset",
    type=str,
    default='ptsd',
    help="The dataset to use for training.")

max_num_words = 20000
max_sequence_length = 1000


if __name__ == '__main__':

    # parse all the arguments
    args = parser.parse_args()

    # Prepare the datasets. Dataset, labels and embedding layer are stored to
    # disk in pickle file.
    pickle_fp = os.path.join(TEMP_DATA_DIR, args.dataset + '_pickle.pickle')

    # load the dataset from disk
    if args.dataset == 'ptsd':
        texts, lbls = load_ptsd_data()
    else:
        texts, lbls = load_drug_data(args.dataset)

    # get the texts and their corresponding labels

    textManager = TextManager(
        max_num_words=max_num_words,
        max_sequence_length=max_sequence_length
    )
    data, labels, word_index = textManager.sequence_maker(texts, lbls)

    if not os.path.exists(TEMP_DATA_DIR):
        os.makedirs(TEMP_DATA_DIR)

    embedding = Word2VecEmbedding(word_index, max_num_words,
                                  max_sequence_length)
    embedding.load_word2vec_data(GLOVE_PATH)
    embedding_layer = embedding.build_embedding()

    with open(pickle_fp, 'wb') as f:
        pickle.dump((data, labels, embedding_layer), f)
