
from os import path, listdir

import pandas as pd
import numpy as np

word2vec_filepath = path.join("word2vec", "wiki.en.vec")

DATA_PATH = "data"
PTSD_PATH = path.join(DATA_PATH, "ptsd_review", "csv",
                      "schoot-lgmm-ptsd-traindata.csv")
DRUG_DIR = path.join(DATA_PATH, "drug_class_review")


def load_ptsd_data():
    """Load ptsd papers and their labels.

    The number of records is 5077. The following labels are included after the
    systematic review: 218,260,466,532,564,565,699,1136,1154,1256,1543,1552,19
    40,1960,1983,1984,1992,2005,2126,2477,2552,2708,2845,2851,3189,3280,3359,3
    361,3487,3542,3560,3637,3913,4048,4049,4145,4301,4334,4491,4636

    """

    # read the data of the file location given as argument to this function
    df = pd.read_csv(PTSD_PATH)

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstract'].fillna(''))
    labels = df["included_final"]

    return texts.values, labels.values


def load_drug_data(name):
    """Load drug datasets and their labels.

    params
    ------
    name: str
        The name of the dataset (should match with file name)
    """

    print("load drug dataset: {}".format(name))

    # create file path based on the argument name.
    fp = path.join(DRUG_DIR, name + ".csv")

    try:
        df = pd.read_csv(fp)
    except FileNotFoundError:
        raise ValueError("Dataset with name {} doesn't exist".format(name))

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstracts'].fillna(''))
    labels = (df["label2"] == "I").astype(int)

    print("number of positive labels: {}".format(labels.sum()))
    print("relative number of positive labels: {}".format(labels.sum()/labels.shape[0]))

    return texts.values, labels.values


def list_drug_datasets():
    """Get a list of all available drug datasets."""

    return [dataset[:-4] for dataset in listdir(DRUG_DIR)]

# def load_word2vec_data(fp =path.join("word2vec", "wiki.en.vec"),embedding_dim=300):
#         """Load word2vec data. fp =path.join("word2vec", "wiki.en.vec")
#         """
#         # Build index mapping words in the embeddings set
#         # to their embedding vector
#
#         print('Indexing word vectors.')
#         embeddings_index = {}
#
#         with open(fp, encoding='utf8') as f:
#             for line in f:
#
#                 values = line.split()
#                 split_on_i = len(values) - embedding_dim
#                 word = ' '.join(values[0:split_on_i])
#                 coefs = np.asarray(values[split_on_i:], dtype='float32')
#                 embeddings_index[word] = coefs
#         print('Found %s word vectors.' % len(embeddings_index))
#         return embeddings_index
