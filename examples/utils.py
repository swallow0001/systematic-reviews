
from os import path

import pandas as pd
import numpy as np


word2vec_filePath =path.join("word2vec", "wiki.en.vec")
def load_ptsd_data(fp=path.join("data", "ptsd_review", "csv", "schoot-lgmm-ptsd-traindata.csv")):
    """Load ptsd papers and their labels.

    The number of records is 5077. The following labels are included after the
    systematic review: 218,260,466,532,564,565,699,1136,1154,1256,1543,1552,19
    40,1960,1983,1984,1992,2005,2126,2477,2552,2708,2845,2851,3189,3280,3359,3
    361,3487,3542,3560,3637,3913,4048,4049,4145,4301,4334,4491,4636

    """

    # read the data of the file location given as argument to this function
    df = pd.read_csv(fp)

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstract'].fillna(''))
    labels = df["included_final"]

    return texts.values, labels.values


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
