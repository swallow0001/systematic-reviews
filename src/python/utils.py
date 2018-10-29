
# Cpython dependencies
import os
import sys
import pickle

# external dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join('..', 'python'))

# project dependencies
from config import DATA_DIR, PTSD_PATH, DRUG_DIR, DEPRESSION_PATH

# Global variables
# word2vec_filePath = os.path.join("word2vec", "wiki.en.vec")


PARAMETER_PATH = os.path.join(DATA_DIR, "parameters.txt")


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
    fp = os.path.join(DRUG_DIR, name + ".csv")

    try:
        df = pd.read_csv(fp)
    except FileNotFoundError:
        raise ValueError("Dataset with name {} doesn't exist".format(name))

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstracts'].fillna(''))
    labels = (df["label2"] == "I").astype(int)

    print("number of positive labels: {}".format(labels.sum()))
    print("relative number of positive labels: {}".format(
        labels.sum() / labels.shape[0]))

    return texts.values, labels.values

def load_depression_data():
    """Load adults_depression papers and their labels.

    """

    # read the data of the file location given as argument to this function
    df = pd.read_csv(DEPRESSION_PATH, encoding= "latin-1")

    # make texts and labels
    texts = (df['title'].fillna('') + ' ' + df['abstracts'].fillna(''))
    labels = df["label"]

    return texts.values, labels.values

def list_drug_datasets():
    """Get a list of all available drug datasets."""

    return [dataset[:-4] for dataset in os.listdir(DRUG_DIR)]


# def load_word2vec_data(fp =os.path.join("word2vec", "wiki.en.vec"),embedding_dim=300):
#         """Load word2vec data. fp =os.path.join("word2vec", "wiki.en.vec")
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

def split_data(data, labels, train_size, init_positives, seed):
    """split dataset to train and test datasets
    """
    if train_size >= len(data):
        print('invalid train size')
        return
     
    print('shape(data)',data.shape) 
    print('shape(labels)',labels.shape)
      
    index_all= [i for i,j in enumerate(data)]
    data =np.c_[data,index_all]
    
    # select init_positive cases from the entire dataset
    if init_positives > 0:
        ##index of all included papers in the entire dataset
        positive_indx = np.where(labels[:, 1] == 1)[0]

        np.random.seed(seed)
        #to test
        to_add_indx = np.random.choice(positive_indx, init_positives, replace=False)
        
        y_train_init = labels[to_add_indx]
        x_train_init = data[to_add_indx]
    
        data = np.delete(data, to_add_indx, 0)
        labels = np.delete(labels, to_add_indx, 0)

    train_size = train_size - init_positives
    validation_split = 1 - (train_size / len(data))

    x_train, x_val, y_train, y_val = train_test_split(
        data,
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels
    )

    x_train = np.vstack((x_train, x_train_init))
    y_train = np.vstack((y_train, y_train_init))

    print("x_train[:,1000]",x_train[:,1000])
    x_train = np.delete(x_train,1000,1)
    x_val = np.delete(x_val,1000,1)
    return (x_train, x_val, y_train, y_val)


def dump_pickle(obj, file_name):
    # open a file, where you want to store the data
    file_pickle = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(obj, file_pickle)

    # close the file
    file_pickle.close()


def load_pickle(file_name):
    # open a file, where you want to store the data
    file_pickle = open(file_name, 'rb')

    # dump information to that file
    obj = pickle.load(file_pickle)

    # close the file
    file_pickle.close()
    return obj


def save_parameters(names, parameters):
    #     with open(PARAMETER_PATH,"w") as outputfile:
    #         # writer = csv.writer(outputfile,lineterminator='\n')
    #         # writer.writerow(names)
    #         # for i, parm in enumerate(parameters):
    #         #       writer.writerow([parm[0], parm[1], parm[2], parm[3]])
    #
    #         outputfile.write(','.join(names)+'\n')
    #         for i, param in enumerate(parameters):
    #               row = ','.join(str(x) for x in param)+'\n'
    #               outputfile.write(row)

    params = pd.DataFrame(parameters, columns=names)
    params.to_csv(PARAMETER_PATH)


def read_parameters():

    return list(pd.read_csv(PARAMETER_PATH, index_col=0).values)
