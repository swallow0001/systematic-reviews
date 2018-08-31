from sklearn.model_selection import train_test_split
from os import path, listdir

import pandas as pd
import numpy as np
import pickle

DATA_PATH = path.join("..", "data")
PTSD_PATH = path.join(DATA_PATH, "ptsd_review", "csv",
                      "schoot-lgmm-ptsd-traindata.csv")
DRUG_DIR = path.join(DATA_PATH, "drug_class_review")
WORD2VEC_PATH = path.join("..", "word2vec", "wiki.en.vec")
PICKLE_PATH = path.join(DATA_PATH, "pickle")
PARAMETER_PATH = path.join(DATA_PATH, "parameters.txt")


def load_ptsd_data():
    """Load ptsd papers and their labels.

    The number of records is 5077. The following labels are included after the
    systematic review: 218,260,466,532,564,565,699,1136,1154,1256,1543,1552,19
    40,1960,1983,1984,1992,2005,2126,2477,2552,2708,2845,2851,3189,3280,3359,3
    361,3487,3542,3560,3637,3913,4048,4049,4145,4301,4334,4491,4636

    """

    print('PTSD_PATH:', PTSD_PATH)
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
    print("relative number of positive labels: {}".format(
        labels.sum() / labels.shape[0]))

    return texts.values, labels.values


def list_drug_datasets():
    """Get a list of all available drug datasets."""

    return [dataset[:-4] for dataset in listdir(DRUG_DIR)]


def split_data(data, labels, train_size, init_positives, seed):
    """split dataset to train and test datasets
    """
    if train_size >= len(data):
        print('invalid train size')
        return

    validation_split = 1 - (train_size / len(data))

    x_train, x_val, y_train, y_val = train_test_split(
        data,
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels)

    # add added_positives positive paper to training dataset
    if init_positives > 0:
        # number of included papers in train dataset
        incl = len(np.where(y_train[:, 1] == 1)[0])

        # number of included papers should be added to train dataset
        to_add = init_positives - incl

        # index of all included papers in test dataset
        positive_indx = np.where(y_val[:, 1] == 1)[0]

        if (to_add > 0) and (len(positive_indx) >= to_add):

            np.random.seed(seed)
            to_add_indx = np.random.choice(
                positive_indx, to_add, replace=False)

            y_train = np.vstack((y_train, y_val[to_add_indx]))
            x_train = np.vstack((x_train, x_val[to_add_indx]))

            x_val = np.delete(x_val, to_add_indx, 0)
            y_val = np.delete(y_val, to_add_indx, 0)

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
