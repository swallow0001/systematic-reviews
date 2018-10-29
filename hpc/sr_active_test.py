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

import os
import copy
import argparse
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.models import Model

# project dependencies
sys.path.insert(0, os.path.join('src', 'python'))  # path to the module.
# sys.path.insert(0, 'python')  # path to the m

from models.lstm_libact import LSTM_Libact
from query_strategies.uncertainty_sampling import UncertaintySampling
from query_strategies.random_sampling import RandomSampling

from sklearn.model_selection import StratifiedKFold


# demo utils
from utils import *
from config import *

# parse arguments if available
parser = argparse.ArgumentParser(description='Active learning parameters')
parser.add_argument("-T", default=1, type=int, help='Task number.')
parser.add_argument(
    "--dataset",
    type=str,
    default='ptsd',
    help="The dataset to use for training.")

# the number of iteration
parser.add_argument(
    "--quota", type=int, default=10, help="The number of queries")
parser.add_argument(
    '--interactive',
    dest='interactive',
    action='store_true',
    help="Interactive or not?")
parser.add_argument(
    '--no-interactive', dest='interactive', action='store_false')
parser.set_defaults(interactive=False)

parser.add_argument(
    "--init_included_papers",
    default=10,
    type=int,
    help='Initial number of included papers')

parser.add_argument(
    "--batch_size",
    default=10,
    type=int,
    help='Batch size')


parser.add_argument(
    '--model',
    type=str,
    default='LSTM',
    help="A deep learning model to use for classification.")
parser.add_argument(
    "--query_strategy", type=str, default='lc', help="The query strategy")


def select_prelabeled(labels, init_included, seed):
    included_indexes = np.where(labels[:, 1] == 1)[0]
    excluded_indexes = np.where(labels[:, 1] == 0)[0]
    included_number = min(init_included, len(included_indexes))
    #np.random.seed(seed)
    to_add_indx = np.random.choice(
        included_indexes, included_number, replace=False)
    to_add_indx = np.append(
        to_add_indx,
        np.random.choice(excluded_indexes, included_number, replace=False))
        
#     to_add_indx =   [1992,4016,3868,1233,1372,4126
# ,1531,3906,290,1438,2037,4636,532,3361,4145,2477,565,1940,2126,466]
#   
#     to_add_indx=[1992,4636,532,3361,4145,2477,565,1940,2126,466,825,2635,3430,3765,1584
# ,4759,4518,1712,1474,2007]

    return to_add_indx


def make_pool(X, y, prelabeled=np.arange(5)):
    """Function to split dataset into train and test dataset.

    Arguments
    ------

    prelabeled: list
        List of indices for which the label is already available.

    """
    print('y.shape',y.shape)
    
    y = y.argmax(axis=1)
    y_train_labeled = np.array([None] * len(y))
    y_train_labeled[prelabeled] = y[prelabeled]

    print('y_train_labeled[prelabeled]',y_train_labeled[prelabeled])
    return (X, y_train_labeled, y)

def get_unlabeled(y_with_trained):
    unlabeld = [i for i,j in enumerate(y_with_trained) if j is None]
    return unlabeld
    
def add_to_pool(y, y_with_trained,batch_size,q):
    unlabeld = get_unlabeled(y_with_trained)
    selected = np.random.choice(unlabeld,batch_size, replace=False)
    print('selected',selected)
#     all_ind=[231,4799,3960,3403,34,3895,1888,124,2550,1053,4961,679,133,1870,2665
# ,710,699,107,1396,879,1318,4795,487,2926,1538,4185,3287,4432,1059,2434
# ,3079,3896,4297,3323,4381,2601,937,929,5019,3421,983,62,1408,2376,591
# ,4148,2889,2338,2857,4425,3274,1197,1090,615,4289,1257,2881,1180,3398,89
# ,3520,4303,3897,614,3697,1389,3830,2220,3642,3976,4215,5038,3072,4136,2129
# ,878,4608,2774,4635,630,3143,3150,3932,1527,3821,1559,2751,3972,3591,1839
# ,4706,246,1208,3131,4478,2775,627,2673,3882,4032]   
#     
#     all_ind =[1093,2089,3271,1698,1123,1433,203,4417,249,1496,2608
# ,2129,3701,124,3601,516,1296,785,1090,3755,845,2029,4662,4930,506,4154
# ,600,968,1770,3578,1389,1786,1308,3082,4461,5066,626,4623,903,3798,2717
# ,4130,3141,2515,5006,133,4282,2025,1113,144,3934,1358,3941,1605,1810,4535
# ,4663,328,1316,2533,1939,3931,3774,4849,2553,3307,3827,2290,3070,1072,2223
# ,515,3030,151,3268,3702,983,757,1647,119,3845,769,2655,3093,3492,1449
# ,4043,2234,4476,1639,2426,3032,213,2047,948,1975,1564,2365,2886,4112,3301
# ,2423,3131,4054,1426,2514,3404,740,3278,64,764,1948,3576,293,3364,2577
# ,4987,3226,1584,4619,780,3176,3375,624,3463,634,359,407,3704,2735,2595
# ,5063,1553,3227,713,38,2172,4257,3627,4745,487,700,4785,1581,3659,1994
# ,894,2861,4088,993,3566,767,630,4158,4940,3387,2354,861,150,2287,3406
# ,686,89,4928,4035,3842,4809,2074,1051,3003,4425,2070,4664,3986,853,2263
# ,4523,3124,4985,2012,587,484,3089,1607,3136,3894,3460,3642,62,543,20
# ,5070,1065,2898,2309,3299,2303,1857,1258,2693,4994,3529,4818,1823,2758,3479
# ,2317,1888,1522,2166,4202,3,3365,1164,4402,3012,3641,2134,2445,2813,1541
# ,3027,898,690,3851,3237,2010,614,2203,4534,2656,2710,238,4711,4457,3829
# ,2734,679,3407,4472,1092,3946,400,452,3746,2424,3383,872,2707,3106,2866
# ,1929,662,1672,1658,4952,2988,4852,4173,882,1436,1450,4509,4073,143,936
# ,15,3867,4912,2949,34,341,356,2442,4546,3402,4252,3243,4069,5067,4606
# ,4061,615,420,828,2822,351,3071,3452,4776,3388,3582,3065,1659,1543,1353
# ,3499,2601,206,2473,1242,459,3815,246,4902,621,3482,2930,1980,2841,1287
# ,3767,3366,1984,3780,3037,3142,4441,4160,1703,2897,918,1180,5075,1608,447
# ,507,2677,2371,2712,2924,965,1890,792,1934,3219,4099,5050,2227,1303,627
# ,3478,4355,2465,3447,2773,935,4917,1153,463,1822,4403,181,738,1959,1328
# ,2069,4579,2244,1907,591,1418,1866,3902,107,3240,1680,1835,202,1816,1435
# ,3878,278,3311,1101,3960,1208,2058,548,4677,2102,4021,1464,710,4700,2637
# ,2222,2555,5058,2971,4265,4665,734,54,2899,2065,2339,4480,197,4214,1011
# ,3802,2011,2978,2232,1172,1766,1124,198,680,3446,4142,3932,1024,4220,2585
# ,441,4886,604,3046,2952,3255,1901,4927,3320,2032,1423,1596,4051,1594,4681
# ,2061,3656,4156,468,503,3668,3993,2740,4556,477,2491,2132,4270,4243,2597
# ,2950,1339,1501,1536,3575,573,1760,2673,3990,231,3421,3123,5076,1640,2123
# ,3620,3628,5054,3192,1029,4343,791,1078,2732,4533,3862,2008,3685,3638,2685
# ,3293,465,1280,3983,3711,3042,1022,3036,4347,3194,5009,2760,4083,4317
# ,3560,260,3359,5g64,4334,1983,1136,1154,4301,1960]

   #  #to test
   #  if q >= 10: return (None,None)
   #  s =q*10
   #  e =(q+1)*10
   #  selected =all_ind[s:e]
   # 
    y_with_trained[selected] = y[selected]
    print('y_with_trained[selected]',y_with_trained[selected])
 
    return (y_with_trained  , selected)
                   
    
    
def main(args):

    # Read dataset, labels and embedding layer from pickle file.
    pickle_fp = os.path.join(TEMP_DATA_DIR, args.dataset + '_pickle.pickle')
    with open(pickle_fp, 'rb') as f:
        data, labels, embedding_layer = pickle.load(f)

    # label the first batch (the initial labels)
    seed = 2016 + args.T
    prelabeled_index = select_prelabeled(labels, args.init_included_papers,
                                         seed)
    print('prelabeled_index', prelabeled_index)
    X, y_train_labeled, y = make_pool(data, labels, prelabeled=prelabeled_index)
    
    
    # get the model
    if args.model.lower() == 'lstm':
        deep_model = LSTM_Libact
        kwargs_model = {
            'backwards': True,
            'dropout': 0.4,
            'optimizer': 'rmsprop', 
            'max_sequence_length': 1000,
            'embedding_layer': embedding_layer
        }
    else:
        raise ValueError('Model not found.')

    # np.random.seed(seed)
    # tf.set_random_seed(seed)

    model = deep_model(**kwargs_model)
    
    # config = model._model.get_config()
    init_weights = model._model.get_weights()
#     print('init_weights.shape',len(init_weights))
#     print('init_weights[0]',init_weights[0])
# 
#     
#     # Give each label its name (labels are from 0 to n_classes-1)
#     if args.interactive:
#         lbr = InteractivePaperLabeler(label_name=["0", "1"])
#     else:
#         lbr = IdealLabeler(dataset=pool_ideal)
# # 
    result_df = pd.DataFrame({'label': y})
    
    query_i = 0
    labeled_indexes=prelabeled_index
    print( 'labeled_indexes',labeled_indexes)
    while query_i <= args.quota:

        # make a query from the pool
        print("Asking sample from pool with Uncertainty Sampling")
        
        # np.random.seed(seed)
        # tf.set_random_seed(seed)
        # graph = tf.Graph()
        # with tf.Session(graph=graph):
        # model = deep_model(**kwargs_model)
        # model._model.set_weights(init_weights)
    
        # train the model
        print('y_train_labeled[labeled_indexes]',y_train_labeled[labeled_indexes])
        print('labeled_indexes',labeled_indexes)
        # model._model = Model.from_config(config)
        # model._model =model._get_lstm_model(**kwargs_model)
        # print('after config')
        unlbls = get_unlabeled(y_train_labeled)
        model._train_model(None,X[labeled_indexes], y_train_labeled[labeled_indexes],X[unlbls],y[unlbls])

        

        pred = model.predict(X[unlbls])
        
        c_name = str(query_i)
        result_df[c_name] = -1
        print('is preds different?',pred[:, 1])
        result_df.loc[unlbls, c_name] = pred[:, 1]
        
        print('is there any mutation?')
        # print(set(labeled_indexes.tolist()) & set(unlbls))

        
        y_train_labeled, selected = add_to_pool(y, y_train_labeled,args.batch_size,query_i)
        #if selected is None: break
        print('y_train_labeled[selected]',y_train_labeled[selected])
        print('type(labeled_indexes)',type(labeled_indexes))
        print('type(selected)',type(labeled_indexes))

        labeled_indexes =np.append(labeled_indexes, selected)
        print( 'labeled_indexes',labeled_indexes)
# 
#         weights = model._model.get_weights()        
#         
#         # model._model.set_weights(init_weights)
# 
 #         
#         # update the query counter
        query_i += 1

    # save the result to a file
    output_dir = os.path.join(ACTIVE_OUTPUT_DIR, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    export_path = os.path.join(
        output_dir, 'dataset_{}_sr_lstm_active{}_q_{}.csv'.format(
            args.dataset, args.T, args.query_strategy))

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
