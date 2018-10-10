"""libact compatible version of the model"""

import numpy as np
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from libact.base.interfaces import ProbabilisticModel

import tensorflow as tf
from keras import backend as K

import sklearn
from scipy import stats


class LSTM_Libact(ProbabilisticModel):
    """
    """
    MAX_SEQUENCE_LENGTH = 1000

    # training arguments
    batch_size = 25
    epoch_no = 10

    def __init__(self, *args, **kwargs):
        self._model = self._get_lstm_model(*args, **kwargs)

    def train(self,dataset, *args, **kwargs):
        self._train_model(dataset,*args, **kwargs)

    def predict(self, feature, *args, **kwargs):
        #return self.model.predict(feature, *args, **kwargs)
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        session = tf.Session(config=config)
        K.set_session(session)

        session.run(tf.global_variables_initializer())

        return self._model.predict(np.array(feature))

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        # return self._get_pred(feature)
        return self.predict(feature)
        
    def score(self, testing_dataset, *args, **kwargs):
        label = np.array(args[0])
        allowed_FN = args[1]
        # pred = self._get_pred(testing_dataset)
        pred = self.predict(testing_dataset)
        return self._get_scores(pred, label, allowed_FN)

    def _get_lstm_model(self, backwards, dropout, optimizer,
                        max_sequence_length, embedding_layer):
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = LSTM(
            10,
            input_shape=(max_sequence_length, ),
            go_backwards=backwards,
            dropout=dropout)(embedded_sequences)
        x = Dense(128, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)

        model_lstm = Model(inputs=sequence_input, outputs=output)

        model_lstm.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

        model_lstm.summary()
        return model_lstm

    # def _get_pred(self, testing_dataset):
    #     prediction = self._model.predict(np.array(testing_dataset))
    #     return prediction

    def _get_scores_threshhold(self, pred, label, threshhold):

        y_classes = ([0 if x < float(threshhold) else 1 for x in pred[:, 1]])
        #y_value = label.argmax(axis=1)
        (tn, fp, fn, tp) = sklearn.metrics.confusion_matrix(
            label, y_classes).ravel()
        # recall = tp/(tp+fn)
        # fp_rate = fp/ (fp+tn)
        # return (recall,fp_rate)
        return (tn, fp, fn, tp)

    def _train_model(self, dataset,*args):

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        session = tf.Session(config=config)
        K.set_session(session)

        session.run(tf.global_variables_initializer())

        if len(args)>1:
            x_train = np.array(args[0])
            y_train = np.array(args[1])
            x_val = np.array(args[2])
            y_val = np.array(args[3])
    
            if y_train.ndim==1:    
                y_train = to_categorical(np.asarray(y_train))

            if y_val.ndim==1:    
                y_val = to_categorical(np.asarray(y_val ))
            
    
            # weights = [1 / y_train[:, 0].mean(), 1 / y_train[:, 1].mean()]
            weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
            self._model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epoch_no,
                validation_data=(x_val, y_val),
                shuffle=True,
                class_weight=weights,
                verbose=0)

        else:
            # dataset = args[0]
            #            
            # x_train = [x[0] for x in dataset.data]    
            # y_train = [x[1] for x in dataset.data]
            # 
            # x_train = np.array(x_train)
            # y_train = np.array(y_train)
            x_train, y_train_ = dataset.format_sklearn()
            
            if y_train_.ndim==1:    
                y_train = to_categorical(np.asarray(y_train_))
            else:
                y_train = y_train_

            weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
    
            #weights = [5033 / 5077,  40/5077]
            self._model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epoch_no,
                shuffle=True,
                class_weight=weights,
                verbose=0)
    def _get_threshholds(self, pred):
        desc = stats.describe(pred[:, 1])
        (min_prob, max_prob) = desc.minmax
        print('min_prob,max_prob', min_prob, max_prob)
        step = (max_prob - min_prob) / 25
        threshholds = [i for i in np.arange(max_prob, min_prob - step, -step)]
        threshholds += [min_prob]
        return threshholds

    def _get_scores(self, pred, label, allowed_FN):
        threshholds = self._get_threshholds(pred)
        best_tn, best_fp, best_fn, best_tp = self._get_scores_threshhold(
            pred, label, threshholds[0])
        ## calculate scores for each threshhold
        for i, threshhold in enumerate(threshholds):
            tn, fp, fn, tp = self._get_scores_threshhold(
                pred, label, threshhold)
            if (fn <= allowed_FN) and (best_fn > allowed_FN) and (threshhold >
                                                                  0):
                best_tn = tn
                best_fp = fp
                best_fn = fn
                best_tp = tp
            else:  #when fn meets the requirement, we select a smaller threshhold just when it decreases fp.
                if (fn <= allowed_FN) and (best_fp > fp) and (threshhold > 0):
                    best_tn = tn
                    best_fp = fp
                    best_fn = fn
                    best_tp = tp
        return (float(best_tn), float(best_fp), float(best_fn), float(best_tp),
                pred.tolist())
