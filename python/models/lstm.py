import numpy as np
import os
import sklearn
from scipy import stats
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard


class LSTM_Model():
    """
    """
    # training arguments
    batch_size = 25
    epoch_no = 10

    def __init__(self, *args, **kwargs):
        self._model = self._get_lstm_model(*args, **kwargs)

    def train(self, *args, **kwargs):
        self._train_model(*args, **kwargs)


    def score(self, testing_dataset, *args, **kwargs):
        label = np.array(args[0])
        allowed_FN = args[1]
        pred = self._get_pred(testing_dataset)
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

    def _get_pred(self, testing_dataset):
        prediction = self._model.predict(np.array(testing_dataset))
        return prediction

    def _get_scores_threshhold(self, pred, label, threshhold):

        y_classes = ([0 if x < float(threshhold) else 1 for x in pred[:, 1]])
        y_value = label.argmax(axis=1)
        (tn, fp, fn, tp) = sklearn.metrics.confusion_matrix(
            y_value, y_classes).ravel()
        # recall = tp/(tp+fn)
        # fp_rate = fp/ (fp+tn)
        # return (recall,fp_rate)
        return (tn, fp, fn, tp)

    def _train_model(self, *args):

        x_train = np.array(args[0])
        y_train = np.array(args[1])
        x_val = np.array(args[2])
        y_val = np.array(args[3])

        weights = [1 / y_train[:, 0].mean(), 1 / y_train[:, 1].mean()]

        self._model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epoch_no,
            validation_data=(x_val, y_val),
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
        print('threshholds', threshholds)
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
