"""libact compatible version of the model"""

import numpy as np
from keras.layers import Dense, Input, LSTM as KERAS_LSTM
from keras.models import Model
from libact.base.interfaces import ProbabilisticModel


class LSTM(ProbabilisticModel):
    """
    """
    MAX_SEQUENCE_LENGTH = 1000

    def __init__(self, *args, **kwargs):
        self.model = self.get_lstm_model(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)

    def get_lstm_model(backwards, dropout, optimizer, max_sequence_length,
                       embedding_layer):
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = KERAS_LSTM(
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
