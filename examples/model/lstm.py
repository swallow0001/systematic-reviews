
import numpy as np
import os
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard


from libact.base.interfaces import ProbabilisticModel


class LSTM_Model(ProbabilisticModel):

    """
    """

    
    def __init__(self, *args, **kwargs):
        for arg in args:
            print(' another arg: ',arg)
            
        if kwargs is not None:
            for key, value in kwargs.items():
                print ("%s == %s" %(key,value))
            
        self._model = self._get_lstm_model(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self._model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self._model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self._model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self._model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self._model.predict_proba(feature, *args, **kwargs)
        
               
    def _get_lstm_model(self, backwards, dropout,optimizer,max_sequence_length,embedding_layer ):
        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
    
        x = LSTM(10,input_shape=(max_sequence_length,),  go_backwards=backwards, dropout=dropout)(embedded_sequences)
        x = Dense(128,activation='relu')(x)
        output = Dense(2, activation='softmax')(x)
    
        model_lstm = Model(inputs=sequence_input, outputs=output)
    
        model_lstm.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['acc'])
    
        model_lstm.summary()
        return model_lstm
        

    def get_scores_pred(self,pred, threshhold):
        y_classes = ( [0 if x < threshhold else 1 for x in pred[:,1]])
        y_value = y_val.argmax(axis=1)
        (tn, fp, fn, tp) = sklearn.metrics.confusion_matrix(y_value,y_classes).ravel()
        return (tn, fp, fn, tp)


