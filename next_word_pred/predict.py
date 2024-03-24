import pickle
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
import numpy as np

class LSTM_next_word_pred:

    def __init__(self):
        with open('./next_word_pred/model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            self.model = tf.keras.models.load_model('./next_word_pred/model/LSTMWordList.keras')


    def inputTransformations(self,line,tokenizer):
        # created Tokenized NGram
        input_sequences = []
        token_list = tokenizer.texts_to_sequences(line)[0]
        input_sequences = [token_list]
        max_seq_length = 11 # 11 instead of 12 because not removing prediction var
        input_seqs = np.array(pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre'))

        # print(input_seqs[:5])
        x_values, labels = input_seqs[:, ], input_seqs[:, ]
        return x_values

    def predict(self,string):
        if type(string) != str: 
            raise ValueError('Prediction was not passed string format, please pass predictor string format.')
        tokenizer = self.tokenizer
        word_index = tokenizer.word_index
        model = self.model
        x_ngram = self.inputTransformations([string],tokenizer)
        # print(x_ngram)
        predicted_words = [ [list(word_index.keys())[list(word_index.values()).index(i)]  for i in tf.math.top_k(model.predict(x_ngram , verbose=1), k =5)[1][0]] ]
        return predicted_words [0]