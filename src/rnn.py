#!/usr/bin/env python3
"""
Recurrent Neural Network on sonnets.
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM #rnn
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def load():
    # load all text as one unit
    text = ''

    with open('../data/shakespeare.txt') as f:
        for line in f:
            line = line.strip()

            if line == '' or line.isdigit():
                continue

            text += line.lower() + '\n'

    # parse characters into numbers
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    return text, char_to_int, int_to_char

seq_length = 40
X = []
Y = []

text, char_to_int, int_to_char = load()

# parse using a moving frame
for i in range(len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i+seq_length]
    X.append([char_to_int[c] for c in seq_in])
    Y.append(char_to_int[seq_out])

# reshape for keras
X = np.reshape(X, (len(X), seq_length, 1))
# normalize
X = X / len(char_to_int)

# make one hot vector for the output
Y = np_utils.to_categorical(Y)

# RNN Network
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, consume_less='cpu', unroll=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, consume_less='cpu', unroll=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, consume_less='cpu', unroll=True))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# save model progress
filename = 'models/rnn-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, period=5, 
    save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print(X.shape)
# fit the model
model.fit(X, Y, nb_epoch=100, batch_size=128, callbacks=callbacks_list)

#model.save('final_model_epochs.h5')