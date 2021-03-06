#!/usr/bin/env python3
"""
Recurrent Neural Network on sonnets.
"""

import numpy as np

from keras.models import Sequential, load_model
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

seq_length = 60
dataX = []
Y = []
train_model = False
generate = True
resume_training = False

text, char_to_int, int_to_char = load()

# parse using a moving frame
for i in range(len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i+seq_length]
    dataX.append([char_to_int[c] for c in seq_in])
    Y.append(char_to_int[seq_out])

# reshape for keras
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / len(char_to_int)

# make one hot vector for the output
Y = np_utils.to_categorical(Y)

# fit the model
if train_model:
    # RNN Network
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), \
        consume_less='cpu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False, consume_less='cpu'))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))

    filename = 'models/rnn-256x256-{epoch:02d}-{loss:.4f}.hdf5'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, 
        save_best_only=True, mode='min', period=10)
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, Y, nb_epoch=1000, batch_size=128, callbacks=callbacks_list)

if resume_training:
    filename = 'models/rnn-256x256-149-1.7294.hdf5'
    model = load_model(filename)
    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    filename = 'models/rnn-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, 
        save_best_only=True, mode='min', period=10)
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, Y, nb_epoch=1000, batch_size=128, callbacks=callbacks_list)

if generate:
    filename = 'models/rnn-128x64-60-10-2.4089.hdf5'
    print('loading', filename)
    model = load_model(filename)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # pick a random seed
    pattern = dataX[np.random.randint(0, len(dataX) - 1)]
    print("Seed:", "".join([int_to_char[i] for i in pattern]))
    print('-'*80)
    newline_count = 0

    while True:
        # shape for rnn
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / len(int_to_char)
        # get generated character and convert to character
        prediction = model.predict(x, verbose=0)
        ind = np.argmax(prediction)
        result = int_to_char[ind]
        # print out the character
        print(result, flush=True, end='')

        if result == '\n':
            newline_count += 1
        if newline_count == 100:
            break
            
        # update the input pattern
        pattern.append(ind)
        pattern = pattern[1:]


#model.save('final_model_epochs.h5')