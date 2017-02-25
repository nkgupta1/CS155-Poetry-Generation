#!/usr/bin/env 

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM #rnn
from keras.callbacks import ModelCheckpoint

filename = 'models/rnn-19-2.4038.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
