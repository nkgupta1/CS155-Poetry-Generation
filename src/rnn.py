import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU, SimpleRNN #rnn
import pickle


datafile = 'shakespeare'
Xmap, X, Y = pickle.load(open('Xm.X.Y_' + datafile + '.pkl', 'rb'))
X, Y = np.array(X), np.array(Y)

model = Sequential()
model.add(LSTM(32, input_dim=1, input_length=5))
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='mse',optimizer='adagrad', metrics=['accuracy'])
score, acc = model.evaluate(X, Y, batch_size=20, verbose=1)

print(score, acc)