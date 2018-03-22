import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# word_index = imdb.get_word_index()

# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# print(decoded_review)

x_train = pad_sequences(x_train, 10000)
x_test = pad_sequences(x_test, 10000)

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
