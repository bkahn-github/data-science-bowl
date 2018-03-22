# READ THE JUPYTER NOTEBOOK TO SEE HOW TO TRAIN ON RAW TEXT

import keras
import numpy as np

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, Dense

from keras.models import Sequential

from keras.datasets import imdb

max_features = 10000
maxlen = 20

# Get data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# print(x_train[0])

# Only use 20 most common words
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# print(x_train[0])

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on dataset
review = "good"
word_index = imdb.get_word_index()
review = [[word_index[w] for w in review if w in word_index]]
review = pad_sequences(review, maxlen=maxlen)

reviewScore = model.predict(review)
print(reviewScore)
