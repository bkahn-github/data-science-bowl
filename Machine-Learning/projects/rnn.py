import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding

from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
word_index = imdb.get_word_index()

print(x_train.shape)
print(y_train.shape)

print(x_train[0])
print(y_train[0])

def reverseIndex(sequence):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
    return decoded_review

x_train = pad_sequences(x_train, 50)
x_test = pad_sequences(x_test, 50)

print(x_train.shape)
print(x_train[0])

# model = Sequential()
# model.add(Embedding(1000, 64, input_length=50))
# model.add(GRU(64, return_sequences=True))
# model.add(GRU(32))
# model.add(Dense(1, activation='sigmoid'))
#
# model.summary()
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
#
# model.save('rnn.h5')

model = keras.models.load_model('/Users/bk/desktop/Machine-Learning/projects/rnn.h5')

predictionText = 'This is a bad review' # Add text to be analyzed
predictionWords = text_to_word_sequence(predictionText)
print(predictionWords)

predictionIndexes = [[word_index[w] for w in predictionWords if w in word_index]]
print(predictionIndexes)

paddedPredictionIndexes = pad_sequences(predictionIndexes, maxlen=50)
print(paddedPredictionIndexes)

FlattenedPaddedPredictionIndexes = np.array([paddedPredictionIndexes.flatten()])
print(FlattenedPaddedPredictionIndexes)

model.predict_classes(FlattenedPaddedPredictionIndexes)
