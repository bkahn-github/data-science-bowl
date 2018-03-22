import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding

from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000)

y_train = keras.utils.to_categorical(y_train, 46)
y_test = keras.utils.to_categorical(y_test, 46)

word_index = reuters.get_word_index()

print(x_train.shape)
print(y_train.shape)

print(x_train[0])
print(y_train[0])

def reverseIndex(sequence):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_category = ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
    return decoded_category

x_train = pad_sequences(x_train, 100)
x_test = pad_sequences(x_test, 100)

print(x_train.shape)
print(x_train[0])

# model = Sequential()
# model.add(Embedding(1000, 64, input_length=100))
# model.add(GRU(64, return_sequences=True))
# model.add(GRU(32))
# model.add(Dense(46, activation='softmax'))
#
# model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', embeddings_freq=1, histogram_freq=1, write_grads=True, write_graph=True, write_images=True)
# model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tbCallBack])
#
# model.save('rnn-2.h5')

model = keras.models.load_model('/Users/bk/desktop/Machine-Learning/projects/rnn-2.h5')
