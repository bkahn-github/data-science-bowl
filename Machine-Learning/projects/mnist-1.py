import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)

model = Sequential()

model.add(Dense(32, input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test))
