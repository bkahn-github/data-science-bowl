import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

# Scaling
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std

model = Sequential()
model.add(Dense(32, input_shape=(13, )))
model.add(Dense(12))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=25)

model.evaluate(x_test, y_test)
