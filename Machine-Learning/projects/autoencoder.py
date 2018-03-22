import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)

# Switch for training
# x_test = x_test.reshape(x_test.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 1, 784)

# Load model
model = keras.models.load_model('/Users/bk/desktop/Machine-Learning/projects/autoencoder.h5')

# Train model
# model = Sequential()
# model.add(Dense(784, input_shape=(784,)))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(256))
# model.add(Dense(784))
#
# model.summary()
#
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])
#
# model.fit(x_train, x_train, epochs=2, validation_data=(x_test, x_test))
# model.save('autoencoder.h5')

img = x_test[2]

test = model.predict(img)

test = test.reshape(28, 28)
img = img.reshape(28, 28)

plt.imshow(img, cmap='gray')
plt.show()

plt.imshow(test, cmap='gray')
plt.show()
