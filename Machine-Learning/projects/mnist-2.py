# Reaches 95.7% accuracy in 1 epoch

import keras
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)

# Uncomment for saved model
# model = load_model('/Users/bk/desktop/Machine-Learning/projects/mnist-2.h5')

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_grads=True, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), callbacks=[tbCallBack])
model.save('mnist-2.h5')
