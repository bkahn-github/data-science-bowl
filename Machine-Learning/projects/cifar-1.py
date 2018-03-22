import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)

plt.imshow(x_train[1])
plt.show()

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_grads=True, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), callbacks=[tbCallBack])
