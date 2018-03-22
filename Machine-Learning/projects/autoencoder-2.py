import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28, 28, 1)

# Load model
# model = keras.models.load_model('/Users/bk/desktop/Machine-Learning/projects/autoencoder-2.h5')

# Train model
model = Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(8, (3,3), padding='same'))
model.add(MaxPooling2D((2,2)))


model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(8, (3,3), padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(16, (3,3), padding='same'))
model.add(Conv2D(1, (3, 3), padding='same'))
model.summary()

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])

model.fit(x_train, x_train, epochs=1, validation_data=(x_test, x_test))
model.save('autoencoder-2.h5')

test = model.predict(x_test)
test = test[0]
test = test.reshape(28, 28)

x_test = x_test[0]
x_test = x_test.reshape(28, 28)

plt.imshow(test)
plt.show()

plt.imshow(x_test)
plt.show()
