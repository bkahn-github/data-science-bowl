import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100, mnist

(cifar, _), (_, _) = cifar10.load_data()
(cifar2, _), (_, _) = cifar100.load_data()
(mnist, _), (_, _) = mnist.load_data()

cifar = cifar[1]
cifar2 = cifar2[0]
mnist = mnist[0]

plt.imshow(cifar ,interpolation="bicubic")
plt.show()

plt.imshow(cifar2 ,interpolation="bicubic")
plt.show()

plt.imshow(mnist ,interpolation="bicubic")
plt.show()
