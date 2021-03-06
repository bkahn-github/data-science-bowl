import os
import random
import imageio
import PIL
from tqdm import tqdm
import numpy as np
from skimage.transform import resize

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

seed = 42
random.seed = seed
np.random.seed(seed=seed)

if os.environ.get('platform') == 'surface':
    train_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/test/'
else:    
    train_path = '../../.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '../../.kaggle/competitions/data-science-bowl-2018/test/'

train_ids = next(os.walk(train_path))[1]
test_ids = next(os.walk(test_path))[1]

def load_train_data(train_ids=train_ids, train_path=train_path):
  x_train = np.zeros((len(train_ids), 256, 256, 3), dtype=np.uint8)
  y_train = np.zeros((len(train_ids), 256, 256, 1), dtype=np.bool)
  
  for i, index in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    img = imageio.imread(train_path + '/' + index + '/images/' + index + ".png")
    img = img[:,:,:3]
    img = resize(img, (256, 256), mode='constant', preserve_range=True)

    masks = np.zeros((256, 256, 1), dtype=np.bool)
    mask_files = next(os.walk(train_path + index + '/masks/'))[2]

    for mask in mask_files:
      mask = imageio.imread(train_path + '/' + index + '/masks/' + mask)
      mask = np.expand_dims(resize(mask, (256, 256), mode='constant', preserve_range=True), axis=-1)
      masks = np.maximum(masks, mask)
        
    x_train[i] = img
    y_train[i] = masks

  return x_train, y_train

def load_test_data(test_ids=test_ids, test_path=test_path):
  x_test = np.zeros((len(test_ids), 256, 256, 3), dtype=np.uint8)

  for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
    item = {}
    
    img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
    img = img[:,:,:3]
    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    
    x_test[i] = img

  return x_test

def load_test_image_sizes(test_ids=test_ids, test_path=test_path):
    x_test_sizes = []

    for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
        x = img.shape[0]
        y = img.shape[1]
        
        x_test_sizes.append([x, y])

    return x_test_sizes

def load_data(train_val_split=0.2, batch_size=4, seed=seed):
  x_train, y_train = load_train_data()
  x_test = load_test_data()
  x_test_sizes = load_test_image_sizes()

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=train_val_split, random_state=0)

  x_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True)

  y_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True)

  x_datagen.fit(x_train, augment=True, seed=seed)
  y_datagen.fit(y_train, augment=True, seed=seed)

  x_train_augmented = x_datagen.flow(x_train, batch_size=4, shuffle=True, seed=seed)
  y_train_augmented = y_datagen.flow(y_train, batch_size=4, shuffle=True, seed=seed)

  train_datagen = zip(x_train_augmented, y_train_augmented)

  return x_train, y_train, x_val, y_val, train_datagen