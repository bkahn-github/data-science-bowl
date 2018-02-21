import os

import numpy as np
from tqdm import tqdm
import imageio

from skimage.transform import resize

if os.environ['platform'] == 'surface':
    train_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/test/'
else:    
    train_path = '../../.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '../../.kaggle/competitions/data-science-bowl-2018/test/'

train_ids = next(os.walk(train_path))[1]
test_ids = next(os.walk(test_path))[1]

def load_train_data():
    x_train = np.zeros((len(train_ids), 128, 128, 3))
    y_train = np.zeros((len(train_ids), 128, 128, 1))

    for i, index in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
        img = imageio.imread(train_path + '/' + index + '/images/' + index + ".png")
        img = img[:,:,:3]
        img = resize(img, (128, 128, 3), mode='constant')
            
        masks = np.zeros((128, 128, 1))
        mask_files = next(os.walk(train_path + index + '/masks/'))[2]
        
        for mask in mask_files:
            mask = imageio.imread(train_path + '/' + index + '/masks/' + mask)
            mask = resize(mask, (128, 128, 1), mode='constant')
            masks = np.maximum(masks, mask)
        
        x_train[i] = img
        y_train[i] = masks

    return x_train, y_train

def load_test_data():
    x_test = np.zeros((len(test_ids), 128, 128, 3))

    for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
        img = img[:,:,:3]
        img = resize(img, (128, 128, 3), mode='constant')
        
        x_test[i] = img

    return x_test

def load_test_image_sizes():
    x_test_sizes = []

    for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
        x = img.shape[0]
        y = img.shape[1]
        
        x_test_sizes.append([x, y])

    return x_test_sizes