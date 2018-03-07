import os
import imageio
import PIL
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

if os.environ.get('platform') == 'surface':
    train_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '/home/bilal/.kaggle/competitions/data-science-bowl-2018/test/'
else:    
    train_path = '../../.kaggle/competitions/data-science-bowl-2018/train/'
    test_path = '../../.kaggle/competitions/data-science-bowl-2018/test/'

train_ids = next(os.walk(train_path))[1]
test_ids = next(os.walk(test_path))[1]

def load_train_data(train_ids=train_ids, train_path=train_path):
  items = []
  
  for i, index in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    item = {}

    img = imageio.imread(train_path + '/' + index + '/images/' + index + ".png")
    img = img[:,:,:3]

    masks = np.zeros((img.shape[0], img.shape[1]))
    mask_files = next(os.walk(train_path + index + '/masks/'))[2]

    for mask in mask_files:
      mask = imageio.imread(train_path + '/' + index + '/masks/' + mask)
      masks = np.maximum(masks, mask)
        
    item['img'] = torch.from_numpy(img)
    item['mask'] = torch.from_numpy(masks)
    
    items.append(item)
   
  return items

def load_test_data(test_ids=test_ids, test_path=test_path):
  items = []    
  for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
    item = {}
    
    img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
    img = img[:,:,:3]
    
    item['img'] = torch.from_numpy(img)
    
    items.append(item)

  return items

def load_test_image_sizes(test_ids=test_ids, test_path=test_path):
    x_test_sizes = []

    for i, index in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imageio.imread(test_path + '/' + index + '/images/' + index + ".png")
        x = img.shape[0]
        y = img.shape[1]
        
        x_test_sizes.append([x, y])

    return x_test_sizes

class TrainDataset():
  def __init__(self, data, x_transform, y_transform):
    self.data = data
    self.x_transform = x_transform
    self.y_transform = y_transform
  
  def __getitem__(self, index):
    data = self.data[index]

    img = data['img'].numpy()
    mask = data['mask'][:,:,None].byte().numpy()
    
    img = self.x_transform(img)
    mask = self.y_transform(mask)
    
    return img, mask
  
  def __len__(self):
    return(len(self.data))
  
class TestDataset():
  def __init__(self, data, x_transform):
    self.data = data
    self.x_transform = x_transform
  
  def __getitem__(self, index):
    data = self.data[index]

    img = data['img'].numpy()
    img = self.x_transform(img)
    
    return img
  
  def __len__(self):
    return(len(self.data))
  
x_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5] ,std=[0.5,0.5,0.5])
])

y_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128),interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()
])

def load_data(train_val_split=0.2, batch_size=4):
  train = load_train_data()
  test = load_test_data()
  x_test_sizes = load_test_image_sizes()

  train, val = train_test_split(train, test_size=train_val_split, random_state=0)

  train_dataset = TrainDataset(train, x_transform, y_transform) 
  val_dataset = TrainDataset(val, x_transform, y_transform)
  test_dataset = TestDataset(test, x_transform)

  train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=2, batch_size=batch_size)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=2, batch_size=batch_size)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=batch_size)

  return train, val, x_test_sizes, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader