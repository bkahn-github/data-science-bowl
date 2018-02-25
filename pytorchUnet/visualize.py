import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

from metrics import iou

def show_val(ix, model):
  input = Variable(val_dataset[ix][0]).cuda()
  model = model.eval()
  out = model(input.unsqueeze(0))

  plt.figure(figsize=(20, 10))
  plt.subplot(1, 3, 1)
  plt.imshow(val_dataset[ix][0][0].numpy().reshape(128, 128)*0.5+0.5)

  plt.subplot(1, 3, 2)
  plt.imshow(val_dataset[ix][1][0].numpy().reshape(128, 128)*0.5+0.5)

  plt.subplot(1, 3, 3)
  plt.imshow((out.cpu().data.numpy() > 0.5).reshape(128, 128)*0.5+0.5)
    
  iou, _ = iou((out.cpu().data.squeeze(0).numpy() > 0.5), val_dataset[ix][1][0].numpy())

  print(iou)

def show_test(ix, model):
  input = Variable(test_dataset[ix]).cuda()
  model = model.eval()
  out = model(input.unsqueeze(0))

  plt.figure(figsize=(15, 10))
  plt.subplot(1, 2, 1)
  plt.imshow(test_dataset[ix][0].numpy().reshape(128, 128)*0.5+0.5)

  plt.subplot(1, 2, 2)
  plt.imshow((out.cpu().data.numpy() > 0.5).reshape(128, 128)*0.5+0.5)