import os
import numpy as np

import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=True):
        super(ConvBlock, self).__init__()
        self.pooling = pooling
      
        self.conv = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Dropout2d(0.2),
          nn.Conv2d(out_ch, out_ch, 3, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Dropout2d(0.2)
        )
        
    def forward(self, x):
        x = self.conv(x)      
  
        if self.pooling == True:
            x = F.max_pool2d(x, 2)
                            
        return x

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, pooling=False)

    def forward(self, x1, x2):
        upsample = self.upsample(x1)
        
        cat = torch.cat([x2, upsample], dim=1)
        conv = self.conv(cat)

        return conv

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)

        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.in_conv = ConvBlock(3, 64, pooling=False)
        self.down_1 = ConvBlock(64, 128)
        self.down_2 = ConvBlock(128, 256)
        self.down_3 = ConvBlock(256, 512)
        self.down_4 = ConvBlock(512, 1024)
        self.up_1 = Upsample(1024, 512)
        self.up_2 = Upsample(512, 256)
        self.up_3 = Upsample(256, 128)
        self.up_4 = Upsample(128, 64)
        
        self.out_conv = OutConv(64, 1)

    def forward(self, x):
        x = x / 255
    
        x1 = self.in_conv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        
        outputs = self.out_conv(x)

        return outputs