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
        
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_ch)
        self.dropout_1 = nn.Dropout2d(0.2)
        
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_ch)
        self.dropout_2 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.out_conv(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)

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

        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)

        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.in_conv = ConvBlock(3, 16)   
        self.down_1 = ConvBlock(16, 32)
        self.down_2 = ConvBlock(32, 64)     
        self.down_3 = ConvBlock(64, 128)
        self.down_4 = ConvBlock(128, 256)

        self.down_5 = ConvBlock(256, 256, pooling=False)

        self.up_1 = Upsample(256, 128)
        self.up_2 = Upsample(128, 64)
        self.up_3 = Upsample(64, 32)
        self.up_4 = Upsample(32, 16)

        # self.out_convs = nn.ModuleList([OutConv(16, 1), OutConv(16, 1), OutConv(16, 1)])
        self.out_conv_1 = OutConv(16, 1)
        self.out_conv_2 = OutConv(16, 1)
        self.out_conv_3 = OutConv(16, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        x6 = self.down_5(x5)
        
        x7 = self.up_1(x6, x4)
        x8 = self.up_2(x7, x3)
        x9 = self.up_3(x8, x2)
        x10 = self.up_4(x9, x1)

        # outputs = [output(x10) for output in self.out_convs]
        outputs = [self.out_conv_1(x10), self.out_conv_2(x10), self.out_conv_3(x10)]

        return outputs