import numpy as np
import torch
from torch.autograd import Variable

from tqdm import tqdm

from load_data import TrainDataset, TestDataset, load_data
from model import UNet
from metrics import dice_loss, iou

def train(model, data_loader, loss, metric  epochs=10):
    model = model
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    _, _, _, _, _, _, train_dataloader, val_dataloader, _ = data_loader()

    for epoch in range(epochs):
        train_ious = []
        val_ious = []
        
        for x_train, y_train in tqdm(train_dataloader):
            x_train = Variable(x_train).cuda()
            y_train = Variable(y_train).cuda()

            optimizer.zero_grad()
            
            outputs = model(x_train)

            loss = loss(outputs, y_train)
            train_iou, _ = metric((outputs.cpu().data.squeeze(0).numpy() > 0.5), y_train.cpu().data.numpy())
            train_ious.append(train_iou)
            
            loss.backward()
            optimizer.step()

        for x_val, y_val in tqdm(val_dataloader):
            x_val = Variable(x_val).cuda()
            y_val = Variable(y_val).cuda()

            optimizer.zero_grad()
            
            o = model(x_val)

            val_loss = loss(o, y_val)
            val_iou, _ = metric((o.cpu().data.squeeze(0).numpy() > 0.5), y_val.cpu().data.numpy())
            val_ious.append(val_iou)

    print('\n')
    print(f'Epoch: {epoch} Training Loss: {round(loss.data[0], 4)} Training IOU: {np.asarray(train_ious).mean()} Val Loss: {round(val_loss.data[0], 4)} Val IOU: {np.asarray(val_ious).mean()}', end='\n')

    return model