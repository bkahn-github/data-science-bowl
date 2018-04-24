from tqdm import tqdm

from metrics import dice_loss

import torch
from torch.autograd import Variable

def model_loop(dataloader, model, optimizer, is_training=True):
    for data in tqdm(dataloader):
        img, target = data['img'], data['target']

        if torch.cuda.is_available():         
            x = Variable(img).cuda()
            y = Variable(target).cuda()
        else:
            x = Variable(img)
            y = Variable(target)                

        optimizer.zero_grad()

        outs = model(x)
        loss = dice_loss(outs, y)

        if is_training:
            loss.backward()
            optimizer.step()

    return model, loss