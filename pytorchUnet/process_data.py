import numpy as np
from tqdm import tqdm

import torch
import torch.autograd.Variable as Variable

def predict(model, test_dataset, test_dataloader, optimizer):
    preds = np.zeros((len(test_dataset), 128, 128))

    for i, x in tqdm(enumerate(test_dataloader)):
        x = Variable(x).cuda()
        optimizer.zero_grad()
        
        outputs = model(x)
        preds[i] = (outputs.cpu().data.numpy().reshape(128, 128))

    return preds

def threshold_preds(preds, threshold=0.5):
    preds = (preds > threshold).astype(np.uint8)
    
    return preds

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def upsample(preds, x_test_sizes):
    preds_test_upsampled = []
    for i in range(len(preds)):
        preds_test_upsampled.append(resize(np.squeeze(preds[i]), x_test_sizes[i], mode='constant', preserve_range=True).astype(np.uint8))

    return preds_test_upsampled

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def encode(preds_test_upsampled, test_ids):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    return rles, new_test_ids