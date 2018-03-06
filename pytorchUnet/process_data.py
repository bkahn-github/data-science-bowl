import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

import skimage
import skimage.io
import skimage.morphology
from skimage.transform import resize

def predict(model, test_dataset, test_dataloader, optimizer):
    preds = np.zeros((len(test_dataset), 128, 128))

    for i, x in tqdm(enumerate(test_dataloader)):
        x = Variable(x).cuda()
        optimizer.zero_grad()
        
        outputs = model(x)
        preds[i] = (outputs.cpu().data.numpy().reshape(128, 128))

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

def prob_to_rles(x, cutoff=0.5):
    lab_img = skimage.morphology.label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def upsample(preds, test_path=test_path, test_ids=test_ids):
    preds_test_upsampled = []
    for i, test_id in enumerate(test_ids):
        img = skimage.io.imread('{0}/{1}/images/{1}.png'.format(test_path, test_id))
        img_upscaled = skimage.transform.resize(preds[i], (img.shape[0], img.shape[1]), mode='constant', preserve_range=True)
        preds_test_upsampled.append(img_upscaled)

    return preds_test_upsampled

def encode(preds_test_upsampled, test_ids):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    return rles, new_test_ids