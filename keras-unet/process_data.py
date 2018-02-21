import numpy as np

from skimage.transform import resize
from skimage.morphology import label

def upsample(x, sizes):
    upsampled = []
    for i in range(len(x)):
        upsampled.append(resize(np.squeeze(x[i]), sizes[i], mode='constant', preserve_range=True))
    
    return upsampled

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
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def encode(preds, ids):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = list(prob_to_rles(preds[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    return rles, new_test_ids