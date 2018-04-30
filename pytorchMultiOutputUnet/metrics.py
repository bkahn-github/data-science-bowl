import numpy as np

import torch
import torch.nn as nn

def loss(inputs, targets):
    bceloss = nn.BCELoss()
    loss = bceloss(inputs, targets) + dice_loss(inputs[:,0], targets[:,0]) * 1 + dice_loss(inputs[:,1], targets[:,1]) * 10
 
    return loss

def dice_loss(inputs, targets):
    num = targets.size(0)
    m1  = inputs.view(num,-1)
    m2  = targets.view(num,-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = 1 - score.sum()/num
    return score

def iou(predict, label):

    # Precision helper function
    def compute_precision(threshold, iou):
        matches = iou > threshold
        true_positives  = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    num_label   = len(np.unique(label  ))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(label.flatten(), predict.flatten(), bins=(num_label, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(label,   bins = num_label  )[0]
    area_pred = np.histogram(predict, bins = num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision
