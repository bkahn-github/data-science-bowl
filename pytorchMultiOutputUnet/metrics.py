import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Taken from Heng Cher Keng's April 27 code
class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.contiguous().view(-1)
        t = labels.contiguous().view(-1)

        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/(w.sum()+ 1e-12)
        return loss

def make_weight(labels_truth):
    B,C,H,W = labels_truth.size()
    weight = torch.FloatTensor(B*C*H*W).requires_grad_().to(device)

    pos = labels_truth.detach().sum()
    neg = B*C*H*W - pos
    
    if pos>0:
        pos_weight = 0.5/pos
        neg_weight = 0.5/neg
    else:
        pos_weight = 0
        neg_weight = 0

    weight[labels_truth.contiguous().view(-1)> 0.5] = pos_weight
    weight[labels_truth.contiguous().view(-1)<=0.5] = neg_weight

    weight = weight.view(B,C,H,W)
    return weight

def loss(inputs, targets):
    mask_weights = make_weight(targets[:,0:1])
    edges_weights = make_weight(targets[:,1:2])
    backgrounds_weights = make_weight(targets[:,2:3])

    mask_loss = WeightedBCELoss2d()(inputs[:,0:1], targets[:,0:1], mask_weights)
    edges_loss = WeightedBCELoss2d()(inputs[:,1:2], targets[:,1:2], edges_weights)
    backgrounds_loss = WeightedBCELoss2d()(inputs[:,2:3], targets[:,2:3], backgrounds_weights)

    loss = mask_loss + edges_loss _ backgrounds_loss

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
