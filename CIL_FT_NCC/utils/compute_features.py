#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()

    features = np.zeros([num_samples, num_features])

    start_idx = 0
    counter = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            feats = tg_feature_model(inputs) #,

            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(feats.cpu())
            start_idx = start_idx + inputs.shape[0]
            counter += 1
            #print("Iter:",counter)

    assert(start_idx==num_samples)


    return features


def compute_predictions(tg_model, evalloader, num_samples, args=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    x_outs = None
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            y = torch.zeros(targets.size(0), args.num_output_capsules).scatter_(1, targets.view(-1, 1), 1.)
            x, y = Variable(inputs.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
            outputs, _ = tg_model(x, y)
            outputs = F.softmax(outputs, dim=1)
            preds = outputs.cpu().numpy()

            #######
            if start_idx == 0:
                x_outs = preds
            else:
                x_outs = np.concatenate((x_outs, preds), axis=0)
            #####

            start_idx = start_idx + inputs.shape[0]

    assert(start_idx==num_samples)


    return x_outs