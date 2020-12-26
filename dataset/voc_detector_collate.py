import torch
import cv2
import numpy as np

def voc_detector_co(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append((torch.FloatTensor(sample[1][0]),torch.FloatTensor(sample[1][1])))
    return torch.stack(imgs, 0), targets