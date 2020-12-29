import torch
import cv2
import numpy as np

def voc_detector_co(batch):
    targets = []
    imgs = []
    img_infos = []
    #from IPython import embed;embed()
    for sample in batch:
        imgs.append(sample[0])
        #from IPython import embed;embed()
        targets.append((sample[1][0].to(torch.float32),sample[1][1]))
        img_infos.append(sample[2])
    return torch.stack(imgs, 0), targets, img_infos