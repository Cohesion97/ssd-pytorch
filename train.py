import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
from dataset.voc_detector_collate import voc_detector_co
from ssd_det import SSD_DET
from dataset.VOC import VOC
dataset_root = '/workspace/data/VOC2007'
batch_size = 10
dataset = VOC(dataset_root)
dataloader = data.DataLoader(dataset, batch_size,
                             num_workers=0, shuffle=True,
                             collate_fn=voc_detector_co,
                             pin_memory=True)
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
batch_iterator = iter(dataloader)
for i in range(10):
    images, targets = next(batch_iterator)

from IPython import embed;embed()