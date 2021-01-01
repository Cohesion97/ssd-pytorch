import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as data
from dataset.voc_detector_collate import voc_detector_co
from ssd_det import SSD_DET
from dataset.VOC import VOC, Compose
from dataset.transform import *

import torch.optim as optim
from load_pretrained import weights_to_cpu, load_checkpoint
from IPython import embed

dataset_root = '/workspace/data/VOC2007'
batch_size = 4
dataset = VOC(dataset_root,use_set='debug',)

dataloader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=0,
                             collate_fn=voc_detector_co,
                             pin_memory=False, shuffle=False,
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
local_rank='cuda'
model = model.to('cuda')
total_epoch=1
iter = range(total_epoch)
for i, (images, targets, img_infos) in enumerate(dataloader):
    images = images.to(local_rank)

    gt_bboxes = [gt[0].to(local_rank) for gt in targets]
    gt_labels = [gt[1].to(local_rank) for gt in targets]
    (labels_list, bboxes_target_list, num_pos_total, num_neg_total) = \
        model.vis_anchor_match(images,gt_bboxes,gt_labels)
    embed()
    break

