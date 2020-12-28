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
from dataset.VOC import VOC
import torch.optim as optim
import torch.distributed.optim as dist_optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Data_Parallel import _DataParallel, _DistDataParallel


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')



dist.init_process_group(backend='gloo')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dataset_root = '/workspace/data/VOC2007'
batch_size = 40
dataset = VOC(dataset_root)
train_sample = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=0,
                             collate_fn=voc_detector_co,
                             pin_memory=False, sampler=train_sample
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth').to(local_rank)
model = _DistDataParallel(model, device_ids=[local_rank], output_device=local_rank)
# model = model.cuda()
# model = _DataParallel(model,device_ids=[0,1,2,3],output_device=0)
print('DDP done')
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=5e-4)
print('start training')
model.train()

iter = tqdm(range(200))
# batch_iterator = iter(dataloader)
for epoch in iter:
    dataloader.sampler.set_epoch(epoch)
    for images, targets in dataloader:
        images = images.to(local_rank)
        #print(targets)
        #from IPython import embed; embed()
        gt_bboxes = [gt[0].to(local_rank) for gt in targets]
        gt_labels = [gt[1].to(local_rank) for gt in targets]

        optimizer.zero_grad()
        #cla, loc = model(images)
        #print(cla)
        loss = 0
        loss_c, loss_l = model.forward_train(images, gt_bboxes, gt_labels)
        for k in loss_c:
            loss += k
        for j in loss_l:
            loss += j
        loss.backward()
        optimizer.step()
        #print('loss:{}'.format(loss))
        iter.desc = "loss = %0.3f" % loss




