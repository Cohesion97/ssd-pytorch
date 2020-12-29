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
from load_pretrained import weights_to_cpu, load_checkpoint
import torch.distributed.optim as dist_optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Data_Parallel import _DataParallel, _DistDataParallel

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {'state_dict':weights_to_cpu(model.state_dict()),
                  'optimizer':optimizer.state_dict()}
    filename = filename.split('.')[0] + str(epoch) + '.pth'
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()

    print('save model{} checkpoint to {}'.format(epoch,filename))

def load_model_optimizer_checkpoint(model, optimizer, checkpoint, map_location='default', strict=False):
    if model is not None:
       if map_location == 'default':
           if torch.cuda.is_available():
               device_id = torch.cuda.current_device()
               checkpoint = load_checkpoint(model,
                   checkpoint,
                   map_location=lambda storage, loc: storage.cuda(device_id))
           else:
               checkpoint = load_checkpoint(model, checkpoint)
       else:
           checkpoint = load_checkpoint(model, checkpoint, map_location=map_location)
    if 'optimizer' in checkpoint and optimizer is not None:
       optimizer.load_state_dict(checkpoint['optimizer'])

def adjust_learning_rate(optimizer, gamma, step, lr):

    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

dist.init_process_group(backend='gloo')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dataset_root = '/workspace/data/VOC2007'
checkpoint = 'workdirs/checkpoint.pth'
batch_size = 40
total_epoch = 384
resume = 0
dataset = VOC(dataset_root)
train_sample = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=0,
                             collate_fn=voc_detector_co,
                             pin_memory=False, sampler=train_sample
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
if resume:
    load_model_optimizer_checkpoint(model, None, checkpoint=checkpoint)
model = model.to(local_rank)
model = _DistDataParallel(model, device_ids=[local_rank], output_device=local_rank)
# model = model.cuda()
# model = _DataParallel(model,device_ids=[0,1,2,3],output_device=0)
print('DDP done')
optimizer = optim.SGD(model.parameters(), lr=4e-3, momentum=0.9,
                          weight_decay=5e-4)
if resume:
    load_model_optimizer_checkpoint(None, optimizer, checkpoint=checkpoint)


print('start training')
model.train()

iter = tqdm(range(total_epoch))
# batch_iterator = iter(dataloader)
step = 0

for epoch in iter:
    dataloader.sampler.set_epoch(epoch)
    if dist.get_rank() == 0 and epoch % 127==0 and epoch != 0:
        save_checkpoint(model, optimizer, epoch, checkpoint)

    if epoch in [256, 320,]:
        step += 1
        adjust_learning_rate(optimizer,0.1,step,lr=4e-3)
    for images, targets, img_infos in dataloader:
        images = images.to(local_rank)

        gt_bboxes = [gt[0].to(local_rank) for gt in targets]
        gt_labels = [gt[1].to(local_rank) for gt in targets]

        optimizer.zero_grad()

        loss = 0
        loss_c, loss_l = model.forward_train(images, gt_bboxes, gt_labels)
        for k in loss_c:
            loss += k
        for j in loss_l:
            loss += j
        loss.backward()
        optimizer.step()
        iter.desc = "loss = %0.3f" % loss
if dist.get_rank() == 0:
    save_checkpoint(model, optimizer, total_epoch, checkpoint)



