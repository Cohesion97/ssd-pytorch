import os
import cv2
import torch
import time
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
       print('load model checkpoint from {}'.format(checkpoint))
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

def warmup_learning_rate(optimizer, step_now, warmup_step, lr):
    lr = lr / warmup_step * (step_now+1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def time_to_hour(sec):
    hour = sec // 3600
    min = (sec % 3600)//60
    sec = (sec%60)
    return ' etc: {}h{}m{}s'.format(int(hour),int(min),int(sec))

def get_current_time():
    time_stamp = time.time()  # 当前时间的时间戳
    local_time = time.localtime(time_stamp)  #
    str_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return str_time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--lr', help='learning rate', type=float, default=2e-4)
    parser.add_argument('--local_rank', help='learning rate', type=int, default=-1)
    args = parser.parse_args()
    return args


arg = parse_args()

dist.init_process_group(backend='nccl')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


dataset_root = '/workspace/data_dir/VOC2007'
checkpoint = 'workdirs/trainaug_checkpoint.pth'
checkpoint_from = '/tmp/pycharm_project_217/workdirs/trainaug_checkpoint240.pth'
log_filename = 'workdirs/log.txt'
batch_size = 32
total_epoch = 240
resume = 0
warmup_step = 200
dataset = VOC(dataset_root,use_set='train',
              transforms=Compose([PhotometricDistort(delta=32,
                                                     contrast_range=(0.5,1.5),
                                                     saturation_range=(0.5,1.5),
                                                     hue_delta=18),
                               #  expand(mean=(123.675, 116.28, 103.53),
                               #         to_rgb=True,
                               #         expand_ratio=(1,4)),
                               # MinIoURandomCrop(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                               #                  min_crop_size=0.3,
                               #                  bbox_clip_border=True),
                               resize(resize_size=(300,300),
                                      keep_ratio=False),
                               normalize(mean=[123.675, 116.28, 103.53],
                                         std=[1,1,1],
                                         to_rgb=True),
                               flip_random(flip_ratio=0.5, direction='horizontal'),
                               DefualtFormat(),]))
train_sample = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=0,
                             collate_fn=voc_detector_co,
                             pin_memory=False, sampler=train_sample
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
if resume:
    load_model_optimizer_checkpoint(model, None, checkpoint=checkpoint_from)
model = model.to(local_rank)
model = _DistDataParallel(model, device_ids=[local_rank], output_device=local_rank)
optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9,
                          weight_decay=5e-4)
model.train()

iter = range(total_epoch)
step = 0
total_loss = []
cla_loss = []
loc_loss = []
start_time = time.time()
with open(log_filename, 'wt') as f:
    for epoch in iter:
        dataloader.sampler.set_epoch(epoch)
        if dist.get_rank() == 0 and epoch % 80==0 and epoch != 0:
            save_checkpoint(model, optimizer, epoch, checkpoint)

        if epoch in [160, 200,]:
            step += 1
            adjust_learning_rate(optimizer,0.1,step,lr=arg.lr)
        for i, (images, targets, img_infos) in enumerate(dataloader):
            step_now = epoch * len(dataloader) + i
            if total_epoch < warmup_step:
                warmup_learning_rate(optimizer, step_now, warmup_step, lr=arg.lr)
            images = images.to(local_rank)

            gt_bboxes = [gt[0].to(local_rank) for gt in targets]
            gt_labels = [gt[1].to(local_rank) for gt in targets]

            optimizer.zero_grad()

            loss = 0
            loss_cla = 0
            loss_loc = 0
            loss_c, loss_l = model.forward_train(images, gt_bboxes, gt_labels)
            for k in loss_c:
                loss_cla += k
            for j in loss_l:
                loss_loc += j
            loss = loss_cla + loss_loc
            loss.backward()
            optimizer.step()
            now_lr = optimizer.param_groups[0]['lr']

            print_loss = get_current_time() + \
                         "[epoch:%d][%d/%d]: loss_c: %0.3f los_l: %0.3f loss: %0.3f lr: %0.3e" % \
                         (epoch,i,len(dataloader),loss_cla ,loss_loc, loss, now_lr)
            if epoch==0 and i==0:
                pass
            else:
                if dist.get_rank() == 0:
                    cla_loss.append(loss_cla)
                    loc_loss.append(loc_loss)
                    total_loss.append(loss)

                    if i % 10==0 :
                        during_step = (time.time() - start_time) / 10
                        rest_step = (total_epoch-epoch)*len(dataloader)-i
                        rest_time = during_step * rest_step
                        print_loss = print_loss + time_to_hour(rest_time)
                        print(print_loss)
                        f.write(print_loss)
                        start_time = time.time()
import pickle
with open('loss_ssd.pkl','wb') as file:
    pickle.dump(total_loss, file)
if dist.get_rank() == 0:
    save_checkpoint(model, optimizer, total_epoch, checkpoint)



