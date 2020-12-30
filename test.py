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
import torch.distributed.optim as dist_optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Data_Parallel import _DataParallel, _DistDataParallel

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

dataset_root = '/workspace/data/VOC2007'
checkpoint = 'workdirs/checkpoint.pth'
batch_size = 60
resume = True
save_folder = 'workdirs'
device='cuda:0'
dataset = VOC(dataset_root,use_set='test',transforms=Compose(
    [resize(),normalize(),DefualtFormat()]))

dataloader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=0,
                             collate_fn=voc_detector_co,
                             pin_memory=False, shuffle=True,
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
model = model.to(device)
model.eval()
if resume:
    load_model_optimizer_checkpoint(model,None,'workdirs/checkpoint384.pth')

# test
# filename = os.path.join(save_folder,'test1.txt')
# num_images = len(dataset)
# results = []
# img_infos_list = []
# for img, targets, img_infos in tqdm(dataloader):
#     with torch.no_grad():
#         img = img.to(device)
#
#         gt_bboxes = [gt[0].to(device) for gt in targets]
#         gt_labels = [gt[1].to(device) for gt in targets]
#         default_gt_bboxes = [img_info['default_gt_bboxes'] for img_info in img_infos]
#         default_gt_labels = [img_info['default_gt_labels'] for img_info in img_infos]
#         bbox_results = model.simple_test(img, img_infos, rescale=True)
#         results.append(bbox_results)
#         img_infos_list.append(img_infos)
# import pickle
# data = {'result':results,'img_info':img_infos_list}
# with open('data.pkl','wb') as p:
#     pickle.dump(data,p)
# print('dump done')
import pickle
with open('data.pkl','rb') as p:
    data = pickle.load(p)
results = []
for i in data['result']:
    for j in i:
        results.append(j)

print(dataset.evaluate_map(results, data['img_info']))

