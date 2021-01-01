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
        print('load optimizer checkpoint from {}'.format(checkpoint))
        optimizer.load_state_dict(checkpoint['optimizer'])


dataset_root = '/workspace/data/VOC2007'
checkpoint = 'workdirs/checkpoint240.pth'
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
    load_model_optimizer_checkpoint(model,None,checkpoint)

# test
filename = os.path.join(save_folder,'test1.txt')
num_images = len(dataset)
results = []
img_infos_list = []
for img, targets, img_infos in tqdm(dataloader):
    with torch.no_grad():
        img = img.to(device)
        gt_bboxes = [gt[0].to(device) for gt in targets]
        gt_labels = [gt[1].to(device) for gt in targets]
        default_gt_bboxes = [img_info['default_gt_bboxes'] for img_info in img_infos]
        default_gt_labels = [img_info['default_gt_labels'] for img_info in img_infos]
        bbox_results = model.simple_test(img, img_infos, rescale=True)
        results.append(bbox_results)
        img_infos_list.append(img_infos)
import pickle
data = {'result':results,'img_info':img_infos_list}
with open('data_240.pkl','wb') as p:
    pickle.dump(data,p)
print('dump done')

import pickle
with open('data_240.pkl','rb') as p:
    data = pickle.load(p)
results = []
for i in data['result']:
    for j in i:
        results.append(j)

def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix

all_boxes=transpose(results)
img_list = []
for i in data['img_info']:
    for k in i:
        img_list.append(k['id'])

C = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

jpeg = '/workspace/data/VOC2007/JPEGImages/'
for i in range(5):
    img_path = jpeg + img_list[i] + '.jpg'
    img = cv2.imread(img_path)
    for j in range(20):
        det = all_boxes[j][i]
        for k in range(len(det)):
            if det[k][-1]>0.5:
                x1, y1, x2, y2 = int(det[k][0]), int(det[k][1]), int(det[k][2]), int(det[k][3])
                img = cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 255),1)
                img = cv2.putText(img, C[j]+'{:.3f}'.format(float(det[k][-1])),(x1,y1),color=(0, 255, 255),
                                  thickness=1,fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    cv2.imwrite('result_vis/vis'+img_list[i]+'.jpg',img)

