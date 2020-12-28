import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
from dataset.voc_detector_collate import voc_detector_co
from ssd_det import SSD_DET
from dataset.VOC import VOC
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Data_Parallel import _DataParallel

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset_root = '/workspace/data/VOC2007'
batch_size = 40
dataset = VOC(dataset_root)
dataloader = data.DataLoader(dataset, batch_size,
                             num_workers=0, shuffle=True,
                             collate_fn=voc_detector_co,
                             pin_memory=False,
                             )
model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')
model = model.cuda()
model = _DataParallel(model,device_ids=[0,],output_device=0)
cudnn.benchmark = True

model.train()

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=5e-4)

batch_iterator = iter(dataloader)
for i in range(100000):
    images, targets = next(batch_iterator)
    from IPython import embed

    gt_bboxes = [Variable(gt[0].cuda()) for gt in targets]
    gt_labels = [Variable(gt[1].cuda()) for gt in targets]

    images = Variable(images.cuda())

    cla, loc = model(images)
    # optimizer.zero_grad()
    # loss = 0
    # loss_c, loss_l = model.forward_train(images,gt_bboxes,gt_labels,batch_size)
    # for k in loss_c:
    #     loss += k
    # for j in loss_l:
    #     loss += j
    # loss.backward()
    # optimizer.step()
    # if i % 20==0:
    #     print('iter:{},loss:{}'.format(i,loss))



