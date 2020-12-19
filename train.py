import os
import cv2
import torch
import numpy as np
from ssd_det import SSD_DET

img_foloder = './img/'
a = 0
for i in os.listdir(img_foloder):
    img = cv2.imread(img_foloder+i)
    img = cv2.resize(img,(300,300))
    if isinstance(a,int):
        a = img
    else:
        from IPython import embed;embed()
        a = np.vstack((a,img))


model = SSD_DET(pretrained='vgg16_caffe-292e1171.pth')


