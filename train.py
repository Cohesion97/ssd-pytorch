import os
import cv2
import torch
import numpy as np
from ssd_det import SSD_DET

model = SSD_DET(20, pretrained='vgg16_caffe-292e1171.pth')

from IPython import embed;embed()