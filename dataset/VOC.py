import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import cv2
import torch.utils.data.dataset as dataset
from functools import partial
from .transform import *

class VOC(dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self,
                 dataset_path,
                 use_set = 'train',
                 transforms = [partial(PhotometricDistort(),delta=32,
                                       contrast_range=(0.5,1.5),
                                       saturation_range=(0.5,1.5),
                                       hue_delta=18),
                               partial(expand(),
                                       mean=[123.675, 116.28, 103.53],
                                       to_rgb=True,
                                       expand_ratio=(1,4)),
                               partial(resize(),resize_size=(300,300), keep_ratio=False),
                               partial(normalize(),mean=[127,127,127],
                                       std=[1,1,1],
                                       to_rgb=True),
                               partial(flip_random(),)
                               ],
                 to_rgb = True):
        self.datase_path =dataset_path
        self.use_set = use_set
        self.transforms = transforms  #transform list()
        self.to_rgb = to_rgb
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

        self.jpg_dir = osp.join(dataset_path, 'JPEGImages', '%s.jpg')
        self.anno_dir = osp.join(dataset_path, 'Annotations', '%s')
        self.ids = list()
        for line in open(osp.join(dataset_path,'ImageSets',
                                  'Main',use_set+'.txt')):
            self.ids.append(line.strip())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        :param idx: data idx
        :return: (img, target)
                target: (bboxes: np.float32
                         labels: np.int64)
        """
        img_id = self.ids[idx]
        xml_name = self.anno_dir % img_id
        jpg_name = self.jpg_dir % img_id

        img = cv2.imread(jpg_name)
        h, w, c = img.shape

        anno = self.get_anno(xml_name)

        if self.transforms is not None:
            for t in self.transforms:
                img, anno = t(img, anno)
        bboxes = anno[0]
        labels = anno[1]

        if self.to_rgb:
            img = img[:,:,(2,1,0)]

        return img, (bboxes, labels)

    def get_anno(self,xml_name):
        tree = ET.parse(xml_name)
        root = tree.getroot()
        bboxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')

            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]

            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0,4))
            labels = np.zeros((0,))
        return (bboxes, labels)