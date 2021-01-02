import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from .transform import *
import cv2
C = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
vis_img='/tmp/pycharm_project_217/vis/'
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None, img_info=None):
        for ind,t in enumerate(self.transforms):
            #transform vis
            # bbox=target[0]
            # label=target[1]
            # write_img = img.copy()
            # color = (0, 255, 255)
            # for i, item in enumerate(bbox):
            #     x1,y1,x2,y2=int(bbox[i][0]),int(bbox[i][1]),int(bbox[i][2]),int(bbox[i][3])
            #     write_img = cv2.rectangle(write_img, (x1,y1),(x2,y2),color,thickness=1,)
            #     write_img = cv2.putText(write_img,C[label[i]],(x1,y1),color=color,thickness=1,fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
            # cv2.imwrite(vis_img+img_info['id']+str(ind)+'.jpg',write_img)
            # print('save img'+vis_img+str(ind)+'.jpg')
            img, target, img_info = t(img, target, img_info)

        return img, target, img_info

class VOC(Dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self,
                 dataset_path,
                 use_set = 'trainval',
                 transforms = Compose([PhotometricDistort(),
                               expand(),
                               MinIoURandomCrop(),
                               resize(),
                               normalize(),
                               flip_random(),
                               DefualtFormat(),
                               ]),
                 to_rgb = True):
        self.datase_path =dataset_path
        self.use_set = use_set
        self.transforms = transforms  #transform list()
        self.to_rgb = to_rgb
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

        self.jpg_dir = osp.join(dataset_path, 'JPEGImages', '%s.jpg')
        self.anno_dir = osp.join(dataset_path, 'Annotations', '%s.xml')
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

        img_info = {}
        img_info['id'] = img_id
        img_info['scale_factor']=np.array([1,1,1,1],dtype=np.float32)

        img = cv2.imread(jpg_name)
        img = img.astype(np.float32)
        h, w, c = img.shape
        img_info['ori_hw'] = (h, w)

        anno = self.get_anno(xml_name)
        img_info['default_gt_bboxes'] = anno[0]
        img_info['default_gt_labels'] = anno[1]

        if self.transforms is not None:
            img, anno, img_info = self.transforms(img, anno, img_info)
        bboxes = anno[0]
        labels = anno[1]

        return torch.from_numpy(img), (torch.from_numpy(bboxes), torch.from_numpy(labels)), img_info

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
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        return (np.array(bboxes), np.array(labels))
