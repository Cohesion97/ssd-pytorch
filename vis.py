import cv2
import pickle
import os.path as osp
import numpy as np
with open('data.pkl','rb') as p:
    data = pickle.load(p)
results = []
for i in data['result']:
    for j in i:
        results.append(j)
infos = []
for i in data['img_info']:
    for j in i:
        infos.append(j)
img_id = infos[0]['id']
dataset_path = '/workspace/data/VOC2007'
jpg_dir = osp.join(dataset_path, 'JPEGImages', '%s.jpg')
anno_dir = osp.join(dataset_path, 'Annotations', '%s.xml')
xml_name = anno_dir % img_id
jpg_name = jpg_dir % img_id
img = cv2.imread(jpg_name)
color = (0,255,0)
for classes in results[0]:
    if len(classes)==0:
        continue
    for bbox in classes:
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        if x1>=x2 or y1>=y2:
            continue
        img=cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness=1,)
    if color==(0,255,0):
        color=(0,255,255)
cv2.imwrite('test{}'.format(img_id)+'.jpg',img)