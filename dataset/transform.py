import torch
import numpy as np
import cv2
import random

def resize(img, target, resize_size=(300,300), keep_ratio=False):
    if not keep_ratio:
        h, w = img.shape[:2]
        img = cv2.resize(img, resize_size, interpolation='bilinear')
        w_scale = resize_size[0]/w
        h_scale = resize_size[1]/h

        bboxes = target[0]
        bboxes[:, 0::2] = np.clip(bboxes[:,0::2], 0, resize_size[0]) # width
        bboxes[:, 1::2] = np.clip(bboxes[:,1::2], 0, resize_size[1]) # height
    return img, (bboxes, target[1])

def PhotometricDistort(img, target,
                       delta=32,
                       contrast_range=(0.5,1.5),
                       saturation_range=(0.5,1.5),
                       hue_delta=18):
    assert delta>=0.
    assert delta<=255.0
    contrast_first = random.uniform(2)


    #brightness
    if random.randint(2):
        d = random.uniform(-delta, delta)
        img += d

    if contrast_first:
        if random.randint(2):
            alpha = random.uniform(contrast_range[0],
                                   contrast_range[1])
            img *= alpha

    #saturation
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    if random.randint(2):
        a = random.uniform(saturation_range[0],
                           saturation_range[1])
        img *= a
    #hue
    if random.randint(2):
        img[..., 0] += random.uniform(-hue_delta, hue_delta)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #contrast
    if not contrast_first:
        if random.randint(2):
            alpha = random.uniform(contrast_range[0],
                                   contrast_range[1])
            img *=alpha

    return img, target

def expand(img, target,
           mean = (0,0,0),
           to_rgb=True):
