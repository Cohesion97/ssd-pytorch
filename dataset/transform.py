import numpy as np
import cv2
from numpy import random
from utils.IoU_calculate.np_bbox_overlaps import bbox_overlaps

class resize(object):
    def __init__(self, resize_size=(300,300), keep_ratio=False):
        self.resize_size = resize_size
        self.keep_ratio = keep_ratio

    def __call__(self, img, target, img_info):
        keep_ratio = self.keep_ratio
        resize_size = self.resize_size
        if not keep_ratio:
            h, w = img.shape[:2]
            img = cv2.resize(img, resize_size,)
            w_scale = resize_size[0]/w
            h_scale = resize_size[1]/h
            bboxes = target[0]

            scale_factor = np.array([w_scale,h_scale,w_scale,h_scale],dtype=np.float32)
            bboxes = bboxes * scale_factor
            bboxes[:, 0::2] = np.clip(bboxes[:,0::2], 0, resize_size[0]) # width
            bboxes[:, 1::2] = np.clip(bboxes[:,1::2], 0, resize_size[1]) # height

            img_info['scale_factor'] = scale_factor
        return img, (bboxes, target[1]), img_info

class PhotometricDistort(object):
    def __init__(self,delta=32,
                       contrast_range=(0.5,1.5),
                       saturation_range=(0.5,1.5),
                       hue_delta=18):
        self.delta = delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, target, img_info):

        assert self.delta>=0.
        assert self.delta<=255.0
        contrast_first = random.randint(2)

        img = img.copy()
        #brightness
        if random.randint(2):
            d = random.uniform(-self.delta, self.delta)
            img += d

        if contrast_first:
            if random.randint(2):
                alpha = random.uniform(self.contrast_range[0],
                                       self.contrast_range[1])
                img *= alpha

        #saturation
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        if random.randint(2):
            a = random.uniform(self.saturation_range[0],
                               self.saturation_range[1])
            img *= a
        #hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        #contrast
        if not contrast_first:
            if random.randint(2):
                alpha = random.uniform(self.contrast_range[0],
                                       self.contrast_range[1])
                img *=alpha

        return img, target,img_info

class expand(object):
    def __init__(self,mean=(123.675, 116.28, 103.53),
           to_rgb=True,
           expand_ratio=(1,4)):
        self.to_rgb = to_rgb
        self.expand_ratio = expand_ratio
        self.mean = mean

    def __call__(self, img, target, img_info):
        if random.randint(2):
            return img, target, img_info
        if self.to_rgb:
            mean = self.mean[::-1]
        h,w,c = img.shape
        ratio = random.uniform(*self.expand_ratio)
        #from IPython import embed;embed()
        ex_img = np.full((int(h*ratio), int(w*ratio),c),mean,dtype=img.dtype)
        left = int(random.uniform(0,w*(ratio-1)))
        top = int(random.uniform(0,h*(ratio-1)))
        ex_img[top:h+top, left:left+w] = img

        bboxes = target[0]
        bboxes[:,0::2] += left
        bboxes[:,1::2] += top

        return ex_img, (bboxes, target[1]), img_info

class normalize(object):
    def __init__(self,mean=[123.675, 116.28, 103.53],
              #std=[58.395, 57.12, 57.375],
              std=[1,1,1],
              to_rgb=True):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def __call__(self, img, target=None, img_info=None):
        img = img.copy().astype(np.float32)
        mean = np.float64(np.array(self.mean,dtype=np.float32).reshape(1,-1))
        std = 1 / np.float64(np.array(self.std, dtype=np.float32).reshape(1, -1))
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, std, img)
        return img, target, img_info

class flip_random(object):
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, img, target=None, img_info=None):
        direction = self.direction
        flip_ratio = self.flip_ratio
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if random.uniform(0, 1) > flip_ratio:
            if direction=='horizontal':
                img = np.flip(img, axis=1)
            elif direction=='vertical':
                img = np.flip(img, axis=0)
            elif direction=='diagonal':
                img = np.flip(img, axis=(0,1))

            bboxes = target[0]
            flipped = bboxes.copy()
            img_shape = img.shape
            if direction == 'horizontal':
                w = img_shape[1]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif direction == 'vertical':
                h = img_shape[0]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif direction == 'diagonal':
                w = img_shape[1]
                h = img_shape[0]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            return img, (flipped, target[1]), img_info
        else:
            return img, target, img_info

class MinIoURandomCrop(object):
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        self.min_ious = min_ious
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, img, target=None, img_info=None):
        min_ious = self.min_ious
        min_crop_size = self.min_crop_size
        bbox_clip_border = self.bbox_clip_border
        iou_select = (1,0, *min_ious)
        bboxes = target[0]
        labels = target[1]


        h, w, c = img.shape
        while True:
            mode = random.choice(iou_select)
            if mode == 1:
                return img, target, img_info

            min_iou = mode
            for i in range(50):
                n_w = random.uniform(min_crop_size * w, w)
                n_h = random.uniform(min_crop_size * h, h)

                if n_w / n_h < 0.5 or n_w / n_h > 2:
                    continue

                left = random.uniform(w - n_w)
                top = random.uniform(h - n_h)

                patch = np.array([int(left), int(top), int(left+n_w), int(top+n_h)])

                if patch[0]==patch[2] or patch[1]==patch[3]:
                    continue
                overlaps = bbox_overlaps(patch, bboxes).reshape(-1)
                if len(overlaps)>0 and overlaps.min() < min_iou:
                    continue

                if len(overlaps)>0:
                    center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) *
                            (center[:, 1] > patch[1]) *
                            (center[:, 0] < patch[2]) *
                            (center[:, 1] < patch[3]))

                    if not mask.any():
                        continue
                    bboxes_copy = bboxes.copy()
                    bboxes_copy = bboxes_copy[mask,:]
                    labels_copy = labels.copy()[mask]
                    if bbox_clip_border:
                        bboxes_copy[:, 2:] = bboxes_copy[:, 2:].clip(max=patch[2:])
                        bboxes_copy[:, :2] = bboxes_copy[:, :2].clip(min=patch[:2])
                    bboxes_copy -= np.tile(patch[:2], 2)
                    img = img[patch[1]:patch[3],patch[0]:patch[2]]

                    return img, (bboxes_copy, labels_copy), img_info

class DefualtFormat(object):
    def __call__(self, img, target, img_info):
        # h,w,c->c,h,w

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return img, target, img_info