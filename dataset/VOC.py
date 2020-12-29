import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from functools import partial
from .transform import *
import logging
from terminaltables import AsciiTable
from multiprocessing import Pool

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None, img_info=None):
        for t in self.transforms:
            # print(t)
            img, target, img_info = t(img, target, img_info)
            # print(img)
            # print(target)
        #from IPython import embed;embed()
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
        #from IPython import embed;embed()
        xml_name = self.anno_dir % img_id
        jpg_name = self.jpg_dir % img_id

        img_info = {}
        img_info['id'] = img_id
        img_info['scale_factor']=np.array([1,1,1,1],dtype=np.float32)

        img = cv2.imread(jpg_name)
        img = img.astype(np.float32)
        h, w, c = img.shape

        anno = self.get_anno(xml_name)

        img_info['default_gt_bboxes'] = anno[0]
        img_info['default_gt_labels'] = anno[1]

        # print('anno:', anno)
        # print('img_id:',img_id)
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

        return (np.array(bboxes), np.array(labels))

    def evaluate_map(self, results, img_info_list ,iou_thr=0.45,nproc=4):
        """
        evaluate map
        :param results: (list[list]): [[cls1_det, cls2_det, ...], ...].
        :return:
        """
        num_imgs = len(results)
        num_scales = 1
        num_classes = len(results[0])  # positive class num
        area_ranges = None
        scale_ranges = None
        annotations = []
        for i in img_info_list:
            for j in i:
                annotations.append(j)
        pool = Pool(nproc)
        eval_results = []
        for i in range(num_classes):
            # get gt and det bboxes of this class
            cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
                results, annotations, i)
            # choose proper function according to datasets to compute tp and fp

            tpfp_fn = tpfp_default
            if not callable(tpfp_fn):
                raise ValueError(
                    f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

            # compute tp and fp for each image with multiple processes
            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)]))
            tp, fp = tuple(zip(*tpfp))
            # calculate gt number of each scale
            # ignored gts or gts beyond the specific scale are not counted
            num_gts = np.zeros(num_scales, dtype=int)
            for j, bbox in enumerate(cls_gts):
                if area_ranges is None:
                    num_gts[0] += bbox.shape[0]
                else:
                    gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                            bbox[:, 3] - bbox[:, 1])
                    for k, (min_area, max_area) in enumerate(area_ranges):
                        num_gts[k] += np.sum((gt_areas >= min_area)
                                             & (gt_areas < max_area))
            # sort all det bboxes by score, also sort tp and fp
            cls_dets = np.vstack(cls_dets)
            num_dets = cls_dets.shape[0]
            sort_inds = np.argsort(-cls_dets[:, -1])
            tp = np.hstack(tp)[:, sort_inds]
            fp = np.hstack(fp)[:, sort_inds]
            # calculate recall and precision with tp and fp
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            # calculate AP
            if scale_ranges is None:
                recalls = recalls[0, :]
                precisions = precisions[0, :]
                num_gts = num_gts.item()
            dataset='voc07'
            mode = 'area' if dataset != 'voc07' else '11points'
            ap = average_precision(recalls, precisions, mode)
            eval_results.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })
        pool.close()
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

        print_map_summary(
            mean_ap, eval_results, dataset, area_ranges, logger=None)

        return mean_ap, eval_results

def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:

        gt_inds_list = np.argwhere(ann['default_gt_labels'] == class_id).squeeze(-1)
        for gt_inds in gt_inds_list:
            cls_gts.append(ann['default_gt_bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore

def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap
