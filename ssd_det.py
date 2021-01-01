import torch
import torch.nn as nn
from ssd import ssd
from ssd_header import ssd_header
import numpy as np

class SSD_DET(nn.Module):
    def __init__(self,
                 num_classes,
                 base=ssd,
                 header=ssd_header,
                 pretrained=False):
        super(SSD_DET, self).__init__()
        self.base = base()
        self.header = header(num_classes)
        self.pretrained = pretrained
        if self.pretrained:
            self.init_weights(pretrained=self.pretrained)

    def init_weights(self, pretrained):
        self.base.init_weights(pretrained=pretrained)
        self.header.init_weights()

    def extract_feat(self, img):
        x = self.base(img)
        return x

    def forward(self, img):
        x = self.extract_feat(img)

        return self.header(x)

    def forward_train(self, x, gt_bboxes, gt_labels, ):
        out = self(x)
        loss_input = out + (gt_bboxes, gt_labels, )
        losses = self.header.loss(*loss_input)
        return losses

    def loss_cal(self, cla_scores, loc_results, gt_bboxes, gt_labels,):
        return self.header.loss(cla_scores, loc_results, gt_bboxes, gt_labels)

    def simple_test(self, img, img_infos, rescale=False):
        out = self(img)
        bbox_list = self.header.get_bboxes(*out, img_infos, rescale=rescale)
        #print(bbox_list)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.header.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results

    def vis_anchor_match(self, x, gt_bboxes, gt_labels):
        out = self(x)
        loss_input = out + (gt_bboxes, gt_labels,)
        losses = self.header.loss(*loss_input, vis_match=True)
        return losses

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]