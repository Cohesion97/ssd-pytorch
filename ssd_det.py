import torch
import torch.nn as nn
from ssd import ssd
from ssd_header import ssd_header

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


