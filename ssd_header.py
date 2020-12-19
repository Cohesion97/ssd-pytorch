import torch
import torch.nn as nn
from ssd import ssd
from init_weights import *
class ssd_header(nn.Module):

    default_inplanes = [512, 1024, 512,256,256,256]

    def __init__(self,
                 num_classes,
                 num_anchors=[4,6,6,6,4,4]):
        super(ssd_header,self).__init__()
        assert len(self.default_inplanes)==len(num_anchors)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        cla_convs = []
        loc_convs = []
        for i in range(len(self.default_inplanes)):
            cla_convs.append(nn.Conv2d(self.default_inplanes[i],
                                       (self.num_classes+1)*self.num_anchors[i],
                                       kernel_size=3,padding=1))
            loc_convs.append(nn.Conv2d(self.default_inplanes[i],
                                       4*self.num_anchors[i],
                                       kernel_size=3,padding=1))
        self.loc_convs = nn.ModuleList(loc_convs)
        self.cla_convs = nn.ModuleList(cla_convs)


    def forward(self,feats):
        cla_scores = []
        loc_results = []
        for feat, cla_layer, loc_layer in zip(feats, self.cla_convs, self.loc_convs):
            cla_scores.append(cla_layer(feat))
            loc_results.append(loc_layer(feat))
        return cla_scores, loc_results

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)




