import torch
import torch.nn as nn
from ssd import ssd
from init_weights import *

def pairs(x):
    assert type(x) != str
    return [x,x]

class ssd_header(nn.Module):

    default_inplanes = [512, 1024, 512,256,256,256]

    def __init__(self,
                 num_classes,
                 anchor_cfg={'strides':[8,16,32,64,100,300],
                             'ratios':[[2],[2,3],[2,3],[2,3],[2,],[2]],
                             'scale_range':[0.15, 0.9],
                             'input_size':300},
                 ):
        super(ssd_header,self).__init__()
        self.num_classes = num_classes
        self.anchor_cfg = anchor_cfg
        assert len(self.default_inplanes)==len(self.anchor_cfg['strides'])
        self.num_multi_feats = len(self.default_inplanes)

        self.anchor_strides = self.anchor_cfg['strides']
        self.anchor_ratios = self.anchor_cfg['ratios']
        self.anchor_input_size = self.anchor_cfg['input_size']

        self.base_len = [int(0.07 * self.anchor_input_size)]
        small, big = self.anchor_cfg['scale_range']
        self.step = (big - small) / (self.num_multi_feats - 2)
        self.step_ = int(self.step * self.anchor_input_size)
        for i in range(int(small*100), int(big*100+1), int(self.step*100)):
            self.base_len.append(int(i * self.anchor_input_size / 100))
        gen_ratios = []
        gen_scales = []
        for j, item in enumerate(self.base_len):
            try:
                gen_scale = torch.Tensor([1.,
                              np.sqrt(self.base_len[j+1] /
                                      self.base_len[j])])
            except:
                gen_scale = torch.Tensor([1.,
                              np.sqrt((self.base_len[j]+self.step_) /
                                      self.base_len[j])])
            gen_scales.append(gen_scale
                )
            gen_ratio = [1.,]
            for k in self.anchor_ratios[j]:
                gen_ratio.extend([k, 1/k])
            gen_ratios.append(
                torch.Tensor(gen_ratio)
            )
        self.gen_ratios = gen_ratios
        self.gen_scales = gen_scales
        self.base_anchors = self.gen_anchor_single_img()
        self.num_anchors = [len(num_anchor) for num_anchor in self.base_anchors ]
        from IPython import embed;embed()

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

    def gen_anchor_single_feat(self,base_len,stride,ratios,scales):
        w, h = base_len
        w_center = int(stride/2)
        h_center = int(stride/2)

        h_ratio = torch.sqrt(ratios)
        w_ratio = 1 / h_ratio

        d_w = w * scales[:,None] * w_ratio[None,:]
        d_h = h * scales[:,None] * h_ratio[None,:]

        d_w = d_w.view(-1)
        d_h = d_h.view(-1)

        anchors = [w_center - 0.5 * d_w,
                   h_center - 0.5 * d_h,
                   w_center + 0.5 * d_w,
                   h_center + 0.5 * d_h,
                   ]

        anchors = torch.stack(anchors, dim=-1)
        index = list(range(len(ratios)+1))
        return torch.index_select(anchors,0,torch.LongTensor(index))

    def gen_anchor_single_img(self,):
        mulit_feat = []
        for i in range(len(self.base_len)):
            mulit_feat.append(self.gen_anchor_single_feat(pairs(self.base_len[i]),
                                                          self.anchor_strides[i],
                                                          self.gen_ratios[i],
                                                          self.gen_scales[i]))
        return mulit_feat

    def grid_anchors(self):
        """
        :argument:
            self.base_anchors: List[Tensor]
            self.anchor_strides: List[int]
        :return:
        """
        featuremap_size = [int(np.ceil(self.anchor_input_size / i)) for i in self.anchor_strides]
        multi_grid_anchors = []
        for num_levels in range(len(self.base_anchors)):
            x = torch.from_numpy(np.array(
                range(0, self.anchor_input_size, self.anchor_strides[num_levels])))
            y = torch.from_numpy(np.array(
                range(0, self.anchor_input_size, self.anchor_strides[num_levels])))

    def shift(self,x,y):
        """
        :param x: Tensor, w of featmap
        :param y: int, h of featmap
        :return: shift: Tensor, w*h
        """
        x_ = x.repeat(len(y)).view(-1)
        y_ = y.view(-1,1).repeat(1,len(x)).view(-1)
        shift = [x_, y_, x_, y_]
        shift = torch.stack(shift, )


