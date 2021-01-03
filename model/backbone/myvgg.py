import torch.nn as nn
from model.init_weights import kaiming_init,constant_init,normal_init
from model.load_pretrained import load_checkpoint
import logging

class vgg16(nn.Module):

    arch_settings = (2, 2, 3, 3, 3)

    def __init__(self,
                 with_bn=False,
                 num_classes=-1,
                 num_stages=5,
                 out_indices=(0,1,2,3,4),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True
                 ):
        super(vgg16,self).__init__()
        self.stage_blocks = self.arch_settings[:num_stages]

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []

        def vgg_blocks(inplanes, planes, num_blocks, with_bn=with_bn, ceil_mode=ceil_mode):
            layers = []
            for _ in range(num_blocks):
                layers.append(nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,
                                        padding=1,dilation=1))
                if with_bn:
                    layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))
                inplanes = planes
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=ceil_mode))
            return layers

        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            self.range_sub_modules.append([start_idx, end_idx])
            planes = 64 * 2 ** i if i < 4 else 512
            vgg_layer = vgg_blocks(self.inplanes,
                                   planes,
                                   num_blocks,
                                   with_bn=with_bn,
                                   ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self,self.module_name)
        for i in range(len((vgg_layers))):
             layer = vgg_layers[i]
             x = layer(x)
             if i in self.out_indices:
                 outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained,strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')


    def train(self, mode=True):
        super(vgg16, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False