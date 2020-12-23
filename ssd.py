import torch
import torch.nn as nn
from myvgg import vgg16
from init_weights import kaiming_init,constant_init,normal_init,xavier_init
from load_pretrained import load_checkpoint


class ssd(vgg16):
    def __init__(self,
                 input_size=300,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3,4),
                 out_feat_indices=(22,34),
                 l2_norm_scale=20.):
        super(ssd,self).__init__(
            with_last_pool=with_last_pool,
            out_indices=out_indices,
            ceil_mode=ceil_mode
        )
        assert input_size in (300,512)
        self.input_size = input_size
        self.out_feature_indices = out_feat_indices

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))

        extra_layers = [nn.Conv2d(1024,256,kernel_size=1),
                        nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
                        nn.Conv2d(512,128,kernel_size=1),
                        nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
                        nn.Conv2d(256,128,kernel_size=1),
                        nn.Conv2d(128,256,kernel_size=3),
                        nn.Conv2d(256, 128, kernel_size=1),
                        nn.Conv2d(128, 256, kernel_size=3),
                        ]
        self.add_module('extra',nn.Sequential(*extra_layers))
        self.l2_norm = L2Norm(self.features[out_indices[0]-1].out_channels, l2_norm_scale)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=None)
            print('load pretrained checkpoint from {}'.format(pretrained))
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        for i, layer in enumerate(self.extra):
            x = nn.functional.relu(layer(x),inplace=True)
            if i % 2 == 1:
                outs.append(x)
        outs[0] = self.l2_norm(outs[0])
        return tuple(outs)

# copy from mmdet
class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)