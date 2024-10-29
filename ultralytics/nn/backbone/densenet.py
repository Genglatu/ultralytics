"""Pytorch Densenet implementation w/ tweaks
This file is a copy of https://github.com/pytorch/vision 'densenet.py' (BSD-3-Clause) with
fixed kwargs passthrough and addition of dynamic global avg/max pool.
"""
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.jit.annotations import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import BatchNormAct2d, create_norm_act_layer, BlurPool2d, create_classifier


__all__ = ['DenseNet']


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.conv0', 'classifier': 'classifier',
    }


default_cfgs = {
    'densenet121': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenet121_ra-50efcf5c.pth'),
    'densenet121d': _cfg(url=''),
    'densenetblur121d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenetblur121d_ra-100dcfbc.pth'),
    'densenet169': _cfg(url='https://download.pytorch.org/models/densenet169-b2777c0a.pth'),
    'densenet201': _cfg(url='https://download.pytorch.org/models/densenet201-c1103571.pth'),
    'densenet161': _cfg(url='https://download.pytorch.org/models/densenet161-8d451a50.pth'),
    'densenet264': _cfg(url=''),
    'densenet264d_iabn': _cfg(url=''),
    'tv_densenet121': _cfg(url='https://download.pytorch.org/models/densenet121-a639ec97.pth'),
}


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            BatchNormAct2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            BatchNormAct2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.layer(x)
        x = torch.cat([x, out], dim=1)  # Concatenate along channels dimension
        assert isinstance(x, torch.Tensor), "Expected tensor but got list"
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        print("DenseBlock output type:", type(x))
        assert isinstance(x, torch.Tensor), "DenseBlock should return tensor, got list"
        return x

class DenseTransition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DenseTransition, self).__init__()
        self.add_module('norm', BatchNormAct2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = super().forward(x)
        print("DenseTransition output type:", type(x), "DenseTransition output shape:", x.shape)
        return x

class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNormAct2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block1 = DenseBlock(num_layers=6, in_channels=64, growth_rate=32, bn_size=4)
        self.trans1 = DenseTransition(in_channels=256, out_channels=128)
        self.block2 = DenseBlock(num_layers=12, in_channels=128, growth_rate=32, bn_size=4)
        self.trans2 = DenseTransition(in_channels=512, out_channels=256)
        self.block3 = DenseBlock(num_layers=24, in_channels=256, growth_rate=32, bn_size=4)
        self.trans3 = DenseTransition(in_channels=1024, out_channels=512)
        self.block4 = DenseBlock(num_layers=16, in_channels=512, growth_rate=32, bn_size=4)
        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = self.final_conv(x)
        return x
