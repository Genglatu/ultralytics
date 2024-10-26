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
    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', norm_layer(bn_size * growth_rate))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = self.conv1(self.norm1(x))
        new_features = self.conv2(self.norm2(new_features))
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=BatchNormAct2d):
        super(DenseBlock, self).__init__()
        self.layer = nn.ModuleList([
            DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, norm_layer=norm_layer)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

class DenseTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=BatchNormAct2d):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, layer_name="", idx=0, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, num_classes=1000, in_chans=3):
        super(DenseNet, self).__init__()
        self.layer_name = layer_name
        self.idx = idx

        # Initial convolution
        num_init_features = growth_rate * 2
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', BatchNormAct2d(num_init_features)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Dense blocks and transition layers
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = DenseTransition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', BatchNormAct2d(num_features))
        self.num_features = num_features

    def forward(self, x):
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if self.layer_name == "densenet_layer" and self.idx == i:
                return x
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)
                if self.layer_name == "densenet_transition_layer" and self.idx == i:
                    return x
        return x
        
class Densenet121(nn.Module):
    def __init__(self, cin=3, cout=64, idx=0, layer_name=""):
        super(Densenet121, self).__init__()
        self.densenet = DenseNet(layer_name=layer_name, idx=idx, growth_rate=32, block_config=(6, 12, 24, 16), in_chans=cin)

    def forward(self, x):
        return self.densenet(x)

class Densenet169(nn.Module):
    def __init__(self, cin=3, cout=64, idx=0, layer_name=""):
        super(Densenet169, self).__init__()
        self.densenet = DenseNet(layer_name=layer_name, idx=idx, growth_rate=32, block_config=(6, 12, 32, 32), in_chans=cin)

    def forward(self, x):
        return self.densenet(x)

class Densenet201(nn.Module):
    def __init__(self, cin=3, cout=64, idx=0, layer_name=""):
        super(Densenet201, self).__init__()
        self.densenet = DenseNet(layer_name=layer_name, idx=idx, growth_rate=32, block_config=(6, 12, 48, 32), in_chans=cin)

    def forward(self, x):
        return self.densenet(x)
