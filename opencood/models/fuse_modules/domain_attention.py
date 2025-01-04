import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=True):
        super(SELayer, self).__init__()
        self.with_sigmoid = with_sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_sigmoid:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class DomainAttention(nn.Module):
    def __init__(self, planes, reduction=16, nclass_list=None, fixed_block=False):
        super(DomainAttention, self).__init__()
        self.planes = planes
        num_adapters = cfg.num_adapters
        if num_adapters == 0:
            self.n_datasets = len(nclass_list)
        else:
            self.n_datasets = num_adapters
        self.fixed_block = fixed_block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.fixed_block or num_adapters == 1:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, with_sigmoid=False) for num_class in range(1)])
        elif num_adapters == 0:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, with_sigmoid=False) for num_class in nclass_list])
        else:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, with_sigmoid=False) for num_class in range(num_adapters)])
        self.fc_1 = nn.Linear(planes, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()

        if self.fixed_block:
            SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            weight = self.fc_1(self.avg_pool(x).view(b, c))
            weight = self.softmax(weight).view(b, self.n_datasets, 1)
            for i, SE_Layer in enumerate(self.SE_Layers):
                if i == 0:
                    SELayers_Matrix = SE_Layer(x).view(b, c, 1)
                else:
                    SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
            SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        return x*SELayers_Matrix