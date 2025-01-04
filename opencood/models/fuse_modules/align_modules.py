from torch.nn import functional as F
import torch
from torch import nn

def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.Sigmoid()
    )

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = norm_layer(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GatedConv(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(GatedConv, self).__init__()
        self.feat_out1 = nn.Conv2d(inplane, inplane//2, 1)
        self.feat_out2 = nn.Conv2d(inplane, inplane//2, 1)
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(inplane),
            nn.Conv2d(inplane, inplane//2, 1),
            nn.ReLU(),
            nn.Conv2d(inplane//2, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out = nn.Conv2d(outplane, outplane, 1)

    def forward(self, feats, x, base_idx=1):
        low_feature, h_feature = feats
        size = low_feature.size()[2:]

        feat1 = self.feat_out1(low_feature)
        feat2 = self.feat_out2(h_feature)
        feat2 = F.upsample(feat2, size=size, mode="bilinear", align_corners=True)

        alphas = self._gate_conv(torch.cat([feat1, feat2], dim=1))
        x = (x * (alphas + 1))
        x = self.out(x)
        return x

class SimpleGate(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(SimpleGate, self).__init__()
        self.gate = conv_sigmoid(inplane, 2)

    def forward(self, feats, x, base_idx=1):
        size = feats[0].size()[2:]
        feature_origin = feats[base_idx]
        flow_gate = F.upsample(self.gate(feature_origin), size=size, mode="bilinear", align_corners=True)
        x = x*flow_gate
        return x


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, gate='simple'):
        super(AlignedModule, self).__init__()

        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)
        if gate == "simple":
            self.gate = SimpleGate(inplane, 2)
        elif gate == "GatedConv":
            self.gate = GatedConv(inplane, 2)
        elif gate == "":
            self.gate = None
        else:
            raise ValueError("no this type of gate")

    def forward(self, x, base_idx=1):
        low_feature, h_feature= x
        # h_feature_orign = h_feature
        feature_origin = x[base_idx]
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        if self.gate:
            flow = self.gate(x, flow, base_idx)
        h_feature = self.flow_warp(feature_origin, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
