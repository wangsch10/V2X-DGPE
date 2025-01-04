import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()#inplace=False
        self.weight = weight #特征补偿图权重参数
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.weight*identity  # 残差连接
        out = self.relu(out)
        
        return out

class FeatureCompensationNet(nn.Module):
    def __init__(self, in_channels, out_channels, weight1=0.5, weight2=0.5, weight3=0.5):
        super(FeatureCompensationNet, self).__init__()
        self.res_block1 = ResidualBlock(in_channels, out_channels, weight=weight1)
        self.res_block2 = ResidualBlock(out_channels, out_channels, weight=weight2)
        self.res_block3 = ResidualBlock(out_channels, out_channels, weight=weight3)
        self.relu = nn.ReLU()#inplace=False

    def forward(self, x):
        x1 = self.res_block1(x)
        x2 = self.res_block2(x1)
        M  = self.res_block3(x2)

        return  self.relu(x + 0.1*M)


