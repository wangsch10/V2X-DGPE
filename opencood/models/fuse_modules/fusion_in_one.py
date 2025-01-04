"""
A model zoo for intermediate fusion.
Please make sure your pairwise_t_matrix is normalized before using it.
Enjoy it.
"""

import torch
from torch import nn
from icecream import ic
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.fuse_modules.where2comm_attn import TransformerFusion
import torch.nn.functional as F
import numpy as np
from opencood.models.fuse_modules.domain_attention import SELayer
from opencood.models.fuse_modules.wg_fusion_modules import CrossDomainFusionEncoder

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def warp_feature(x, record_len, pairwise_t_matrix):
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        i = 0 # ego
        neighbor_feature = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W))
        out.append(neighbor_feature)

    out = torch.cat(out, dim=0)
    
    return out

class DimReduction(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=None):
        super(DimReduction, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class DomainEncoder(nn.Module):
    def __init__(self, inplanes, planes):
        super(DomainEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
  
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        identity = self.bn2(self.conv2(x))
        out += identity
        out = self.relu(out)
        out = self.conv3(out)
        return out

class SpatialEncoder(nn.Module):
    def __init__(self, inplanes, planes):
        super(SpatialEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out2 = torch.max(x, dim=1).values
        out += out2

        return out



class SpatialEncoder2(nn.Module):
    def __init__(self, inplanes, planes):
        super(SpatialEncoder2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
  
        self.conv2 = nn.Conv2d(inplanes, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(x))
        #out3 = torch.max(x, dim=1).values
        out3 = torch.max(x, dim=1, keepdim=True).values
        out = out + out2 + out3

        return out

class SpatialEncoder3(nn.Module):
    def __init__(self, inplanes, planes):
        super(SpatialEncoder3, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, 1, kernel_size=3, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(1)
  
        self.conv2 = nn.Conv2d(inplanes, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        #out = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(x))
        out = out2

        return out

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class MaxFusion_2(nn.Module):
    def __init__(self):
        super(MaxFusion_2, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = batch_node_features[b]
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class MaxFusion_multiscale(nn.Module):
    def __init__(self, args):
        super(MaxFusion_multiscale, self).__init__()
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'MAX':
                    fuse_network = MaxFusion2()
                self.fuse_modules.append(fuse_network)

        print('using SumFusion_multiscale, with RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4) # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            return  x_fuse, feat_list

        else:
            split_x = regroup(x, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                #(N, C, H, W)
                neighbor_feature = warp_affine_simple(split_x[b],
                                                t_matrix[i, :, :, :],
                                                (H, W))
                # (N, C, H, W)
                feature_fused = torch.sum(neighbor_feature, dim=0)
                out.append(feature_fused)

            return torch.stack(out)


class MaxFusion2(nn.Module):
    def __init__(self):
        super(MaxFusion2, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]

class MaxFusion3(nn.Module):
    def __init__(self):
        super(MaxFusion3, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)

class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)
        print('using attetnion fusion with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class OverlapAware_AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(OverlapAware_AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)
        print('using overlap-aware attetnion fusion with RT matrix.')

    def generate_overlapmask(self, t_matrix, x):
        B, C, H, W = x.shape
        overlap_mask = torch.ones(B, 1, H, W).to(x)
        grid = F.affine_grid(t_matrix, [B, 1, H, W], align_corners=True).to(x)
        overlap_mask = F.grid_sample(overlap_mask, grid, align_corners=True) 
        #visualize:
        # from torchvision import transforms
        # unloader = transforms.ToPILImage()
        # image = overlap_mask[1].cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        # image = unloader(image)
        # image.save('/home/lixiang/CoAlign/images/overlap.jpg')

        return overlap_mask


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]

            overlap_mask = self.generate_overlapmask(t_matrix[i, :, :, :], x) #[cav_num, 1, H, W]

            #x = x * overlap_mask

            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            overlap_mask = overlap_mask.view(cav_num, 1, -1).permute(2, 0, 1)

            h = self.att(x*overlap_mask, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class AttFusion2(nn.Module):
    def __init__(self, feature_dims, args):
        super(AttFusion2, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)

        print('using Attention_multiscale, with RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4) # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    cav_num, C, H, W = node_features.shape
                    feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                    feature = feature.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
                    self.att.sqrt_dim = np.sqrt(C)
                    h = self.att(feature, feature, feature)
                    h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
                    x_fuse.append(h)
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            return  x_fuse, feat_list
        else:        
            split_x = regroup(x, record_len)
            batch_node_features = split_x
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                # update each node i
                i = 0 # ego
                x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
                cav_num = x.shape[0]
                x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
                h = self.att(x, x, x)
                h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
                out.append(h)

            out = torch.stack(out)
        return out

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.shareMLP = nn.Sequential(
			nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avgout = self.shareMLP(self.avg_pool(x))
		maxout = self.shareMLP(self.max_pool(x))
		return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()
		assert kernel_size in (3, 7), "kernel size must be 3 or 7"
		padding = 3 if kernel_size == 7 else 1

		self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		avgout = torch.mean(x, dim=1, keepdim=True)
		maxout, _ = torch.max(x, dim=1, keepdim=True)
		x = torch.cat([avgout, maxout], dim=1)
		x = self.conv(x)
		return self.sigmoid(x)

class SpatialAttV2X(nn.Module):
    def __init__(self, inplanes, kernel_size=3, padding=1):
        super(SpatialAttV2X, self).__init__()
        self.norm_layer = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.relu(x)
        return x

class AttFusion3(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion3, self).__init__()
        self.ca = ChannelAttention(feature_dims)
        self.sa = SpatialAttention()
        print('using se attetnion fusion with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            x = torch.max(x, dim=0)[0].view(1, C, H, W) #(C, H, W)
            x = self.ca(x) * x
            x = self.sa(x) * x
            #x = x.view(1, C, -1).permute(0, 2, 1) #  (1, H*W, C), perform spatial attention
            #h = self.att(x, x, x)
            #h = h.view(1, C, H, W)[0, ...]  # C, W, H before
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion4(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion4, self).__init__()
        self.ca = ChannelAttention(feature_dims)
        self.sa = SpatialAttention()
        self.conv = nn.Conv2d(2*feature_dims, feature_dims, 1, bias=False)
        print('using se attetnion fusion with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x = torch.cat([x[0], x[1]], dim=0).view(1, 2*C, H, W)
                x = self.conv(x)
            #x = torch.max(x, dim=0)[0].view(1, C, H, W) #(C, H, W)
            x = self.ca(x) * x
            x = self.sa(x) * x
                
            #x = x.view(1, C, -1).permute(0, 2, 1) #  (1, H*W, C), perform spatial attention
            #h = self.att(x, x, x)
            #h = h.view(1, C, H, W)[0, ...]  # C, W, H before
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion5(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion5, self).__init__()
        self.ca = ChannelAttention(feature_dims)
        self.sa = SpatialAttention()
        print('using se attetnion fusion with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                x1 = self.ca(x1) * x1
                x2 = self.ca(x2) * x2
                x = torch.cat([x1, x2], dim=0)
            x = torch.max(x, dim=0)[0].view(1, C, H, W) #(C, H, W)
            x = self.ca(x) * x
            x = self.sa(x) * x
                
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att6 attetnion fusion with RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat)
                # Spatial mask across different datasets
                spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                mask = mask * spatial_att
                shared = torch.mul(mask, concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6_1(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_1, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att61 attetnion fusion with RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                shared = torch.mul(shared.view(1, C, 2, H, W), concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6_1_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_1_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att61 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                shared = torch.mul(shared.view(1, C, 2, H, W), concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att62 attetnion fusion with RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=1).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                # shared = torch.mul(shared.view(1, C, 2, H, W), concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6_2_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_2_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att62 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = batch_node_features[b]
            
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=1).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out


class AttFusion6_3(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_3, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att63 attetnion fusion with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            domain_att = torch.max(x, dim=0).values #[1, C, H, W]
            spatial_att = torch.max(x.view(1, -1, H, W), dim=1).values.view(1, 1, H, W)
            x =  spatial_att * domain_att #[1, C, H, W]
            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion6_4_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_4_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att64 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.sa = SpatialAttV2X(self.shared_channels)


    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = batch_node_features[b] #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat)
                # Spatial mask across different datasets
                spatial_att = self.sa(concat).view(1, 1, 1, H, W)  #torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                mask = mask * spatial_att
                shared = torch.mul(mask, concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out


class AttFusion7(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion7, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att7 attetnion fusion with RT matrix.')

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)

                #adaptor bank
                x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
                concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
                #domain weight
                domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
                domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]

                weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
                weight = self.sigmoid(weight)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(weight*concat)

            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion7_1(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion7_1, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att71 attetnion fusion with RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)

                #adaptor bank
                x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
                concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
                #domain weight
                domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
                domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]
                weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
                weight = self.sigmoid(weight)

                weighted_concat = weight*concat #(1, 2*C, H, W)
                shared = self.non_linear(weighted_concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                # spatial_att = torch.max(weighted_concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                shared = torch.mul(shared.view(1, C, 2, H, W), weighted_concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)

            out.append(x[0])

        out = torch.stack(out)
        return out

class AttFusion7_1_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion7_1_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att71_2 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            #x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)

                #adaptor bank
                x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
                concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
                #domain weight
                domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
                domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]
                weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
                weight = self.sigmoid(weight)

                weighted_concat = weight*concat #(1, 2*C, H, W)
                shared = self.non_linear(weighted_concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                # spatial_att = torch.max(weighted_concat, dim=1).values.view(1, 1, 1, H, W) 

                # dataset attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                shared = torch.mul(shared.view(1, C, 2, H, W), weighted_concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)

            out.append(x[0])

        out = torch.stack(out)
        return out

class AlignFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AlignFusion, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using align_att71 fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.down_v = nn.Conv2d(feature_dims, feature_dims, 1, bias=False)
        self.down_i = nn.Conv2d(feature_dims, feature_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(feature_dims*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            x = batch_node_features[b]
            
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                #algin feature
                x1 = self.down_v(x1)
                x2 = self.down_i(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 
                # domain attention mask
                mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                mask = mask * spatial_att
                shared = torch.mul(mask, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out
    
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


class SimpleGate(nn.Module):
    def __init__(self, inplane, outplane):
        super(SimpleGate, self).__init__()
        self.gate = nn.Sequential(
        nn.Conv2d(inplane, outplane, 3, 1, 1),
        nn.Sigmoid()
    )

    def forward(self, feature_origin, flow):
        flow_gate = self.gate(feature_origin)
        flow = flow*flow_gate
        return flow

class AlignFusion2(nn.Module):
    def __init__(self, feature_dims):
        super(AlignFusion2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using align2_att6 fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.down_v = nn.Conv2d(feature_dims, feature_dims, 1, bias=False)
        self.down_i = nn.Conv2d(feature_dims, feature_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(feature_dims*2, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(feature_dims, 2)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            x = batch_node_features[b]
            
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                #algin feature
                x1 = self.down_v(x1)
                x2 = self.down_i(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))

                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                shared = self.non_linear(concat) #(1, 2*C, H, W)
                # # Spatial mask across different datasets
                # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 
                # # domain attention mask
                # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
                # mask = mask * spatial_att
                # shared = torch.mul(shared, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)
            out.append(x[0])

        out = torch.stack(out)
        return out
    
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

class DomainGeneralizedFeatureFusion(nn.Module):
    def __init__(self, single_dims):
        super(DomainGeneralizedFeatureFusion, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                # spatial attention 
                spatial_attention = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 
                attention = domain_attention * spatial_attention 

                fused_feature = torch.mul(attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)
        return out
    
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

class DomainGeneralizedFeatureFusion2(nn.Module):
    def __init__(self, single_dims):
        super(DomainGeneralizedFeatureFusion2, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF2 fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                
                spatial_attention = self.spatial_encoder(concat).view(1, 1, 1, H, W)  #(1, 1, H, W)

                fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)
        return out
    
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


class DomainGeneralizedFeatureFusion3(nn.Module):
    def __init__(self, single_dims, vis_feats):
        super(DomainGeneralizedFeatureFusion3, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF3 fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = vis_feats

    def forward(self, xx, spatial_features_2d_history,record_len,record_len_history, history_mark,pairwise_t_matrix):
        # #print(xx.shape)#[1, 256, 100, 252]
        # # xx= xx.view(-1, 100, 252, 256)#[4, 2, 100, 252, 256]
        # # xx= xx.permute(0,1, 4, 2, 3)
        # _,C, H, W = 1, 256, 100, 252
        # #print(xx.shape)
        # B, L = pairwise_t_matrix.shape[:2]
        # #split_x = regroup(xx, record_len)
        # #split_x=xx
        # batch_node_features = xx
        # out = []
        
                # xx= xx.view(-1, 100, 252, 256)#[4, 2, 100, 252, 256]
        #xx= xx.permute(0,1, 4, 2, 3)
        
        _,C, H, W = xx.shape
        print(xx.shape)#[2, 256, 100, 252]
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        split_x=torch.stack(split_x, dim=0)
        '''
        spatial_features_2d_history=regroup(spatial_features_2d_history, record_len_history)
        
        spatial_features_2d_history = torch.stack(spatial_features_2d_history, dim=0)
        '''
        ####对历史数据的操作
        x_train=split_x
        #history_mark决定是否参与训练,对于没有前一帧的特征，就不参加训练
        replace_indices =torch.where(history_mark == 1)[0]
        history_out_inf=split_x[replace_indices][:, 1:2, :, :, :]
        ####################################################################################################时延测试start
        time_delay=0
        if time_delay==1:
            # print('2222222222222222222222')replace_indices
            # print('delay...............................................')
            history_out_inf=spatial_features_2d_history[replace_indices][:, 1:2, :, :, :]
            print(replace_indices)
        ####################################################################################################时延测试end
        #替换history_mark == False的history_out对应的batch
        #print(history_mark)
        history_out_cav=split_x[replace_indices][:, 0:1, :, :, :]
        history_out=torch.cat((history_out_cav, history_out_inf), dim=1)
        # print(history_out[torch.where(history_mark == 0)[0]]) 
        if xx.shape[0]==2:
            print('sdsdsdsdsds')
            x_train[replace_indices] = history_out            
        x_train=torch.unbind(x_train, dim=0)

        #split_x=xx
        batch_node_features = x_train
        out = []
        
        # iterate each batch

        for b in range(B):
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                # print(x[0].shape)
                # print(x[0])
                x1 = x[0].view(1, C, H, W)#[256, 100, 252]
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                
                spatial_attention = self.spatial_encoder(concat).view(1, 1, 1, H, W)  #(1, 1, H, W)

                fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)

        if self.vis_feats:
            return  out, batch_node_features[0]
        else:
            return out
    
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
class DomainGeneralizedFeatureFusion4(nn.Module):
    def __init__(self, single_dims, vis_feats):
        super(DomainGeneralizedFeatureFusion4, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF3 fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = vis_feats

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch

        for b in range(B):
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                
                spatial_attention = self.spatial_encoder(concat).view(1, 1, 1, H, W)  #(1, 1, H, W)

                fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)

        if self.vis_feats:
            return  out, batch_node_features[0]
        else:
            return out
    
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
class DGFF(nn.Module):
    def __init__(self, single_dims, vis_feats):
        super(DGFF, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFF3 fusion with RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = vis_feats

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch

        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                # algin feature
                x1 = self.down_vehicle(x1)
                x2 = self.down_inf(x2)
                flow = self.flow_make(torch.cat([x1, x2], 1))
                flow = self.gate(x1, flow)
                x2 = self.flow_warp(x2, flow, size=(H, W))
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
                # domain attention
                domain_attention = self.domain_encoder(concat) #(1, 2*C, H, W)
                domain_attention = F.softmax(domain_attention.view(1, C, 2, H, W), dim = 2)
                
                spatial_attention = self.spatial_encoder(concat).view(1, 1, 1, H, W)  #(1, 1, H, W)

                fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(1, C, 2, H, W)).view(1,-1, H, W)
                # dimension reduction 
                x = self.dim_reduction(fused_feature)
            out.append(x[0])

        out = torch.stack(out)

        if self.vis_feats:
            return  out, batch_node_features[0]
        else:
            return out
    
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

class DAF(nn.Module):
    def __init__(self, single_dims, vis_feats):
        super(DAF, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DAF fusion with RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = vis_feats

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []
        # iterate each batch

        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],t_matrix[i, :, :, :],(H, W)) #(N, C, H, W)
            num_cav = neighbor_feature.shape[0]

            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            concat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # domain attention
            domain_attention = self.domain_encoder(concat) #(N, 2*C, H, W)
            domain_attention = F.softmax(domain_attention.view(N, C, 2, H, W), dim = 2)

            spatial_attention = self.spatial_encoder(concat).view(N, 1, 1, H, W)  #(N, 1, H, W)

            fused_feature = torch.mul(domain_attention*spatial_attention, concat.view(N, C, 2, H, W)).view(N,-1, H, W)
            fused_feature = torch.sum(fused_feature, dim=0, keepdim=True) #(1, 2C, H, W)
            # dimension reduction 
            fused_feature = self.dim_reduction(fused_feature) #(1, C, H, W)
            out.append(fused_feature[0])
        out = torch.stack(out)

        if self.vis_feats:
            return  out, split_x[0]
        else:
            return out
    
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


class AttFusion7_3_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion7_3_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att73_2 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            #x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)

                #adaptor bank
                x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
                concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
                #domain weight
                domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
                domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]
                weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
                weight = self.sigmoid(weight)
                
                shared = self.non_linear(concat) * weight #(1, 2*C, H, W)

                shared = torch.mul(shared.view(1, C, 2, H, W), concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)

            out.append(x[0])

        out = torch.stack(out)
        return out


class AttFusion7_2_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion7_2_2, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att72_2 attetnion fusion without RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False), nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            #x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            x = batch_node_features[b]
            num_cav = x.shape[0]
            if num_cav > 1:
                x1 = x[0].view(1, C, H, W)
                x2 = x[1].view(1, C, H, W)
                concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)

                #adaptor bank
                x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
                concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
                #domain weight
                domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
                domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]
                weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
                weight = self.sigmoid(weight)

                weighted_concat = weight*concat #(1, 2*C, H, W)
                shared = self.non_linear(weighted_concat) #(1, 2*C, H, W)
                # Spatial mask across different datasets
                spatial_att = torch.max(weighted_concat, dim=1).values.view(1, 1, 1, H, W)
                spatial_att =  shared.view(1, C, 2, H, W) * spatial_att

                shared = torch.mul(spatial_att.view(1, C, 2, H, W), weighted_concat.view(1, C, 2, H, W)).view(1,-1, H, W)

                # Perform dimensionality reduction 
                x = self.dimensionality_reduction(shared)

            out.append(x[0])

        out = torch.stack(out)
        return out


class AttFusion8(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion8, self).__init__()
        self.shared_channels = feature_dims * 2
        print('using att8 attetnion fusion with RT matrix.')

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)

        self.num_adaptor = 2
        self.SE_Layers = nn.ModuleList([SELayer(self.shared_channels, with_sigmoid=False) for num_class in range(self.num_adaptor )])
        self.fc_1 = nn.Linear(self.shared_channels, self.num_adaptor )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W)) #(2, C, H, W)
            num_cav = x.shape[0]
            x1 = x[0].view(1, C, H, W)
            x2 = x[0].view(1, C, H, W)
            if num_cav > 1:
                x2 = x[1].view(1, C, H, W)

            concat = torch.cat([x1, x2], dim=0).view(1, 2*C, H, W)
            #adaptor bank
            x1_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
            if num_cav > 1:
                x2_adapt = self.SE_Layers[1](concat).view(1, 2*C, 1) #[1, 2C, 1]
            else:
                x2_adapt = self.SE_Layers[0](concat).view(1, 2*C, 1) #[1, 2C, 1]
            concat_adapt = torch.cat([x1_adapt, x2_adapt], dim=2) #[1, 2C, 2]
            #domain weight
            domain_weight = self.fc_1(self.avg_pool(concat).view(1, 2*C))
            domain_weight = self.softmax(domain_weight).view(1, self.num_adaptor, 1) #[1, 2, 1]

            weight = torch.matmul(concat_adapt, domain_weight).view(1, 2*C, 1, 1)
            weight = self.sigmoid(weight)

            # Perform dimensionality reduction 
            x = self.dimensionality_reduction(weight*concat)

            out.append(x[0])

        out = torch.stack(out)
        return out


class DiscoFusion(nn.Module):
    def __init__(self, feature_dims, vis_feats=False):
        super(DiscoFusion, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)
        print('using discofuison, with RT matrix.')
        self.vis_feats = vis_feats
    def forward(self, xx, spatial_features_2d_history,record_len, record_len_history,history_mark,pairwise_t_matrix):
        # _, C, H, W = xx.shape
        # B, L = pairwise_t_matrix.shape[:2]
        # split_x = regroup(xx, record_len)
        # out = []
        # split_x = regroup(xx, record_len)
        # out = []

        _,C, H, W = xx.shape
        # print(xx.shape)#[2, 256, 100, 252]
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        split_x=torch.stack(split_x, dim=0)
        spatial_features_2d_history=regroup(spatial_features_2d_history, record_len_history)
        
        spatial_features_2d_history = torch.stack(spatial_features_2d_history, dim=0)
        ####对历史数据的操作
        x_train=split_x
        #history_mark决定是否参与训练,对于没有前一帧的特征，就不参加训练
        replace_indices =torch.where(history_mark == 1)[0]
        history_out_inf=split_x[replace_indices][:, 1:2, :, :, :]
        ####################################################################################################时延测试start
        time_delay=1
        if time_delay==1:
            # print('2222222222222222222222')replace_indices
            # print('delay...............................................')
            history_out_inf=spatial_features_2d_history[replace_indices][:, 1:2, :, :, :]
            print(replace_indices)
        ####################################################################################################时延测试end
        #替换history_mark == False的history_out对应的batch
        #print(history_mark)
        history_out_cav=split_x[replace_indices][:, 0:1, :, :, :]
        history_out=torch.cat((history_out_cav, history_out_inf), dim=1)
        # print(history_out[torch.where(history_mark == 0)[0]]) 
        if xx.shape[0]==2:
            # print('sdsdsdsdsds')
            x_train[replace_indices] = history_out            
        x_train=torch.unbind(x_train, dim=0)

        # #split_x=xx
        split_x = x_train
        batch_node_features=split_x

        out = []

        for b in range(B):
            
            N = record_len[b]
            x = batch_node_features[b]
            print(x.shape)
            num_cav = x.shape[0]
            # if num_cav > 1:
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            # neighbor_feature = warp_affine_simple(split_x[b],
            #                                 t_matrix[i, :, :, :],
            #                                 (H, W))
            node_features = batch_node_features[b]
            C, H, W = node_features.shape[1:]
            # neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            neighbor_feature = split_x[b][1].view(1, C, H, W).expand(N, -1, -1, -1)

            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            
            # else:
            #     t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            #     i = 0 # ego
            #     node_features = batch_node_features[b]
            #     C, H, W = node_features.shape[1:]
            #     neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
            #     # (N, C, H, W)
            #     ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)

            #     # (N, 2C, H, W)
            #     neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            #     # (N, 1, H, W)
            #     agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            #     # (N, 1, H, W)
            #     agent_weight = F.softmax(agent_weight, dim=0)

            #     agent_weight = agent_weight.expand(-1, C, -1, -1)
            #     # (N, C, H, W)
            #     feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            '''
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            '''

            out.append(feature_fused)
        out = torch.stack(out)
        
        if self.vis_feats:
            return  out, neighbor_feature_cat
        else:
            return out

class DiscoFusion_res(nn.Module):
    def __init__(self, feature_dims):
        super(DiscoFusion, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)
        print('using discofuison_res, with RT matrix.')
    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []

        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))

            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)


class DiscoFusion2(nn.Module):
    def __init__(self, feature_dims, args):
        super(DiscoFusion2, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.pixel_weight_layers = nn.ModuleList()
            for num_filter in num_filters:
                self.pixel_weight_layers.append(PixelWeightLayer(num_filter))
        else:
            self.pixel_weight_layer = PixelWeightLayer(feature_dims)
        print('using discofuison, with RT matrix.')

    def forward(self, xx, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        feat_list = []
        out = []
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(xx)
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    #i = 0 # ego
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                    # (N, C, H, W)
                    ego_feature = node_features[0].view(1, C, H, W).expand(N, -1, -1, -1)
                    # (N, 2C, H, W)
                    neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
                    # (N, 1, H, W)
                    agent_weight = self.pixel_weight_layers[i](neighbor_feature_cat) 
                    # (N, 1, H, W)
                    agent_weight = F.softmax(agent_weight, dim=0)

                    agent_weight = agent_weight.expand(-1, C, -1, -1)
                    # (N, C, H, W)
                    feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
                    x_fuse.append(feature_fused)
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            return  x_fuse, feat_list
        
        else:
            split_x = regroup(xx, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                neighbor_feature = warp_affine_simple(split_x[b],
                                                t_matrix[i, :, :, :],
                                                (H, W))
                # (N, C, H, W)
                ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
                # (N, 2C, H, W)
                neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
                # (N, 1, H, W)
                agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
                # (N, 1, H, W)
                agent_weight = F.softmax(agent_weight, dim=0)

                agent_weight = agent_weight.expand(-1, C, -1, -1)
                # (N, C, H, W)
                feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
                out.append(feature_fused)

                return torch.stack(out)

class SumFusion(nn.Module):
    def __init__(self):
        super(SumFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []

        split_x = regroup(x, record_len)
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            #(N, C, H, W)
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # (N, C, H, W)
            feature_fused = torch.sum(neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)

class SumFusion_multiscale(nn.Module):
    def __init__(self, args):
        super(SumFusion_multiscale, self).__init__()
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)

        print('using SumFusion_multiscale, with RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4) # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                    x_fuse.append(torch.sum(neighbor_feature, dim=0))
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            return  x_fuse, feat_list

        else:
            split_x = regroup(x, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                #(N, C, H, W)
                neighbor_feature = warp_affine_simple(split_x[b],
                                                t_matrix[i, :, :, :],
                                                (H, W))
                # (N, C, H, W)
                feature_fused = torch.sum(neighbor_feature, dim=0)
                out.append(feature_fused)

            return torch.stack(out)


class SumFusion_multiscale2(nn.Module):
    def __init__(self, args):
        super(SumFusion_multiscale2, self).__init__()
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'MAX':
                    fuse_network = MaxFusion2()
                self.fuse_modules.append(fuse_network)
        print('using SumFusion_multiscale2, without RT matrix.')

    def forward(self, x, record_len, pairwise_t_matrix, backbone=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        feat_list = []
        
        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_node_features = regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    x_fuse.append(self.fuse_modules[i](node_features))
                x_fuse = torch.stack(x_fuse)
                feat_list.append(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
            return  x_fuse, feat_list

        else:
            split_x = regroup(x, record_len)
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0 # ego
                # (N, C, H, W)
                feature_fused = torch.sum(split_x[b], dim=0)
                out.append(feature_fused)

            return torch.stack(out)

class SumFusion_2(nn.Module):
    def __init__(self, vis_feats):
        super(SumFusion_2, self).__init__()
        self.vis_feats = vis_feats

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        split_x = regroup(x, record_len)
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            #(N, C, H, W)
            neighbor_feature = split_x[b]
            # (N, C, H, W)
            feature_fused = torch.sum(neighbor_feature, dim=0)
            out.append(feature_fused)
        out = torch.stack(out)

        if self.vis_feats:
            return  out, split_x[0]
        return out


class CDT_2(nn.Module):
    def __init__(self, feature_dims, args):
        super(CDT_2, self).__init__()
        self.cdt = CrossDomainFusionEncoder(args['cdt'])

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        out = []
        split_x = regroup(x, record_len)
        for b in range(B):
            N = record_len[b]
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # i = 0 # ego
            #(N, C, H, W)
            neighbor_feature = split_x[b]
            if len(neighbor_feature) > 1:
                ego_feature, cav_feature = neighbor_feature[0][None], neighbor_feature[1][None]
                cav_feature = self.cdt(ego_feature, cav_feature)
                neighbor_feature = torch.cat([ego_feature, cav_feature], dim=0)
            # (N, C, H, W)
            feature_fused = torch.sum(neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)


class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        print('transformer fusion: ', channels, n_head, dropout)
        
    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
            key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(C, H, W)

            out.append(fused_feature)
        out = torch.stack(out)
        return out



class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        from opencood.models.sub_modules.convgru import ConvGRU
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W'] # remember to modify for v2xsim dataset
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels] * num_gru_layers,
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = regroup(x, record_len)
        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

                updated_node_features = []

                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, i, :N, ...]
                    neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                   t_matrix[i, :, :, :],
                                                   (H, W))

                    # (N,C,H,W)
                    ego_agent_feature = batch_node_features[b][i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    #(N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    # message contains all feature map from j to ego i.
                    message = self.msg_cnn(neighbor_feature) * mask

                    # (C,H,W)
                    if self.agg_operator=="avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator=="max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_features[b][i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_features[b][i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W) -> (B, H, W, C) -> (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out

class V2XViTFusion(nn.Module):
    def __init__(self, args):
        super(V2XViTFusion, self).__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
        self.fusion_net = V2XTransformer(args['transformer'])

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, L)
        prior_encoding = \
            torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        
        # prior encoding should include [velocity, time_delay, infra], but it is not supported by all basedataset.
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []

        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], pairwise_t_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion. In perfect setting, there is no delay. 
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        spatial_correction_matrix = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        
        return fused_feature

class When2commFusion(nn.Module):
    def __init__(self, args):
        super(When2commFusion, self).__init__()
        import numpy as np
        from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator_v2, MIMOGeneralDotProductAttention, AdditiveAttentin

        self.in_channels = args['in_channels']
        self.feat_H = args['H']
        self.feat_W = args['W']
        self.query_size = args['query_size']
        self.key_size = args['key_size']
        

        self.query_key_net = policy_net4(self.in_channels)
        self.key_net = km_generator_v2(out_size=self.key_size)
        self.query_net = km_generator_v2(out_size=self.query_size)
        # self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)
        self.attention_net = AdditiveAttentin(self.key_size, self.query_size)

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
        
        weight: torch.Tensor
            Weight of aggregating coming message
            shape: (B, L, L)
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        updated_node_features = []
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            # (N,1,H,W)
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            query_key_maps = self.query_key_net(neighbor_feature)

            keys = self.key_net(query_key_maps).unsqueeze(0) # [N, C_k]
            query = self.query_net(query_key_maps[0].unsqueeze(0)).unsqueeze(0) # [1, C_q]

            neighbor_feature = neighbor_feature.unsqueeze(0) # [1, N, C, H, W]

            feat_fuse, prob_action = self.attention_net(query, keys, neighbor_feature, sparse=False)

            updated_node_features.append(feat_fuse)

        out = torch.cat(updated_node_features, dim=0)
        
        return out

class Where2commFusion(nn.Module):
    def __init__(self, args):
        super(Where2commFusion, self).__init__()

        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
 
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttFusion(num_filters[idx])
                elif self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                elif self.agg_mode == 'Transformer':
                    fuse_network = TransformerFusion(
                                                channels=num_filters[idx], 
                                                n_head=args['agg_operator']['n_head'], 
                                                with_spe=args['agg_operator']['with_spe'], 
                                                with_scm=args['agg_operator']['with_scm'])
                self.fuse_modules.append(fuse_network)
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   
            elif self.agg_mode == 'Transformer':
                self.fuse_network = TransformerFusion(
                                            channels=args['agg_operator']['feature_dim'], 
                                            n_head=args['agg_operator']['n_head'], 
                                            with_spe=args['agg_operator']['with_spe'], 
                                            with_scm=args['agg_operator']['with_scm'])     

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication:
                        batch_confidence_maps = regroup(rm, record_len)
                        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = regroup(x, record_len)
                
                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix[0, :, :, :],
                                                    (H, W))
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                ############ 4. Deconv ####################################
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse, communication_rates, {}