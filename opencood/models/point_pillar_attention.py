# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from opencood.models.fuse_modules.fusion_in_one import BasicBlock, BasicBlock_Rescale

import numpy as np

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


    def forward(self, x):
        B, C, H, W = x.shape
        out = []
        concat = torch.cat([x, x], dim=1).view(B, 2*C, H, W)
        shared = self.non_linear(concat) #(1, 2*C, H, W)
        # Spatial mask across different datasets
        # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

        # dataset attention mask
        # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
        # mask = mask * spatial_att
        shared = torch.mul(shared.view(B, C, 2, H, W), concat.view(B, C, 2, H, W)).view(B,-1, H, W)

        # Perform dimensionality reduction 
        x = self.dimensionality_reduction(shared)
        return x

class AttFusion6_2(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion6_2, self).__init__()
        self.shared_channels = feature_dims
        print('using att62 attetnion fusion with RT matrix.')

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, feature_dims)


    def forward(self, x):
        B, C, H, W = x.shape
        out = []
        # x2 = torch.zeros_like(x).to(x) #place for cav2

        # if np.random.random() > 0.5:
        #     concat = torch.cat([x, x2], dim=1).view(B, 2*C, H, W)
        # else:
        #     concat = torch.cat([x2, x], dim=1).view(B, 2*C, H, W)
        shared = self.non_linear(x) #(1, 2*C, H, W)
        # Spatial mask across different datasets
        # spatial_att = torch.max(concat, dim=1).values.view(1, 1, 1, H, W) 

        # dataset attention mask
        # mask = F.softmax(shared.view(1, C, 2, H, W), dim = 2)
        # mask = mask * spatial_att
        # shared = torch.mul(shared.view(B, C, 2, H, W), concat.view(B, C, 2, H, W)).view(B,-1, H, W)

        # Perform dimensionality reduction 
        x = self.dimensionality_reduction(shared)
        return x


class PointPillarAttention(nn.Module):
    def __init__(self, args):
        super(PointPillarAttention, self).__init__()
        print('using PointPillarAttention as early fusion model')

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        if 'fusion_net' in args['fusion_args']:
            if args['fusion_args']['fusion_net'] == 'att62':
                self.fusion_net = AttFusion6_2(self.out_channel)


        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], # 384
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], # 384
                                  kernel_size=1)
        
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        spatial_features_2d = self.fusion_net(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict