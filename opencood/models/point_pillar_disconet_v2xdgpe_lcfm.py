# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import numpy as np
import time

import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.data_utils.post_processor import UncertaintyVoxelPostprocessor
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_compensation import FeatureCompensationNet
from opencood.models.sub_modules.fuse_utils import regroup as regroup_vit
from opencood.utils.transformation_utils import normalize_pairwise_tfm, regroup
from opencood.models.fuse_modules.fusion_in_one import DiscoFusion, SumFusion, MaxFusion, SumFusion_multiscale, SumFusion_multiscale2, \
    AttFusion, AttFusion2, AttFusion4, AttFusion5, AttFusion6, AttFusion6_1, AttFusion6_2, AttFusion6_3, AttFusion7, AttFusion7_1, AttFusion7_1_2, AttFusion8, \
        AttFusion6_2_2, AttFusion6_4_2,  AttFusion6_1_2, AttFusion7_2_2, AttFusion7_3_2, DiscoFusion2, MaxFusion_2, SumFusion_2, TransformerFusion, CDT_2, \
             AlignFusion, AlignFusion2, DomainGeneralizedFeatureFusion, DomainGeneralizedFeatureFusion2, DomainGeneralizedFeatureFusion3, DAF
from opencood.models.sub_modules.v2xvit_basic import V2XTransformer


from opencood.models.sub_modules.naive_compress import NaiveCompressor

class PointPillarDiscoNetV2xdgpeLcfm(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNetV2xdgpeLcfm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # if 'resnet' in args['base_bev_backbone']:
        #     self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        # else:
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.voxel_size = args['voxel_size']
        
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        self.feature_compensation=FeatureCompensationNet(256,256)
        self.vis_feats = 0

        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            #compress spatially
            if 'stride' in args:
                stride = args['stride']
            else:
                stride = 1 
            self.naive_compressor = NaiveCompressor(256, args['compression'], stride)
           # print('using compression ratio {}, stride {}:'.format(args['compression'], stride))
        else:
            self.compression = False

        if 'fusion_net' in args['fusion_args']:
            if args['fusion_args']['fusion_net'] == 'disconet':
                if self.vis_feats:
                    self.fusion_net = DiscoFusion(self.out_channel, vis_feats=1)
                else:
                    self.fusion_net = DiscoFusion(self.out_channel, vis_feats=0)
            # elif args['fusion_args']['fusion_net'] == 'attention':
            #     self.fusion_net = AttFusion(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'max':
                self.fusion_net = MaxFusion()
            elif args['fusion_args']['fusion_net'] == 'daf':
                self.fusion_net = DAF(self.out_channel, vis_feats=0)
            # elif args['fusion_args']['fusion_net'] == 'dgff2':
            #     self.fusion_net = DomainGeneralizedFeatureFusion2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'dgff3':
                if self.vis_feats:
                    self.fusion_net = DomainGeneralizedFeatureFusion3(self.out_channel, vis_feats=1)
                else:
                    self.fusion_net = DomainGeneralizedFeatureFusion3(self.out_channel, vis_feats=0)
            elif args['fusion_args']['fusion_net'] == 'transformer':
                    self.fusion_net = TransformerFusion(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'v2xdgpe':
                    self.fusion_net = V2XTransformer(args['transformer'])
            else:
                self.fusion_net = SumFusion_multiscale2(args['fusion_args']) 
        else:
            self.fusion_net = SumFusion_multiscale(args['fusion_args']) 
        #print('using {} for student'.format(self.fusion_net))
        if 'fusion_args' in args.keys():
            self.multi_scale = args['fusion_args']['multi_scale']
        else:
            self.multi_scale = False
        #print('multi_scale status:', self.multi_scale)
        
        if 'early_distill' in args['fusion_args']:
            self.early_distill = args['fusion_args']['early_distill']
            #print('early_distill status:', self.early_distill)
        else:
            self.early_distill = False

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.upsample = nn.Sequential(
            nn.Upsample(size=(128, 256), mode='bilinear', align_corners=True))

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        


        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}


        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)#N, C, H, W


        batch_dict = self.backbone(batch_dict)
        
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # transformer fusion
        output_dict = {}

        output_dict.update({'feature_before_fusion': spatial_features_2d})

        return output_dict
