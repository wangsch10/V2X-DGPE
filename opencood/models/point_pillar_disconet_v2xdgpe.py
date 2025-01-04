# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import numpy as np
import time
import umap

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
from opencood.models.sub_modules.v2xdgpe_basic_deformable import V2XTransformer
from opencood.models.sub_modules.v2xdgpe_basic_deformable import STCM
from opencood.models.sub_modules.v2xdgpe_basic_deformable import ResidualBlockHistory




from opencood.models.sub_modules.naive_compress import NaiveCompressor

class PointPillarDiscoNetV2xdgpe(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNetV2xdgpe, self).__init__()
        
        self.history_encoder_cav=ResidualBlockHistory()
        #self.history_encoder_inf=ResidualBlockHistory()

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
            print('using compression ratio {}, stride {}:'.format(args['compression'], stride))
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
                    self.stcm=STCM(args['transformer']['encoder'])

            else:
                self.fusion_net = SumFusion_multiscale2(args['fusion_args']) 
        else:
            self.fusion_net = SumFusion_multiscale(args['fusion_args']) 
        # print('using {} for student'.format(self.fusion_net))

        if 'fusion_args' in args.keys():
            self.multi_scale = args['fusion_args']['multi_scale']
        else:
            self.multi_scale = False
        print('multi_scale status2:', self.multi_scale)
        
        if 'early_distill' in args['fusion_args']:
            self.early_distill = args['fusion_args']['early_distill']
            print('early_distill status:', self.early_distill)
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

    def forward(self, data_dict,history_spatial_features_2d):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        record_len = data_dict['record_len']
        record_len_history=data_dict['record_len_history']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        spatial_correction_matrix = data_dict['spatial_correction_matrix']#单位矩阵 1矩阵，不用考虑
        spatial_correction_matrix_history= data_dict['spatial_correction_matrix_history']
        
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)


        history_mark=data_dict['history_mark']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix,
                      'history_mark':history_mark}


        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)#N, C, H, W

        _, _, H0, W0 = batch_dict['spatial_features'].shape
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        batch_dict = self.backbone(batch_dict)

        
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        max_cav=2   #2
        
        #    N, C, H, W-> B, L, C, H, W
        regroup_feature, mask = regroup_vit(spatial_features_2d,
                                record_len,
                                max_cav) #self.max_cav=2=l  #mask tensor([[1, 1]], device='cuda:0')


        regroup_feature_history, history_mask = regroup_vit(history_spatial_features_2d,
                                record_len_history,
                                max_cav) #self.max_cav=2=l                        


        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)

        regroup_feature_history= regroup_feature_history.permute(0, 1, 3, 4, 2)

        # transformer fusion
        output_dict = {}
        if self.multi_scale: #false
            spatial_features_2d, multiscale_feats = self.fusion_net(batch_dict['spatial_features'], record_len, t_matrix, self.backbone)
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
            output_dict.update({'multiscale_feats': multiscale_feats})
        else:
            if self.vis_feats: ##false
                fusion_features_2d, single_features = self.fusion_net(spatial_features_2d, record_len, t_matrix)
            else:
                prior_encoding=regroup_feature[..., -3:]
                x_train=regroup_feature[..., :-3]
                replace_indices =torch.where(history_mark == 1)[0]

                batchsize=x_train.shape[0]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                prior_encoding_history = torch.from_numpy(np.tile([1, 1], (batchsize, 1))).to(device)
                regroup_feature_history=self.stcm(regroup_feature_history, prior_encoding_history,mask, spatial_correction_matrix_history)#spatial_correction_matrix_history用t时刻和t-1时刻自车的坐标计算

                history_out_cav=self.history_encoder_cav(x_train[replace_indices][:, 0:1, :, :, :],regroup_feature_history[replace_indices][:, 0:1, :, :, :])

                new_history_mask = history_mask[:, 1]
                replace_indices_inf =torch.where(new_history_mask == 1)[0]
                intersection = replace_indices[torch.isin(replace_indices, replace_indices_inf)]

                history_out_inf=x_train[replace_indices][:, 1:2, :, :, :]

                history_out_inf[torch.isin(replace_indices, replace_indices_inf)]=self.history_encoder_cav(x_train[intersection][:, 1:2, :, :, :],regroup_feature_history[intersection][:, 1:2, :, :, :])

                history_out=torch.cat((history_out_cav, history_out_inf), dim=1)
                x_train[replace_indices] = history_out 

                x_train_compensation= x_train.view(-1, 100, 252, 256)
                x_train_compensation= x_train_compensation.permute(0, 3, 1, 2)

                x_train_compensation = self.feature_compensation(x_train_compensation)
                x_train_original = x_train_compensation.permute(0, 2, 3, 1)
                x_train_original = x_train_original.view(-1, 2, 100, 252, 256)

                regroup_feature=torch.cat([x_train_original, prior_encoding], dim=4)

                fusion_features_2d = self.fusion_net(regroup_feature, regroup_feature_history,mask, spatial_correction_matrix,spatial_correction_matrix_history,history_mark,history_mask)
        # b h w c -> b c h w
                fusion_features_2d = fusion_features_2d.permute(0, 3, 1, 2)
            psm = self.cls_head(fusion_features_2d)
            rm = self.reg_head(fusion_features_2d)

        output_dict.update({'feature_before_fusion': x_train_compensation,
                        'feature': fusion_features_2d,
                       'cls_preds': psm,
                       'reg_preds': rm})
        if self.vis_feats:
            output_dict.update({'single_features':single_features})

        if self.early_distill:#true
            output_dict.update({'single_features': spatial_features_2d, 'record_len': record_len, 't_matrix': t_matrix})
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fusion_features_2d)})

        return output_dict
