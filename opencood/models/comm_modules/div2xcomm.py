# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # [2, 1, 100, 252] dim1=2 represents the confidence of two anchors
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask) # [2, 1, 100, 252]

            communication_rate = communication_mask[0].sum()/(H*W)

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2] # [2, 1, 100, 252] set ego_comm_mask to all ones

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
    

class CommunicationTopk(nn.Module):
    def __init__(self, args):
        super(CommunicationTopk, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, cav_distance):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # [2, 1, 100, 252] dim1=2 represents the confidence of two anchors
            communication_maps = ori_communication_maps

            #teacher_maps = teacher_psm[b].unsqueeze(0).sigmoid().max(dim=1)[0].unsqueeze(0)# [1, 1, 100, 252] (0~1)

            if N == 1:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                communication_rate = 0
                communication_mask_nodiag = communication_mask
            else:
                ones_mask = torch.ones_like(communication_maps[0]).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps[0]).to(communication_maps.device)

                request = 1 - communication_maps[0]
                confidence = communication_maps[1]
                distance = cav_distance[b][1]
                # dynamic_thre = distance / 5000 + 0.04  #[0.01, 0.03] linear
                # dynamic_thre =  0.02 * torch.exp(0.01*distance)  #[0.02, 0.08] exp
                # dynamic_thre = 1 - teacher_maps[0]
                # overlap area guided threshold
                # generate overlap mask
                # overlap_mask = torch.ones(1, 1, H, W).to(communication_maps.device)
                # t_matrix_inf = pairwise_t_matrix[b, 0, 1, :, :].unsqueeze(0)
                # grid = F.affine_grid(t_matrix_inf, [1, 1, H, W], align_corners=True).to(communication_maps)
                # overlap_mask = F.grid_sample(overlap_mask, grid, align_corners=True)  #[1, 1, H, W]
                # dynamic_thre = (overlap_mask.sum() / (H*W)) * 0.05

                communication_mask = torch.where((request*confidence)>0.02, ones_mask, zeros_mask)
                # communication_mask = torch.where((request*confidence)>self.thre, ones_mask, zeros_mask)


                communication_rate = communication_mask[0].sum()/(H*W) #request_mask[0].sum()/(H*W)
                # communication_rate = communication_mask[0].sum()/(H*W)
                communication_mask_nodiag = torch.cat((ones_mask, communication_mask), dim=0).view(N,1,H,W)
                # communication_mask_nodiag = torch.cat((communication_maps[0], soft_comm_mask), dim=0).view(N,1,H,W)

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates