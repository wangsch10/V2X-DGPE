import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
#from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.comm_modules.div2xcomm import Communication, CommunicationTopk


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


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
    def __init__(self, single_dims):
        super(DAF, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DAF fusion without  RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)
        
    def forward(self, xx):
        N, C, H, W = xx.shape
        #neighbor_feature = warp_affine_simple(split_x[b],t_matrix[i, :, :, :],(H, W)) #(N, C, H, W)
        neighbor_feature = xx #(N, C, H, W)

        # (N, C, H, W)
        ego_feature = xx[0].view(1, C, H, W).expand(N, -1, -1, -1)
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

        return fused_feature[0]
    
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


class DGFFComm(nn.Module):
    def __init__(self, single_dims, args):
        super(DGFFComm, self).__init__()
        self.fused_dims = single_dims * 2
        print('using DGFFComm fusion without RT matrix.')

        # Doamin generalized feature  encoder
        self.domain_encoder = DomainEncoder(self.fused_dims, self.fused_dims//8)
        self.spatial_encoder = SpatialEncoder2(self.fused_dims, self.fused_dims//8)
        # Dimension reduction 
        self.dim_reduction = DimReduction(self.fused_dims, single_dims)

        self.down_vehicle = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.down_inf = nn.Conv2d(single_dims, single_dims, 1, bias=False)
        self.flow_make = nn.Conv2d(self.fused_dims, 2, kernel_size=3, padding=1, bias=False)
        self.gate = SimpleGate(single_dims, 2)

        self.vis_feats = 0

        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = CommunicationTopk(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        print('communication status->{}'.format(self.communication))
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        

    def forward(self, xx, rm, record_len, pairwise_t_matrix, cav_distance, backbone=None):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        ############ 1. Split the features #######################
        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        batch_node_features = regroup(xx, record_len)
        batch_confidence_maps = regroup(rm, record_len)
        ############ 2. Communication (Mask the features) #########
        if self.communication:
            cav_distance = regroup(cav_distance, record_len)
            _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix, cav_distance)
        else:
            communication_rates = torch.tensor(0).to(xx.device)

        ############ 3. Fusion ####################################
        out = []
        for b in range(B):
            x = batch_node_features[b]
            if self.communication:
                x = x * communication_masks[b]
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

        return out, communication_rates, communication_masks
    
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


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


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
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe, batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N,C,H*W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value, confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()
        print('using where2comm')
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        print('self.communication status->{}'.format(self.communication))
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        print('self.multi_scale status->{}'.format(self.multi_scale))
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
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
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   
            elif self.agg_mode == 'Transformer':
                self.fuse_network = TransformerFusion(
                                            channels=args['agg_operator']['feature_dim'], 
                                            n_head=args['agg_operator']['n_head'], 
                                            with_spe=args['agg_operator']['with_spe'], 
                                            with_scm=args['agg_operator']['with_scm'])     

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None):
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

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

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
                        batch_confidence_maps = self.regroup(rm, record_len)
                        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks = F.max_pool2d(communication_masks, kernel_size=2)
                        x = x * communication_masks
                        
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                
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


class Where2comm2(nn.Module):
    def __init__(self, args):
        super(Where2comm2, self).__init__()
        print('using where2comm without RT matrix')
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        print('self.communication status->{}'.format(self.communication))
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        print('self.multi_scale status->{}'.format(self.multi_scale))
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
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
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   
            elif self.agg_mode == 'Transformer':
                self.fuse_modules = TransformerFusion(
                                            channels=args['agg_operator']['feature_dim'], 
                                            n_head=args['agg_operator']['n_head'], 
                                            with_spe=args['agg_operator']['with_spe'], 
                                            with_scm=args['agg_operator']['with_scm'])   
            elif self.agg_mode == 'DAF':
                self.fuse_modules = DAF(args['agg_operator']['feature_dim'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None):
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

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

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
                        batch_confidence_maps = self.regroup(rm, record_len)
                        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks = F.max_pool2d(communication_masks, kernel_size=2)
                        x = x * communication_masks
                        
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                
                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    neighbor_feature = batch_node_features[b]
                    # C, H, W = node_features.shape[1:]
                    # neighbor_feature = warp_affine_simple(node_features,
                    #                                 t_matrix[0, :, :, :],
                    #                                 (H, W))
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
                # neighbor_feature = warp_affine_simple(node_features,
                #                                 t_matrix[0, :, :, :],
                #                                 (H, W))
                x_fuse.append(self.fuse_modules(node_features))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse, communication_rates, {}
