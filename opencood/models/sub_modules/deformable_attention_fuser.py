from typing import List

import math
import torch
from torch import nn
from torch.nn import functional as F
#from mmcv.cnn import xavier_init, constant_init
from mmengine.model import xavier_init, constant_init
# from mmdet3d.models.builder import FUSERS
from .deformable_attn_function import MultiScaleDeformableAttnFunction
import warnings

# __all__ = ["FlowFuser", "AttentionFuser"]


# @FUSERS.register_module()
class DeformableAttentionFuser(nn.Module):
    def __init__(self,
                 cam_embed_dims=80,
                 lidar_embed_dims=256,
                 embed_dims=256,
                 num_heads=4,
                 num_levels=1,
                 num_points=8,   
                 in_channels=[256, 256],
                 out_channels=256,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,) -> None:

        super(DeformableAttentionFuser, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)
        # self.ref_points = self.get_ref_points(H,W) # h *w, 1, 2
        self.init_weights()

        self.conv4 = nn.Conv2d(sum(in_channels), out_channels, 1,  bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(True)

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_ref_points(self, H, W):
        dtype = torch.float32
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.squeeze(0)
        return ref_2d

    def forward(self, inputs: List[torch.Tensor],) -> torch.Tensor:
         #b, l, h, w, c为v2xvit多窗口注意力机制的输入，先试试融合的，融合cav和infra的特征，不像v2xvit的多窗口注意力机制那样分开处理
         #如果要像v2xvit那样处理，可以把b和l维度放到一起
        inputs_b, inputs_l, inputs_h, inputs_w, inputs_c = inputs.shape
        inputs= inputs.view(inputs_b * inputs_l, inputs_h, inputs_w, inputs_c)

        # cam_bev = inputs[:, 0, :, :, :]  # b,c,h,w    cam为cav，lidar为infra
        # lidar_bev =inputs[:, 1, :, :, :] # b,c,h,w
        cam_bev=inputs.permute(0, 3, 1, 2)
        lidar_bev=inputs.permute(0, 3, 1, 2)

        bs = cam_bev.shape[0]
        h = cam_bev.shape[2]
        w = cam_bev.shape[3]
        reference_points= self.get_ref_points(h, w).unsqueeze(
            0).repeat(bs, 1, 1).to(cam_bev.device)
        # ref_point -> bs, h*w, 1, 2

        query = lidar_bev.reshape(bs, h*w, -1)
        value = cam_bev.reshape(bs, h*w, -1)
        # query: lidar_bev -> b, h*w, c
        # value: cam_bev -> b, h*w, num_head, c / num_head

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        spatial_shapes = torch.tensor([[h, w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, None, None, :] \
                + sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)

        output = self.output_proj(output)

        # value_shape -> shape 1, 2  [180,180]
        # lvl_star_idx -> 0 gpu tensor[0]
        # im2col_step -> 64 / 32

        # sampling_location -> (bs ,num_queries, num_heads, num_levels, num_points, 2)
        # attention_weights -> (bs ,num_queries, num_heads, num_levels, num_points)

        output = output.view(inputs_b, inputs_l, 256, h, w)
        output=output.permute(0, 1,3,4, 2)
        # out [num_query, bs, embed_dims]
        #out_bev = torch.cat([output, inputs[1]], dim=1)
        #out_bev = self.relu4(self.bn4(output))

        # fuser
        # out_proj  + lidar_bev
        #   concat conv
        # + lidar_bev

        return output
