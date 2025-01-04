import math
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_transformer import *
from opencood.models.sub_modules.hmsa import *
from opencood.models.sub_modules.mswin_change250100 import *
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
import torch.nn.functional as F


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, mask, spatial_correction_matrix):
        x = x.permute(0, 1, 4, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, 1:, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, 1:, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)
        x = torch.cat([x[:, 0, :, :, :].unsqueeze(1), cav_features], dim=1)
        x = x.permute(0, 1, 3, 4, 2)
        return x
class STTF_history(nn.Module):
    def __init__(self, args):
        super(STTF_history, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, mask, spatial_correction_matrix):
        x = x.permute(0, 1, 4, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)
        #x = torch.cat([x[:, 0, :, :, :].unsqueeze(1), cav_features], dim=1)
        x = cav_features.permute(0, 1, 3, 4, 2)
        return x

class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        # When t has unit of 50ms, rte_ratio=1.
        # So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(
            0).unsqueeze(1)
class RelTemporalEncodingHistory(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncodingHistory, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        # When t has unit of 50ms, rte_ratio=1.
        # So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(
            0).unsqueeze(1)

class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):
                rte_list.append(
                    self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)
class RTE_history(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE_history, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncodingHistory(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):
                rte_list.append(
                    self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)

class V2XFusionBlock(nn.Module):
    def __init__(self, num_blocks, cav_att_config, pwindow_config):
        super().__init__()
        # first multi-agent attention and then multi-window attention
        self.layers = nn.ModuleList([])
        self.num_blocks = num_blocks

        for _ in range(num_blocks):
            att = HGTCavAttention(cav_att_config['dim'],
                                  heads=cav_att_config['heads'],
                                  dim_head=cav_att_config['dim_head'],
                                  dropout=cav_att_config['dropout']) if \
                cav_att_config['use_hetero'] else \
                CavAttention(cav_att_config['dim'],
                             heads=cav_att_config['heads'],
                             dim_head=cav_att_config['dim_head'],
                             dropout=cav_att_config['dropout'])
            self.layers.append(nn.ModuleList([
                PreNorm(cav_att_config['dim'], att),
                PreNorm(cav_att_config['dim'],
                        PyramidWindowAttention(pwindow_config['dim'],
                                               heads=pwindow_config['heads'],
                                               dim_heads=pwindow_config[
                                                   'dim_head'],
                                               drop_out=pwindow_config[
                                                   'dropout'],
                                               window_size=pwindow_config[
                                                   'window_size'],
                                               relative_pos_embedding=
                                               pwindow_config[
                                                   'relative_pos_embedding'],
                                               fuse_method=pwindow_config[
                                                   'fusion_method']))]))

    def forward(self, x, mask, prior_encoding):
        for cav_attn, pwindow_attn in self.layers:
            x = cav_attn(x, mask=mask, prior_encoding=prior_encoding) + x
            x = pwindow_attn(x) + x
           # print('x.shape',x.shape)
        return x


class V2XTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']

        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']

        self.downsample_rate = args['sttf']['downsample_rate']
        self.discrete_ratio = args['sttf']['voxel_size'][0]
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF(args['sttf'])
        # adjust the channel numbers from 256+3 -> 256
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3,
                                    cav_att_config['dim'])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config),
                PreNorm(cav_att_config['dim'],
                        FeedForward(cav_att_config['dim'], mlp_dim,
                                    dropout=dropout))
            ]))

    def forward(self, x, mask, spatial_correction_matrix):

        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        x = x[..., :-3]
        ###注释掉时延补偿和空间扭曲模块satrt
        if self.use_RTE:
            # dt: (B,L)

            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)###时延
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        ######################注释end
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                  mask,
                                                                  spatial_correction_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x
class STCM(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']

        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']

        self.downsample_rate = args['sttf']['downsample_rate']
        self.discrete_ratio = args['sttf']['voxel_size'][0]
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF_history(args['sttf'])
        # adjust the channel numbers from 256+3 -> 256
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3,
                                    cav_att_config['dim'])
        self.rte = RTE_history(cav_att_config['dim'], self.RTE_ratio)


    def forward(self, x, prior_encoding, mask, spatial_correction_matrix):

        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        #prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        # if self.use_RTE:
            # dt: (B,L)
        dt = prior_encoding.to(torch.int)###时延
        x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                  mask,
                                                                  spatial_correction_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        return x


class HistoryFeatureFusionConv1d_cav(nn.Module):
    def __init__(self):
        super(HistoryFeatureFusionConv1d_cav, self).__init__()
        # 使用一个 1x1 卷积层来降维
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)


    def forward(self, bev1, bev2):
        # 将两个 BEV 特征图沿通道维度 (dim=4) 进行拼接
        fused_features = torch.cat((bev1, bev2), dim=4)


        fused_features = fused_features.view(-1, 512, 252 * 100 * 1)

        #print(fused_features.shape)
        # 使用 1x1 卷积进行降维
        reduced_features = self.conv1d(fused_features)

        #train
        #reduced_features = reduced_features.view(4, 2, 100, 252, 256)
        #validate
        reduced_features = reduced_features.view(-1, 1, 100, 252, 256)

        return reduced_features
class HistoryFeatureFusionConv1d_inf(nn.Module):
    def __init__(self):
        super(HistoryFeatureFusionConv1d_inf, self).__init__()
        # 使用一个 1x1 卷积层来降维
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)


    def forward(self, bev1, bev2):
        # 将两个 BEV 特征图沿通道维度 (dim=4) 进行拼接
        fused_features = torch.cat((bev1, bev2), dim=4)


        fused_features = fused_features.view(-1, 512, 252 * 100 * 1)

        #print(fused_features.shape)
        # 使用 1x1 卷积进行降维
        reduced_features = self.conv1d(fused_features)

        #train
        #reduced_features = reduced_features.view(4, 2, 100, 252, 256)
        #validate
        reduced_features = reduced_features.view(-1, 1, 100, 252, 256)

        return reduced_features
class HistoryFeatureFusionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(HistoryFeatureFusionResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出的通道数不同，需要1x1卷积层进行匹配
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
class ResidualBlockHistory(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(ResidualBlockHistory, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn_residual = nn.BatchNorm2d(out_channels)

    def forward(self, bev1,bev2):

        bev1 = bev1.reshape(-1, 256, 100, 252)
        bev2 = bev2.reshape(-1, 256, 100, 252)
        # 沿通道维度级联
        concat_bev = torch.cat((bev1, bev2), dim=1)

        out = self.conv1(concat_bev)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #residual = self.bn_residual(bev1)

        out += bev1
        out = F.relu(out)

        out=out.view(-1, 1, 100, 252, 256) 

        return out

class V2XTransformer(nn.Module):
    def __init__(self, args):
        super(V2XTransformer, self).__init__()

        encoder_args = args['encoder']
        #self.history_encoder=HistoryFeatureFusionConv1d()使用1*1的卷积对通道维进行降维
        #self.history_encoder_cav=HistoryFeatureFusionConv1d_cav()
        self.history_encoder_cav=ResidualBlockHistory()
        self.stcm=STCM(encoder_args)
        self.encoder = V2XTEncoder(encoder_args)

    def forward(self, x,regroup_feature_history, mask, spatial_correction_matrix,spatial_correction_matrix_history,history_mark,history_mask):


       # print('shapeeeeee',x.shape)#torch.Size([4, 2, 100, 252, 259])
        #print(regroup_feature_history)
        #对regroup_feature_history进行时延补偿和空间扭曲
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prior_encoding_history=torch.from_numpy(np.array([[1,1],[1,1],[1,1],[1,1]])).to(device)#如果batchsize为4
        regroup_feature_history=self.stcm(regroup_feature_history, prior_encoding_history,mask, spatial_correction_matrix_history)#spatial_correction_matrix_history用t时刻和t-1时刻自车的坐标计算
        prior_encoding=x[..., -3:]

        x_train=x[..., :-3]
        #history_mark决定是否参与训练,对于没有前一帧的特征，就不参加训练
        replace_indices =torch.where(history_mark == 1)[0]
        #如果历史特征没有inf，不输入history_encoder_cav和当前帧的特征进行融合
        history_out_cav=self.history_encoder_cav(x_train[replace_indices][:, 0:1, :, :, :],regroup_feature_history[replace_indices][:, 0:1, :, :, :])
        if history_mask[0,1]==1: 
            history_out_inf=self.history_encoder_cav(x_train[replace_indices][:, 1:2, :, :, :],regroup_feature_history[replace_indices][:, 1:2, :, :, :])
        else:
            history_out_inf=x_train[replace_indices][:, 1:2, :, :, :]

        ####加入时延模块，用t-1时刻的infra特征替换t时刻的infra特征，对原来完美设置下的模型进行微调
        #####################################################################################################时延测试start
        # time_delay=1
        # if time_delay==1 and history_mask[0,1]==1:
        #     history_out_inf=regroup_feature_history[replace_indices][:, 1:2, :, :, :]
        #####################################################################################################时延测试end

        #替换history_mark == False的history_out对应的batch
        #print(history_mark)
        history_out=torch.cat((history_out_cav, history_out_inf), dim=1)
        # print(history_out[torch.where(history_mark == 0)[0]]) 
        x_train[replace_indices] = history_out
        # print(history_out[0]) 
        x_out=torch.cat([x_train, prior_encoding], dim=4)
        output = self.encoder(x_out, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output
class V2XTransformerv2xvit(nn.Module):
    def __init__(self, args):
        super(V2XTransformerv2xvit, self).__init__()

        encoder_args = args['encoder']
        #self.history_encoder=HistoryFeatureFusionConv1d()使用1*1的卷积对通道维进行降维
        #self.history_encoder_cav=HistoryFeatureFusionConv1d_cav()
        self.history_encoder_cav=ResidualBlockHistory()
        self.stcm=STCM(encoder_args)
        self.encoder = V2XTEncoder(encoder_args)


    def forward(self, x,regroup_feature_history, mask, spatial_correction_matrix,spatial_correction_matrix_history,history_mark,history_mask):


    #    # print('shapeeeeee',x.shape)#torch.Size([4, 2, 100, 252, 259])
    #     batchsize=x.shape[0]
    #     #print(regroup_feature_history)
    #     #对regroup_feature_history进行时延补偿和空间扭曲
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     #prior_encoding_history=torch.from_numpy(np.array([[1,1],[1,1],[1,1],[1,1]])).to(device)#如果batchsize为4
    #     #空间扭曲模块，时延补偿模块************************************************************************start
    #     #prior_encoding_history = torch.from_numpy(np.tile([1, 1], (batchsize, 1))).to(device)
    #     #regroup_feature_history=self.stcm(regroup_feature_history, prior_encoding_history,mask, spatial_correction_matrix_history)#spatial_correction_matrix_history用t时刻和t-1时刻自车的坐标计算
    #     #空间扭曲模块，时延补偿模块*************************************************************************end

    #     prior_encoding=x[..., -3:]
    #     x_train=x[..., :-3]
    #     #history_mark决定是否参与训练,对于没有前一帧的特征，就不参加训练
    #     replace_indices =torch.where(history_mark == 1)[0]
    #     #如果历史特征没有inf，不输入history_encoder_cav和当前帧的特征进行融合
    #     history_out_cav=self.history_encoder_cav(x_train[replace_indices][:, 0:1, :, :, :],regroup_feature_history[replace_indices][:, 0:1, :, :, :])
    #     if history_mask[0,1]==1: 
    #         history_out_inf=self.history_encoder_cav(x_train[replace_indices][:, 1:2, :, :, :],regroup_feature_history[replace_indices][:, 1:2, :, :, :])
    #     else:
    #         history_out_inf=x_train[replace_indices][:, 1:2, :, :, :]

    #     ####加入时延模块，用t-1时刻的infra特征替换t时刻的infra特征，对原来完美设置下的模型进行微调
    #     #####################################################################################################时延测试start
    #     # time_delay=1
    #     # if time_delay==1 and history_mask[0,1]==1:
    #     #     history_out_inf=regroup_feature_history[replace_indices][:, 1:2, :, :, :]
    #     #####################################################################################################时延测试end

    #     #替换history_mark == False的history_out对应的batch
    #     #print(history_mark)
    #     history_out=torch.cat((history_out_cav, history_out_inf), dim=1)
    #     # print(history_out[torch.where(history_mark == 0)[0]]) 
    #     x_train[replace_indices] = history_out
    #     # print(history_out[0]) 
    #     x_out=torch.cat([x_train, prior_encoding], dim=4)
        
        output = self.encoder(x, mask, spatial_correction_matrix)
        # print(output.shape)
        

        output1 = output[:, 0]
        output2 = output

        # print(output.shape)

        return (output1,output2)

