import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        print(bandwidth_list)
        kernel_val = [torch.exp(-L2_distance / (2*bandwidth_temp**2)) for bandwidth_temp in bandwidth_list]
        print(kernel_val)
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        
        kernels = self.gaussian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss

# # 使用方法
# def compute_mmd_loss(feature_lidar_1, feature_lidar_2):
#     feature_lidar_1 = feature_lidar_1.view(feature_lidar_1.size(0), -1)
#     feature_lidar_2 = feature_lidar_2.view(feature_lidar_2.size(0), -1)
    
#     mmd_loss_fn = MMDLoss()
#     loss = mmd_loss_fn(feature_lidar_1, feature_lidar_2)
#     return loss

# # 示例输入特征图
# batch_size, channels, height, width = 4, 3, 32, 32
# feature_lidar_1 = torch.rand(batch_size, channels, height, width)
# feature_lidar_2 = torch.rand(batch_size, channels, height, width)

# # 计算MMD损失
# loss = compute_mmd_loss(feature_lidar_1, feature_lidar_2)
# print('MMD Loss:', loss.item())
