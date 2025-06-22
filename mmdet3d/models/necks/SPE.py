import torch
import torch.nn as nn
from mmdet.models import NECKS
@NECKS.register_module()
class SpatialAttentionEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionEnhancement, self).__init__()
        # 定义子网络 Φs
        self.subnetwork = nn.Sequential(
            # 3x3 卷积层
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # 批量归一化
            nn.BatchNorm2d(in_channels),
            # ReLU 激活函数
            nn.ReLU(inplace=True),
            # 1x1 卷积层
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        # Sigmoid 函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        # 输入 BEV 特征图 F 的形状为 [B, C, H, W]
        # 通过子网络 Φs 预测注意力权重图 M
        M = self.subnetwork(F)
        # 应用 Sigmoid 函数
        M = self.sigmoid(M)
        # 计算增强后的 BEV 特征 F'
        F_prime = (1 + M) * F
        return F_prime