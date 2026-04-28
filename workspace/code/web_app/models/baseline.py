"""
Baseline 模型 - 单分支 CNN
用于快速验证流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    单分支 CNN 配准模型
    输入：SAR 影像 + 光学影像
    输出：变换参数 (平移 x, y)
    """
    
    def __init__(self, input_channels=2, feature_dim=64):
        super().__init__()
        
        # 特征提取
        self.conv1 = nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_dim * 2)
        
        self.conv3 = nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_dim * 4)
        
        self.conv4 = nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_dim * 8)
        
        # 全局平均池化 + 回归头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feature_dim * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 输出：平移 (dx, dy)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, sar, optical):
        """
        前向传播
        sar: (B, 1, H, W)
        optical: (B, 1, H, W)
        """
        # 拼接双模态输入
        x = torch.cat([sar, optical], dim=1)  # (B, 2, H, W)
        
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 全局平均池化
        x = self.gap(x).squeeze()  # (B, feature_dim*8)
        
        # 回归头
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (B, 2)
        
        return x


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    model = BaselineCNN()
    print(f"模型参数量：{count_parameters(model):,}")
    
    # 测试前向传播
    sar = torch.randn(4, 1, 512, 512)
    optical = torch.randn(4, 1, 512, 512)
    output = model(sar, optical)
    print(f"输入：SAR {sar.shape}, Optical {optical.shape}")
    print(f"输出：{output.shape}")
