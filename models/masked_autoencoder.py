# models/masked_autoencoder.py

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """
    ResNet 编码器：将图像编码为潜在表示

    架构：
    - 初始卷积层：降采样 + 特征提取
    - 4 个 ResNet 层：逐步提取多尺度特征
    """

    def __init__(self, in_channels=5, base_channels=64):
        super().__init__()

        # 初始卷积 (32x32 -> 16x16 -> 8x8)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 层 (逐步增加通道数，减小空间尺寸)
        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, 2)  # 8x8
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)  # 4x4
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)  # 2x2
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)  # 1x1

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetDecoder(nn.Module):
    """
    ResNet 解码器：从潜在表示重构图像

    架构：
    - 4 个上采样层：逐步恢复空间分辨率
    - 最终卷积：恢复原始通道数
    """

    def __init__(self, base_channels=64, out_channels=5):
        super().__init__()

        # 上采样层 (逐步恢复空间尺寸)
        self.uplayer4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                               kernel_size=2, stride=2),  # 1x1 -> 2x2
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.uplayer3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=2, stride=2),  # 2x2 -> 4x4
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.uplayer2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=2, stride=2),  # 4x4 -> 8x8
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.uplayer1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels,
                               kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 最终上采样和重构
        self.final_up = nn.ConvTranspose2d(base_channels, base_channels // 2,
                                           kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.final_conv = nn.Conv2d(base_channels // 2, out_channels,
                                    kernel_size=3, padding=1)

    def forward(self, x):
        x = self.uplayer4(x)
        x = self.uplayer3(x)
        x = self.uplayer2(x)
        x = self.uplayer1(x)
        x = self.final_up(x)
        x = self.final_conv(x)

        return x


class MaskedAutoEncoder(nn.Module):
    """
    掩码自编码器：学习重构被遮盖的图像区域

    训练过程：
    1. 输入图像通过掩码遮盖部分像素
    2. 编码器提取特征
    3. 解码器重构完整图像
    4. 只在被遮盖区域计算损失

    这样模型学习到图像的内在结构和上下文信息
    """

    def __init__(self, in_channels=5, base_channels=64):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, base_channels)
        self.decoder = ResNetDecoder(base_channels, in_channels)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            mask: 掩码 (B, C, H, W)，1=遮盖，0=保留
        Returns:
            重构的图像 (B, C, H, W)
        """
        # 应用掩码（被遮盖的像素设为 0）
        if mask is not None:
            x = x * (1 - mask)

        # 编码-解码
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed
