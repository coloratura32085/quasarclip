# models/masked_autoencoder_with_redshift.py

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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    """
    ResNet 编码器：将图像编码为潜在表示
    输入: (B, 5, 32, 32)
    输出: (B, 512, 1, 1)
    """

    def __init__(self, in_channels=5, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, 2)  # 64
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)  # 128
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)  # 256
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)  # 512

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetDecoder(nn.Module):
    """
    ResNet 解码器：从潜在表示重构图像
    输入: (B, 512, 1, 1)
    输出: (B, 5, 32, 32)
    """

    def __init__(self, base_channels=64, out_channels=5):
        super().__init__()

        self.uplayer4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.uplayer3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.uplayer2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.uplayer1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 2, 2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final_up = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)
        self.final_conv = nn.Conv2d(base_channels // 2, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.uplayer4(x)
        x = self.uplayer3(x)
        x = self.uplayer2(x)
        x = self.uplayer1(x)
        x = self.final_up(x)
        return self.final_conv(x)


class RedshiftHead(nn.Module):
    """
    Few-shot 红移预测头
    从编码器特征预测红移值
    输入: (B, 512, 1, 1)
    输出: (B, 1)
    """

    def __init__(self, base_channels=64, hidden_dim=256, dropout=0.3):
        super().__init__()

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Few-shot 预测头（3层MLP + Dropout）
        self.fc = nn.Sequential(
            nn.Flatten(),

            # 第一层
            nn.Linear(base_channels * 8, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # 第二层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # 输出层
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: 编码器输出 (B, 512, H, W)
        Returns:
            redshift: (B, 1)
        """
        x = self.global_pool(x)  # (B, 512, 1, 1)
        redshift = self.fc(x)  # (B, 1)
        return redshift


class MaskedAutoEncoderWithRedshift(nn.Module):
    """
    带红移预测的掩码自编码器

    架构：
        输入 (5, 32, 32)
          ↓
        [Encoder] → 特征 (512, 1, 1)
          ↓           ↓
        [Decoder]   [Redshift Head]
          ↓           ↓
        重构 (5, 32, 32)  红移 (1,)

    联合训练：
        总损失 = α * 重构损失 + β * 红移损失
    """

    def __init__(self, in_channels=5, base_channels=64,
                 hidden_dim=256, dropout=0.3):
        super().__init__()

        # 共享编码器（提取图像特征）
        self.encoder = ResNetEncoder(in_channels, base_channels)

        # 重构解码器（图像重构任务）
        self.decoder = ResNetDecoder(base_channels, in_channels)

        # 红移预测头（下游任务）
        self.redshift_head = RedshiftHead(base_channels, hidden_dim, dropout)

    def forward(self, x, mask=None, return_features=False):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            mask: 掩码 (B, C, H, W), 1=遮盖, 0=保留
            return_features: 是否返回编码器特征
        Returns:
            reconstructed: 重构图像 (B, C, H, W)
            redshift_pred: 红移预测 (B, 1)
            [features]: 编码器特征（可选）
        """
        # 应用掩码（被遮盖的像素设为0）
        if mask is not None:
            x_masked = x * (1 - mask)
        else:
            x_masked = x

        # 编码：提取图像特征
        features = self.encoder(x_masked)

        # 解码：重构图像
        reconstructed = self.decoder(features)

        # 预测：红移值
        redshift_pred = self.redshift_head(features)

        if return_features:
            return reconstructed, redshift_pred, features
        else:
            return reconstructed, redshift_pred

    def encode(self, x, mask=None):
        """只编码，不解码（用于特征提取）"""
        if mask is not None:
            x = x * (1 - mask)
        return self.encoder(x)

    def decode(self, features):
        """只解码（用于可视化）"""
        return self.decoder(features)

    def predict_redshift(self, x, mask=None):
        """只预测红移"""
        features = self.encode(x, mask)
        return self.redshift_head(features)


class UncertaintyWeights(nn.Module):
    """
    可学习的不确定性权重
    自动平衡多任务损失

    论文: Multi-Task Learning Using Uncertainty to Weigh Losses
    https://arxiv.org/abs/1705.07115
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        # 初始化 log(σ²) 为 0，即 σ = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: list of losses [L_recon, L_redshift]
        Returns:
            weighted_loss: 加权后的总损失
            weights: 实际的权重 [w1, w2]
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            # 损失加权：1/(2σ²) * L + log(σ)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        total_loss = torch.stack(weighted_losses).sum()

        # 计算实际权重（用于监控）
        weights = torch.exp(-self.log_vars)

        return total_loss, weights


class MaskedAutoEncoderWithRedshift(nn.Module):
    """
    带红移预测的掩码自编码器
    支持固定权重和可学习权重两种模式
    """

    def __init__(self, in_channels=5, base_channels=64,
                 hidden_dim=256, dropout=0.3,
                 learnable_weights=True):
        super().__init__()

        self.encoder = ResNetEncoder(in_channels, base_channels)
        self.decoder = ResNetDecoder(base_channels, in_channels)
        self.redshift_head = RedshiftHead(base_channels, hidden_dim, dropout)

        # 可学习权重
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.uncertainty_weights = UncertaintyWeights(num_tasks=2)

    def forward(self, x, mask=None, return_features=False):
        if mask is not None:
            x_masked = x * (1 - mask)
        else:
            x_masked = x

        features = self.encoder(x_masked)
        reconstructed = self.decoder(features)
        redshift_pred = self.redshift_head(features)

        if return_features:
            return reconstructed, redshift_pred, features
        else:
            return reconstructed, redshift_pred

    def compute_weighted_loss(self, loss_recon, loss_redshift):
        """
        计算加权损失
        """
        if self.learnable_weights:
            losses = [loss_recon, loss_redshift]
            total_loss, weights = self.uncertainty_weights(losses)
            return total_loss, weights
        else:
            # 如果不使用可学习权重，返回 None
            return None, None
