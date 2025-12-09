import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- ResNet 基础组件 --------------------- #
class BasicBlock(nn.Module):
    """适用于 ResNet‑18 / 34 的基本残差块"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # 如果尺寸 / 通道数不一致，用 1×1 卷积对 shortcut 做对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x), inplace=True)


class BottleNeck(nn.Module):
    """适用于 ResNet‑50 及以上深度的瓶颈残差块"""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x), inplace=True)


# --------------------------- ResNet 主干 --------------------------- #
class ResNet(nn.Module):
    """自定义 ResNet 支持 5 通道输入"""

    def __init__(self, block, num_block, num_classes: int = 1024):
        super().__init__()
        self.in_channels = 64

        # 输入是 5 通道，因此 kernel_size=3 即可
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """构建一个 stage，由多个残差块组成"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x  # 形状 [B, num_classes]


# ----------------------- ResNet 工厂函数 ----------------------- #

def resnet18(num_classes: int = 1024):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 1024):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 1024):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 1024):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes: int = 1024):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)

# 特殊：输出 512 维度，用于与序列特征融合

def resnet18_fc():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=512)


# --------------------------- MLP 模块 --------------------------- #
class FCNetwork(nn.Module):
    """四层 MLP，将序列特征映射到 512 维"""
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )

    def forward(self, x):
        return self.net(x)  # [B, 512]


# --------------------------- 单头交叉注意力 --------------------------- #
class CrossAttention(nn.Module):
    """输入 Q 为图像特征，K/V 为序列特征，均形状 [B, L, D]"""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features: torch.Tensor, seq_features: torch.Tensor):
        # image_features: [B, L_q, D] (这里 L_q = 1)
        # seq_features  : [B, L_k, D] (这里 L_k = 1)
        Q = self.query(image_features)              # [B,1,D]
        K = self.key(seq_features)                  # [B,1,D]
        V = self.value(seq_features)                # [B,1,D]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # [B,1,1]
        attn   = self.softmax(scores)                                        # [B,1,1]
        context = torch.matmul(attn, V)                                      # [B,1,D]
        return context.squeeze(1)                                            # [B,D]


# --------------------------- 整体模型 --------------------------- #
class CustomResNet18(nn.Module):
    """图像 ResNet + 序列 MLP + 交叉注意力 融合"""
    def __init__(self, resnet_model: nn.Module, seq_input_size: int, num_classes: int = 1024, attention_dim: int = 512):
        super().__init__()
        self.resnet = resnet_model
        self.mlp = FCNetwork(seq_input_size)
        self.cross_attention = CrossAttention(attention_dim)
        self.fc = nn.Linear(attention_dim * 2, num_classes)

    def forward(self, image: torch.Tensor, seq: torch.Tensor):
        # 提取特征
        img_feat = self.resnet(image)                 # [B, 512]
        seq_feat = self.mlp(seq)                      # [B, 512]

        # 扩展维度，方便 batch 矩阵乘
        img_feat_exp = img_feat.unsqueeze(1)          # [B,1,512]
        seq_feat_exp = seq_feat.unsqueeze(1)          # [B,1,512]

        context = self.cross_attention(img_feat_exp, seq_feat_exp)  # [B,512]

        merged = torch.cat([img_feat, context], dim=1)  # [B,1024]
        out = self.fc(merged)                           # [B,num_classes]
        return out


def custom_resnet18(seq_input_size: int, num_classes: int = 1024, attention_dim: int = 512):
    """方便的工厂函数"""
    backbone = resnet18_fc()  # 先实例化主干
    return CustomResNet18(backbone, seq_input_size, num_classes, attention_dim)




# # --------------------------- 简单测试 --------------------------- #
# if __name__ == "__main__":
#     image = torch.randn(1, 5, 64, 64)  # 随机 5 通道图像
#     seq = torch.randn(1, 3900)         # 随机序列特征
#
#     model = custom_resnet18(3900)
#     out = model(image, seq)
#     print(out.shape)  # 预期: torch.Size([1, 1024])clip
#     # print(model)
#     # print(out)
