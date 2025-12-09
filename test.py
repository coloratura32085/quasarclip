# # import torch
# # import torch.nn.functional as F
# # import numpy as np
# #
# #
# # def preprocess(x, slice_section_length=20, slice_overlap=10):
# #     # 计算标准差和均值
# #     std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
# #     print("Shape after std and mean calculation:", x.shape)
# #
# #     # 标准化
# #     x = (x - mean) / std
# #     print("Shape after normalization:", x.shape)
# #
# #     # 切片操作
# #     x = _slice(x, slice_section_length, slice_overlap)
# #     print("Shape after slice:", x.shape)
# #
# #     # 填充
# #     x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
# #     print("Shape after padding:", x.shape)
# #
# #     # 修改特定位置的值
# #     x[:, 0, 0] = ((mean.squeeze() - 2) / 2)
# #     x[:, 0, 1] = ((std.squeeze() - 2) / 8)
# #
# #     return x
# #
# #
# # def _slice(x, slice_section_length, slice_overlap):
# #     """Slice input tensor into smaller sections based on the given lengths."""
# #     start_indices = np.arange(
# #         0,
# #         x.shape[1] - slice_overlap,
# #         slice_section_length - slice_overlap,
# #     )
# #
# #     sections = [
# #         x[:, start:start + slice_section_length].transpose(1, 2)
# #         for start in start_indices
# #     ]
# #
# #     # 如果最后一个部分不够长，可以选择丢弃它
# #     if sections[-1].shape[1] < slice_section_length:
# #         sections.pop(-1)  # 丢弃最后一个部分
# #
# #     return torch.cat(sections, 1)
# #
# #
# # # 测试
# # x = torch.randn(10, 7781, 1)  # 创建形状为 (10, 4501, 1) 的输入
# # output = preprocess(x, slice_section_length=20, slice_overlap=10)
# #
# # print("Final output shape:", output.shape)
# import torch
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
# from lightning import LightningModule   # PyTorch-Lightning ≥2.2
#
# class ResNet18RedshiftPredictor(LightningModule):
#     def __init__(self,
#                  weights: ResNet18_Weights.IMAGENET1K_V1,
#                  learning_rate: float = 1e-3):
#         super().__init__()
#
#         # --- ① 先构建“空白”ResNet-18 结构 -----------------------------
#         backbone = resnet18(weights=None)          # 不直接加载权重
#         backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7,
#                                    stride=2, padding=3, bias=False)  # 5-通道
#
#         # --- ② 把除 conv1 以外的预训练权重灌进来 -----------------------
#         if weights is not None:
#             pretrained_sd = resnet18(weights=weights).state_dict()   # 官方权重
#             # 过滤掉 conv1.weight（形状不匹配）
#             filtered = {k: v for k, v in pretrained_sd.items()
#                         if not k.startswith('conv1')}
#             # strict=False → 允许缺少 conv1
#             missing, unexpected = backbone.load_state_dict(filtered, strict=False)
#             if self.global_rank == 0:   # 多卡时只在 rank 0 打印
#                 print(f'[ResNet18] missing keys: {missing}')         # conv1.weight
#                 print(f'[ResNet18] unexpected keys: {unexpected}')   # []
#
#         # --- ③ 搭建后续层 ------------------------------------------------
#         self.features = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             backbone.relu,
#             backbone.maxpool,
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4,
#             backbone.avgpool,      # 输出 (B, 512, 1, 1)
#         )
#         self.fc = nn.Linear(512, 1)      # 红移回归
#
#         # 超参数
#         self.learning_rate = learning_rate
#         self.save_hyperparameters()
#
#     # --------------------------------------------------------------
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = torch.flatten(x, 1)          # (B, 512)
#         return self.fc(x)                # (B, 1)
#
#
#
#
# net = ResNet18RedshiftPredictor(weights=ResNet18_Weights.IMAGENET1K_V1)   # 不加载预训练，先验证前向
# x   = torch.randn(2, 5, 64, 64)     # (B=2, C=5, H=W=64)
# y   = net(x)
# print('输出形状:', y.shape)
#
# print(y)# 预期 (2, 1)
