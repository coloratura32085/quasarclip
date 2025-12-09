from models.resnet_cross import custom_resnet18
from typing import Tuple
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设 image_encoder 和 util.pos_embed 模块已经正确导入
from models import image_encoder
from models.modules import CrossAttentionHead, MLP
from models.specformer import SpecFormer
from imageutil.pos_embed import interpolate_pos_embed
from root_path import ROOT_PATH

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_cross import custom_resnet18


class AstroClipModel(nn.Module):
    def __init__(self, spec_weight_path: str, seq_input_size: int, num_classes: int = 1024, attention_dim: int = 512):
        super(AstroClipModel, self).__init__()

        # 定义图像编码器，使用您的 CustomResNet18
        self.image_encoder = custom_resnet18(seq_input_size=seq_input_size, num_classes=1024,
                                             attention_dim=attention_dim)

        # 假设光谱编码器（spectrum_encoder）是通过某种方式加载的
        self.spectrum_encoder = SpectrumHead(model_path=spec_weight_path)  # 您的光谱编码器

        # Logit 缩放参数，假设为固定值（可以设置为 learnable）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))

        # CLIP 损失
        self.criterion = CLIPLoss()

    def forward(self, image: torch.Tensor, spectrum: torch.Tensor, params: torch.Tensor = None):
        # 获取图像和光谱的特征
        image_features = self.image_encoder(image, params)  # 图像特征和光谱通过交叉注意力融合
        spectrum_features = self.spectrum_encoder(spectrum)  # 光谱特征

        return image_features, spectrum_features

    def training_step(self, batch, batch_idx):
        im, sp, params = batch["image"], batch["spectrum"], batch["probs"]


        # 获取图像和光谱的特征
        image_features, spectrum_features = self(im, sp, params)

        # 计算 CLIP 损失
        loss = self.criterion(image_features, spectrum_features, self.logit_scale)

        return loss

    def validation_step(self, batch, batch_idx):
        im, sp, params = batch["image"], batch["spectrum"], batch["probs"]

        # 获取图像和光谱的特征
        image_features, spectrum_features = self(im, sp, params)

        # 计算验证损失
        val_loss = self.criterion(image_features, spectrum_features, self.logit_scale)

        return val_loss


class CLIPLoss(nn.Module):
    def get_logits(self, image_features: torch.FloatTensor, spectrum_features: torch.FloatTensor, logit_scale: float):
        # 归一化图像特征和光谱特征
        image_features = F.normalize(image_features, dim=-1)
        spectrum_features = F.normalize(spectrum_features, dim=-1)

        # 计算 logits
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        logits_per_spectrum = logits_per_image.T  # 对称的 logits

        return logits_per_image, logits_per_spectrum

    def forward(self, image_features: torch.FloatTensor, spectrum_features: torch.FloatTensor, logit_scale: float):
        # 获取 logits
        logits_per_image, logits_per_spectrum = self.get_logits(image_features, spectrum_features, logit_scale)

        # 对比损失计算
        labels = torch.arange(logits_per_image.size(0), device=image_features.device)
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_spectrum = F.cross_entropy(logits_per_spectrum, labels)

        total_loss = (loss_image + loss_spectrum) / 2
        return total_loss


class SpectrumHead(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        super().__init__()
        self.model_path = model_path
        # 加载模型的权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = self._remove_prefix(checkpoint["state_dict"], 'model.')

        hyper_parameters = checkpoint["hyper_parameters"]
        unnecessary_params = ['lr', 'T_max', 'T_warmup', 'dropout', 'weight_decay']
        for param in unnecessary_params:
            hyper_parameters.pop(param, None)
        self.backbone = SpecFormer(**hyper_parameters)
        if load_pretrained_weights:
            self.backbone.load_state_dict(state_dict, strict=False)

        # 是否冻结backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 设置交叉注意力层
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # 设置MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        """
        移除权重前缀，方便加载模型。
        """
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # 使用预训练模型编码光谱数据
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # 通过交叉注意力层
        x, attentions = self.cross_attention(embedding)

        # 通过MLP和残差连接
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()

# 初始化模型
# model = AstroClipModel(spec_weight_path = f'D:/Desktop/epoch=227-step=128820.ckpt', seq_input_size=3900)
# print(model)


