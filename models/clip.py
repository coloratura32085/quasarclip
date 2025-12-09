# 导入必要的模块
import torch
from torch import nn


import os
import sys
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


class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        # image_encoder: nn.Module,
        # spectrum_encoder: nn.Module,
        image_weight_path :str,
        spec_weight_path : str,
        temperature: float = 15.5,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        eta_min: float = 5e-7,
        logit_scale: float = 15.5,
        learnable_logit_scale: bool = False,
    ):
        """
        The AstroCLIP model that takes an image and a spectrum and embeds them into a common space using CLIP loss.
        Note that you must provide the image and spectrum encoders to be used for the embedding.

        Args:
            image_encoder (nn.Module): The image encoder to be used for embedding.
            spectrum_encoder (nn.Module): The spectrum encoder to be used for embedding.
            temperature (float): The temperature parameter for the CLIP loss.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            epochs (int): The number of epochs for training.
            eta_min (float): The minimum learning rate for the scheduler.
            logit_scale (float): The logit scale for the CLIP loss.
            learnable_logit_scale (bool): Whether the logit scale should be learnable.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define the image and spectrum encoder
        self.image_encoder = ImageHead(model_path=image_weight_path)
        self.spectrum_encoder = SpectrumHead(model_path=spec_weight_path)

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    def forward(
        self,
        input: torch.Tensor,
        input_type: str,
    ):
        if input_type == "image":
            return self.image_encoder(input)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def training_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]

        # Get the image and spectrum features
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        # Calculate the CLIP loss
        loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )

        # Log the losses
        # self.log("train_loss_withlogit", loss_withlogit)
        # self.log("train_loss_nologit", loss_nologit)
        # self.log("scale", self.logit_scale)

        # Return the loss
        return loss_withlogit, loss_nologit,self.logit_scale

    def validation_step(self, batch, batch_idx):
        # print(f"Batch Index: {batch_idx}")
        # print(f"Batch: {batch}")
        # print(f"Keys in batch: {list(batch.keys())}")

        im, sp = batch["image"], batch["spectrum"]

        # Get the image and spectrum features
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        # Calculate the CLIP loss
        val_loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        return  val_loss_nologit,  val_loss_withlogit

        # Log the losses
        # self.log("val_loss_nologit", val_loss_nologit)
        # self.log("val_loss_withlogit", val_loss_withlogit)


class CLIPLoss(nn.Module):
    def get_logits(
        self,
        image_features: torch.FloatTensor,
        spectrum_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(
        self,
        image_features: torch.FloatTensor,
        spectrum_features: torch.FloatTensor,
        logit_scale: float,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the image and spectrum features
        logits_per_image, logits_per_spectrum = self.get_logits(
            image_features, spectrum_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss

class ImageHead(nn.Module):
    def __init__(self,
                 model_path: str,
                 embed_dim: int = 1024,
                 n_head: int = 4,
                 model_embed_dim: int = 192,
                 dropout: float = 0.1,
                 freeze_backbone: bool = True,
                 ):
        super().__init__()

        self.model_path = model_path
        self.model = image_encoder.__dict__["mae_vitae_base_patch8_enc"]()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        print(f"Load pre-trained checkpoint from: {self.model_path}")
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()

        # 移除形状不匹配的权重
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 插值位置嵌入
        interpolate_pos_embed(self.model, checkpoint_model)

        # 加载预训练模型
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # 设置交叉注意力
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # 通过骨干网络
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.model(x)
        # print("embedding shape:", embedding.shape)

        # 通过交叉注意力
        x, attentions = self.cross_attention(embedding)

        # 通过 MLP 和残差连接
        x = self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()




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
        # Load the model from the checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = self._remove_prefix(checkpoint["state_dict"], 'model.')
        self.backbone = SpecFormer(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(state_dict)

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
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
        # Embed the spectrum using the pretrained model
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]
            # print("embedding shape:", embedding.shape)

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)
        # print("attentions shape:", attentions.shape)
        # print("x shape:", x.shape)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


