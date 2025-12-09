from models.param_net import photo_net
from typing import Tuple
from typing import Union
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


class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        spec_weight_path : str,
        temperature: float = 15.5,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        eta_min: float = 5e-7,
        logit_scale: float = 10,
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
        self.photo_encoder = photo_net
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
        if input_type == "params":
            return self.photo_encoder(input)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def training_step(self, batch, batch_idx):
        ph, sp = batch["params"], batch["spectrum"]

        # print(f"logit_scale grad: {self.logit_scale.grad}")
        # Get the image and spectrum features
        photo_features = self.photo_encoder(ph)
        # print("image_features :", image_features.shape)
        spectrum_features = self.spectrum_encoder(sp)
        # print("spectrum_features :", spectrum_features.shape)

        # Calculate the CLIP loss
        loss_withlogit = self.criterion(
            photo_features, spectrum_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            photo_features, spectrum_features, self.hparams.logit_scale
        )

        # Log the losses
        # self.log("train_loss_withlogit", loss_withlogit)
        # self.log("train_loss_nologit", loss_nologit)
        # self.log("scale", self.logit_scale)

        # Return the loss
        return loss_withlogit, loss_nologit, self.logit_scale

    def validation_step(self, batch, batch_idx):
        # print(f"Batch Index: {batch_idx}")
        # print(f"Batch: {batch}")
        # print(f"Keys in batch: {list(batch.keys())}")

        ph, sp = batch["image"], batch["spectrum"]

        # Get the image and spectrum features
        photo_features = self.photo_encoder(ph)
        # print("image_features :", image_features.shape)
        # print("image_feature_before", image_features)

        spectrum_features = self.spectrum_encoder(sp)
        # print("spectrum_features :", spectrum_features.shape)
        # print("spectrum_feature_before", spectrum_features)

        # Calculate the CLIP loss
        val_loss_nologit = self.criterion(
            photo_features, spectrum_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            photo_features, spectrum_features, self.hparams.temperature
        )
        return  val_loss_nologit,  val_loss_withlogit

        # Log the losses
        # self.log("val_loss_nologit", val_loss_nologit)
        # self.log("val_loss_withlogit", val_loss_withlogit)





class CLIPLoss(nn.Module):
    """
    仅计算均方误差 (MSE)。
    output_dict=True 时返回 {'mse_loss': loss_tensor}，
    否则直接返回张量。
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_features: torch.Tensor,
        spectrum_features: torch.Tensor,
        *_,                 # 兼容旧签名，忽略多余参数
        output_dict: bool = False,
        **__,
    ) -> Union[torch.Tensor, dict]:
        loss = F.mse_loss(image_features, spectrum_features)
        return {"mse_loss": loss} if output_dict else loss


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

        hyper_parameters = checkpoint["hyper_parameters"]
        # print(hyper_parameters)

        # 删除不需要的超参数
        for i in ['lr', 'T_max', 'T_warmup', 'dropout','weight_decay']:
            hyper_parameters.pop(i, None)
        # 传递剩余的超参数给 SpecFormer
        # self.backbone = SpecFormer(hyper_parameters)

        print(checkpoint["hyper_parameters"])
        self.backbone = SpecFormer(**hyper_parameters)
        if load_pretrained_weights:
            self.backbone.load_state_dict(state_dict, strict=False)

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

        # 添加额外的全连接层
        self.additional_layers = nn.Sequential(
            nn.Linear(1024, 512),  # 第一层：将维度减半
            nn.ReLU(),  # 激活函数
            # nn.Dropout(dropout),  # Dropout 防止过拟合
            nn.Linear(512, 256),  # 第二层：进一步减少维度
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(256, 128)  # 输出层：最终维度
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
        # print('after_attention : ', x)
        x = x + self.mlp(x)
        print('after_mlp : ', x)

        x = self.additional_layers(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        print(x.squeeze().shape)



        return x.squeeze()

# 模拟输入数据
batch_size = 2  # 批量大小
param_dim = 120  # 假设 params 的维度为 512
spectrum_length = 1024  # 假设 spectrum 的长度为 1024

# 随机生成 params 和 spectrum 数据
params = torch.randn(batch_size, param_dim)  # (batch_size, param_dim)
spectrum = torch.randn(batch_size, 3900, 1)  # (batch_size, spectrum_length)

# 创建模型实例
spec_weight_path = "D:/Desktop/epoch=227-step=128820.ckpt"
model = AstroClipModel(spec_weight_path)

# 测试前向传播
with torch.no_grad():  # 关闭梯度计算以加速测试
    # 测试 params 输入
    params_output = model(params, input_type="params")
    print(f"Params output shape: {params_output.shape}")

    # 测试 spectrum 输入
    spectrum_output = model(spectrum, input_type="spectrum")
    print(f"Spectrum output shape: {spectrum_output.shape}")


