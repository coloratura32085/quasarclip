import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models.modules import CrossAttentionHead, MLP
from models.specformer import SpecFormer



class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        spec_weight_path: str,
        temperature: float = 15.5,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        eta_min: float = 5e-7,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define the spectrum encoder
        self.spectrum_encoder = SpectrumHead(model_path=spec_weight_path)

        # Use MSE loss
        self.criterion = nn.MSELoss()


    def forward(self, sp):
        """
        Defines the computation performed at every call.
        """
        return self.spectrum_encoder(sp)
    def training_step(self, batch, batch_idx):
        sp, z = batch["spectrum"], batch["z"]

        spectrum_2_z = self.spectrum_encoder(sp)

        # Calculate the loss
        loss = self.criterion(spectrum_2_z, z)

        # Log the loss
        self.log('train_loss', loss)

        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx):
        sp, z = batch["spectrum"], batch["z"]

        spectrum_2_z = self.spectrum_encoder(sp)

        # Calculate the loss
        loss = self.criterion(spectrum_2_z, z)

        # Log the loss
        self.log('val_loss', loss)

        # Return the loss
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.eta_min
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


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
        self.prediction = nn.Sequential(
            nn.Linear(4 * embed_dim, 512),  # 添加一个隐藏层，输出维度为256
            nn.ReLU(),  # 添加激活函数，例如ReLU
            nn.Linear(512, 256),
            nn.ReLU(),  # 添加激活函数，例如ReLU
            nn.Linear(256, 128),
            nn.ReLU(),  # 添加激活函数，例如ReLU
            nn.Linear(128, 64),
            nn.ReLU(),  # 添加激活函数，例如ReLU
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        x = x + self.mlp(x)

        x = self.prediction(x).suqueeze()

        return x

# model = SpectrumHead(model_path="/Users/coldplay/Desktop/epoch=227-step=128820.ckpt")
# print(model)


