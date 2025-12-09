# astro_clip.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models import resnet18

# 假设以下模块已存在于你的项目中
from models.specformer import SpecFormer
from models.modules import CrossAttentionHead, MLP


# ---------- 图像编码器：改造后的 ResNet-18 ----------
class ResNet18Encoder(nn.Module):
    """
    使用 5 通道输入的 ResNet-18，输出 1024 维嵌入。
    可以从 ResNet18RedshiftPredictor 的 checkpoint 加载除 fc 之外的全部权重。
    """
    def __init__(
        self,
        image_weight_path: str = None,
        embed_dim: int = 1024,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # 1. 基础 ResNet-18
        backbone = resnet18(pretrained=pretrained_backbone)
        backbone.conv1 = nn.Conv2d(
            5, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,  # [B, 512, 1, 1]
        )
        # 2. 新的投影层：512 → 1024
        self.proj = nn.Linear(512, embed_dim)

        # 3. 选择性加载权重
        if image_weight_path is not None:
            ckpt = torch.load(image_weight_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            # 去掉最后 fc 的参数
            filtered = {
                k.replace("features.", "", 1) if k.startswith("features.") else k: v
                for k, v in state_dict.items()
                if k.startswith("features") and not k.startswith("features.fc")
            }
            missing, _ = self.load_state_dict(filtered, strict=False)
            if missing:
                print(f"[ResNet18Encoder] 未加载的参数: {missing}")

        # 4. 可选冻结
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.proj(x)         # [B, 1024]
        return F.normalize(x, dim=-1)  # 与谱特征保持一致，提前归一化


# ---------- 光谱编码器 ----------
class SpectrumHead(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights: bool = True,
    ):
        super().__init__()

        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = self._strip_prefix(ckpt["state_dict"], "model.")
        hp = ckpt["hyper_parameters"]
        for k in ["lr", "T_max", "T_warmup", "dropout", "weight_decay"]:
            hp.pop(k, None)

        self.backbone = SpecFormer(**hp)
        if load_pretrained_weights:
            self.backbone.load_state_dict(state_dict, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.cross_attn = CrossAttentionHead(
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

    @staticmethod
    def _strip_prefix(sd, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        with torch.no_grad():
            emb = self.backbone(x)["embedding"]  # [B, L, 1024]

        x, att = self.cross_attn(emb)           # [B, 1, 1024]
        x = x + self.mlp(x)                     # 残差
        x = F.normalize(x.squeeze(1), dim=-1)   # [B, 1024]

        return (x, att[1]) if return_weights else x


# ---------- CLIP 损失 ----------
class CLIPLoss(nn.Module):
    @staticmethod
    def _get_logits(i_feat, s_feat, logit_scale):
        i_feat = F.normalize(i_feat, dim=-1)
        s_feat = F.normalize(s_feat, dim=-1)
        logits_i = logit_scale * i_feat @ s_feat.T
        return logits_i, logits_i.T

    def forward(self, i_feat, s_feat, logit_scale):
        logits_i, logits_s = self._get_logits(i_feat, s_feat, logit_scale)
        labels = torch.arange(i_feat.size(0), device=i_feat.device)
        loss = (
            F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_s, labels)
        ) * 0.5
        return loss


# ---------- Astro-CLIP 主模型 ----------
class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        spec_weight_path: str,
        image_weight_path: str,
        temperature: float = 15.5,
        lr: float = 1e-4,
        weight_decay: float = 5e-2,
        epochs: int = 100,
        eta_min: float = 5e-7,
        logit_scale_init: float = 10,
        learnable_logit_scale: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- 编码器 ----
        self.image_encoder = ResNet18Encoder(
            image_weight_path,
            embed_dim=1024,
            pretrained_backbone=False,
            freeze_backbone=False,
        )
        self.spectrum_encoder = SpectrumHead(
            model_path=spec_weight_path,
            embed_dim=1024,
            freeze_backbone=True,
        )

        # logit scale
        if learnable_logit_scale:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(logit_scale_init)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(np.log(logit_scale_init))
            )

        self.criterion = CLIPLoss()

    # --------- Forward：根据类型调用 ----------
    def forward(self, x: torch.Tensor, input_type: str):
        if input_type == "image":
            return self.image_encoder(x)
        elif input_type == "spectrum":
            return self.spectrum_encoder(x)
        else:
            raise ValueError("input_type 只能是 'image' 或 'spectrum'")

    # --------- Lightning Hooks ----------
    def training_step(self, batch, _):
        img, spec = batch["image"], batch["spectrum"]
        i_feat = self.image_encoder(img)
        s_feat = self.spectrum_encoder(spec)
        loss = self.criterion(i_feat, s_feat, self.hparams.temperature)
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        img, spec = batch["image"], batch["spectrum"]
        i_feat = self.image_encoder(img)
        s_feat = self.spectrum_encoder(spec)
        loss = self.criterion(i_feat, s_feat, self.hparams.temperature)
        return loss
        # self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs, eta_min=self.hparams.eta_min
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# # ------- 按你的 ckpt 路径替换 -------
# SPEC_CKPT  = "D:/Desktop/outputs/spec/epoch=227-step=128820.ckpt"
# IMG_CKPT   = "D:/Desktop/outputs/img/version_0/checkpoints/epoch=11-step=13548.ckpt"
#
#
# model = AstroClipModel(spec_weight_path=SPEC_CKPT,
#                        image_weight_path=IMG_CKPT).eval()
#
# # ----- 造一批假数据 -----
# BATCH = 4
# img = torch.randn(BATCH, 5, 64, 64)          # 5-通道图像
# spec_len = 128                                 # 长度取决于你的光谱向量化方式
# spec = torch.randn(BATCH,3900,1)      # [B, L, 1024] 只是示例
#
# with torch.no_grad():
#     img_feat  = model(img,   "image")          # [B, 1024]
#     spec_feat = model(spec,  "spectrum")       # [B, 1024]
#     loss      = model.criterion(img_feat, spec_feat, model.hparams.temperature)
#
# print("image embedding :", img_feat.shape)
# print("spectrum embedding :", spec_feat.shape)
# print("contrastive loss :", loss.item())
