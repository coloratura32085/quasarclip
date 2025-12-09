import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from datasets import load_from_disk
from dataset_util.PairDataset import PairDataset
from root_path import ROOT_PATH


# ------------------------- 模型定义 ------------------------- #
class ResNet18RedshiftPredictor(LightningModule):
    """
    5-通道输入 → 红移回归（标量）；
    若 pretrained=True，则加载除 conv1 外的 ImageNet 权重。
    """
    def __init__(self, pretrained: bool = True, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # ① 先构建空白骨干，替换 conv1 为 5 通道
        backbone = resnet18(weights=None)           # 不直接加载
        backbone.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)

        # ② 加载预训练权重（过滤 conv1.weight）
        if pretrained:
            imnet_sd = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1  # 官方权重
            ).state_dict()
            filtered = {k: v for k, v in imnet_sd.items()
                        if not k.startswith("conv1")}
            missing, _ = backbone.load_state_dict(filtered, strict=False)
            if self.global_rank == 0:
                print(f"[Pretrain] skipped keys: {missing}")  # conv1.weight 及 fc.*

        # ③ 组装网络
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool                                 # (B,512,1,1)
        )
        self.fc = nn.Linear(512, 1)

    # -------- 前向 -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)               # (B,) 方便计算 MSE

    # -------- Lightning Hooks -------- #
    def training_step(self, batch, _):
        y_hat = self(batch["image"])
        mae = F.l1_loss(y_hat, batch["z"])  # 训练目标
        mse = F.mse_loss(y_hat, batch["z"])  # 评估指标

        # 记录日志：MAE 用 train_loss，MSE 用 train_mse
        self.log("train_loss", mae, prog_bar=True)
        self.log("train_mse", mse, prog_bar=False)
        return mae  # 反向传播只用 MAE

    # -------- 验证 -------- #
    def validation_step(self, batch, _):
        y_hat = self(batch["image"])
        mae = F.l1_loss(y_hat, batch["z"])
        mse = F.mse_loss(y_hat, batch["z"])

        self.log("val_mae", mae, prog_bar=True)  # 也可以叫 val_loss
        self.log("val_mse", mse, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ------------------------- 训练脚本 ------------------------- #
def main():
    # 1. 读数据
    train_raw = load_from_disk(f"{ROOT_PATH}/data_g3_z/train_dataset")
    val_raw   = load_from_disk(f"{ROOT_PATH}/data_g3_z/test_dataset")
    train_ds  = PairDataset(train_raw)
    val_ds    = PairDataset(val_raw)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # 2. 模型
    model = ResNet18RedshiftPredictor(pretrained=True, lr=1e-3)

    # 3. 回调 & 日志
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min",
                              save_top_k=1, save_last=True)
    logger  = TensorBoardLogger(f"{ROOT_PATH}/outputs/image_pre/resnet18_z/logs",
                                name="resnet_z")

    # 4. Trainer
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu", devices=8, strategy="ddp",
        callbacks=[ckpt_cb],
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
