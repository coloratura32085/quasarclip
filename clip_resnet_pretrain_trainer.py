# train_astro_clip.py
"""
Astro-CLIP Lightning 训练脚本
------------------------------------------------
数据增强支持 **PerChannelMinMaxNorm**（已移动到 `imageutil.trans` 模块）。
"""
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# ---------- 项目内依赖 ----------
from dataset_util.PairDataset import PairDataset
from imageutil.trans import (
    CustomCenterCrop,
    CustomRandomHorizontalFlip,
    CustomRandomVerticalFlip,
    CustomRandomRotation,
    CustomExpStretchWithOffset,
    CustomRandom,
    PerChannelMinMaxNorm,  # <-- 已从 trans 模块导入
)
from specutil.scheduler import CosineAnnealingWithWarmupLR
from models.clip_resnet_pretrain import AstroClipModel
from root_path import ROOT_PATH

# ---------------- 全局随机种子 ----------------
pl.seed_everything(55)

# ---------------- 参数解析 ----------------

def parse_args():
    parser = argparse.ArgumentParser(description="Astro-CLIP Lightning Training Script")

    # 数据路径
    parser.add_argument("--train_data_path", type=str, default=f"F:/database/filterWave/data_g3_z/train_dataset")
    parser.add_argument("--val_data_path", type=str, default=f"F:/database/filterWave/data_g3_z/test_dataset")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--T_max", type=int, default=10_000)
    parser.add_argument("--T_warmup", type=int, default=1_000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=int, default=100)

    # 权重
    parser.add_argument("--spec_weight_path", type=str,
                        default=f"{ROOT_PATH}/outputs/spec/epoch=227-step=128820.ckpt")
    parser.add_argument("--image_weight_path", type=str, default=f"{ROOT_PATH}/outputs/img/version_0/checkpoints/epoch=11-step=13548.ckpt")

    # 输出
    parser.add_argument("--output_dir", type=str, default=f"{ROOT_PATH}/outputs/clip/resnet_pretrain")

    return parser.parse_args()


# ---------------- LightningModule ----------------

class AstroClipLightning(pl.LightningModule):
    def __init__(self, spec_weight_path, image_weight_path, lr, weight_decay, T_max, T_warmup):
        super().__init__()
        self.save_hyperparameters()
        self.model = AstroClipModel(spec_weight_path=spec_weight_path, image_weight_path=image_weight_path)

    def training_step(self, batch, _):
        loss = self.model.training_step(batch, None)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.model.validation_step(batch, None)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sch = CosineAnnealingWithWarmupLR(
            opt, T_max=self.hparams.T_max, T_warmup=self.hparams.T_warmup, eta_min=self.hparams.lr / 500
        )
        return {"optimizer": opt, "lr_scheduler": sch}


# ---------------- 主函数 ----------------

def main():
    args = parse_args()

    # ----------- 数据集加载 -----------
    train_raw = load_from_disk(args.train_data_path)
    val_raw = load_from_disk(args.val_data_path)

    # ----------- 数据增强流水线 -----------
    transform = transforms.Compose([
        # CustomCenterCrop(size=32),
        # # 可选几何增强：
        # # CustomRandomHorizontalFlip(p=0.4),
        # # CustomRandomVerticalFlip(p=0.4),
        # # CustomRandomRotation(degrees=45, resample="bilinear", p=0.4),
        # # CustomExpStretchWithOffset(a=1, b=100),
        # # CustomRandom(),
        PerChannelMinMaxNorm(),  # <-- 每通道 Min‑Max 归一化
    ])

    train_ds = PairDataset(train_raw, transform=transform)
    val_ds = PairDataset(val_raw, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ----------- 模型 & Trainer -----------
    model = AstroClipLightning(
        args.spec_weight_path, args.image_weight_path, args.lr, args.weight_decay, args.T_max, args.T_warmup
    )

    output_dir = Path(args.output_dir)
    tb_logger = pl_loggers.TensorBoardLogger(output_dir / "logs")

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clip,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=16,
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=2, save_last=True,
                filename="epoch={epoch}-step={step}-val_loss={val_loss:.4f}"
            ),
        ],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
