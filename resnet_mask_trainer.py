# resnet_mask_trainer.py (完整版本)

import argparse
import os
import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader
from datetime import datetime

from dataset_util.PairDataset import PairDataset
from imageutil.trans import CustomSmartCrop
from models.masked_autoencoder import MaskedAutoEncoder
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from root_path import ROOT_PATH
import torchvision.transforms as transforms

# 导入 wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# 从 .env 文件加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 设置随机种子
pl.seed_everything(42)


class FlattenAndReshape:
    """将图像展平后重新整形"""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        x_flat = x.flatten()
        x_reshaped = x_flat.view(self.size, self.size, 5)
        x_final = x_reshaped.permute(2, 0, 1)
        return x_final


def parse_args():
    parser = argparse.ArgumentParser(description="Masked AutoEncoder Training Script")

    # 裁剪和掩码参数
    parser.add_argument('--crop_size', type=int, default=32,
                        help="Crop size for transforms (default: 32)")
    parser.add_argument('--core_size', type=int, default=10,
                        help="Size of quasar core region that must be included (default: 10)")
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help="Ratio of pixels to mask (default: 0.75)")

    # 数据路径
    parser.add_argument('--train_data_path', type=str,
                        default=f'../data/data_g3_z/train_dataset',
                        help="Path to the training dataset")
    parser.add_argument('--test_data_path', type=str,
                        default=f'../data/data_g3_z/test_dataset',
                        help="Path to the test dataset")

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=5,
                        help="Number of input channels (default: 5)")
    parser.add_argument('--base_channels', type=int, default=64,
                        help="Base number of channels in ResNet (default: 64)")

    # 优化器超参数
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument('--max_epochs', type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument('--limit_val_batches', type=int, default=100,
                        help="Limit validation batches (default: 100)")
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help="Gradient clipping value (default: 1.0)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loading workers (default: 4)")

    # 输出路径
    parser.add_argument('--output_dir', type=str,
                        default=f'{ROOT_PATH}/outputs/mae',
                        help="Root output directory for checkpoints and logs")

    # wandb 参数
    parser.add_argument('--wandb_project', type=str, default='astro-mae',
                        help="W&B project name")
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="W&B run name")
    parser.add_argument('--wandb_offline', action='store_true',
                        help="Run W&B in offline mode")

    # GPU 设置
    parser.add_argument('--devices', type=int, nargs='+', default=[0],
                        help="GPU device IDs (default: [0])")

    return parser.parse_args()


class MAELightning(pl.LightningModule):
    """
    PyTorch Lightning 模块：封装掩码自编码器训练逻辑
    """

    def __init__(self, in_channels=5, base_channels=64, lr=1e-3,
                 weight_decay=1e-4, mask_ratio=0.75):
        super(MAELightning, self).__init__()
        self.save_hyperparameters()

        # 初始化模型
        self.model = MaskedAutoEncoder(in_channels=in_channels,
                                       base_channels=base_channels)

        # 损失函数
        self.criterion = torch.nn.MSELoss()

        # 保存 mask_ratio
        self.mask_ratio = mask_ratio

    def generate_mask(self, images):
        """
        动态生成随机掩码
        Args:
            images: (B, C, H, W)
        Returns:
            mask: (B, C, H, W), 1=遮盖，0=保留
        """
        B, C, H, W = images.shape
        # 为每张图像生成不同的掩码
        mask = torch.rand(B, 1, H, W, device=images.device) < self.mask_ratio
        mask = mask.expand(-1, C, -1, -1).float()
        return mask

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        images = batch['image']

        # 动态生成掩码
        mask = self.generate_mask(images)

        # 前向传播
        reconstructed = self(images, mask)

        # 只在掩码区域计算损失（被遮盖的部分）
        loss_masked = self.criterion(reconstructed * mask, images * mask)

        # 同时监控全图重构损失（用于分析）
        loss_full = self.criterion(reconstructed, images)

        # 计算未被掩码区域的损失（可见部分）
        loss_visible = self.criterion(reconstructed * (1 - mask), images * (1 - mask))

        # 记录日志
        self.log('train_loss_masked', loss_masked, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('train_loss_full', loss_full, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('train_loss_visible', loss_visible, on_step=False, on_epoch=True,
                 sync_dist=True)

        return loss_masked

    def validation_step(self, batch, batch_idx):
        images = batch['image']

        # 生成掩码
        mask = self.generate_mask(images)

        # 前向传播
        reconstructed = self(images, mask)

        # 计算各种损失
        loss_masked = self.criterion(reconstructed * mask, images * mask)
        loss_full = self.criterion(reconstructed, images)
        loss_visible = self.criterion(reconstructed * (1 - mask), images * (1 - mask))

        # 记录日志
        self.log('val_loss_masked', loss_masked, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_loss_full', loss_full, on_epoch=True, sync_dist=True)
        self.log('val_loss_visible', loss_visible, on_epoch=True, sync_dist=True)

        return loss_masked

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 100
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main():
    args = parse_args()

    # 生成唯一实验路径
    current_time = datetime.now().strftime("%m%d_%H%M")

    if args.wandb_name:
        run_name = f"{args.wandb_name}_{current_time}"
    else:
        run_name = f"mae_crop{args.crop_size}_core{args.core_size}_mask{int(args.mask_ratio * 100)}_lr{args.lr}_{current_time}"

    print("=" * 70)
    print(f"🚀 Starting Experiment: {run_name}")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"  - Crop Size: {args.crop_size}x{args.crop_size}")
    print(f"  - Core Size: {args.core_size}x{args.core_size}")
    print(f"  - Mask Ratio: {args.mask_ratio}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Max Epochs: {args.max_epochs}")
    print(f"  - Devices: {args.devices}")
    print("=" * 70)

    experiment_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"📁 Experiment Directory: {experiment_dir}\n")

    # 加载数据集
    print("📦 Loading datasets...")
    train_dataset_hf = load_from_disk(args.train_data_path)
    test_dataset_hf = load_from_disk(args.test_data_path)
    print(f"  - Train dataset size: {len(train_dataset_hf)}")
    print(f"  - Test dataset size: {len(test_dataset_hf)}")

    # 定义 transform
    transform = transforms.Compose([
        CustomSmartCrop(crop_size=args.crop_size, core_size=args.core_size),
        FlattenAndReshape(size=args.crop_size),
    ])

    # 创建数据集
    train_dataset = PairDataset(train_dataset_hf, transform=transform)
    test_dataset = PairDataset(test_dataset_hf, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}\n")

    # 初始化模型
    print("🤖 Initializing model...")
    model = MAELightning(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}\n")

    # 初始化 WandbLogger
    print("📊 Initializing Wandb Logger...")
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        offline=args.wandb_offline,
        log_model=True,
        save_dir=experiment_dir,
        version=run_name
    )

    wandb_logger.log_hyperparams({
        "architecture": "Masked-AutoEncoder-ResNet",
        "dataset": "astro-g3",
        "crop_size": args.crop_size,
        "core_size": args.core_size,
        "mask_ratio": args.mask_ratio,
        "in_channels": args.in_channels,
        "base_channels": args.base_channels,
        "experiment_dir": experiment_dir
    })

    # 设置训练器
    trainer = pl.Trainer(
        log_every_n_steps=16,
        default_root_dir=experiment_dir,
        enable_checkpointing=True,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        limit_val_batches=args.limit_val_batches,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=os.path.join(experiment_dir, "checkpoints"),
                monitor="val_loss_masked",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1,
                mode="min",
                filename='epoch_{epoch:03d}-val_loss_masked_{val_loss_masked:.4f}',
                auto_insert_metric_name=False
            ),
        ],
        strategy='ddp' if len(args.devices) > 1 else 'auto',
        accelerator='gpu',
        devices=args.devices,
        precision='16-mixed',
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # 训练
    print("\n" + "=" * 70)
    print("🎯 Starting training...")
    print("=" * 70 + "\n")

    trainer.fit(model, train_loader, test_loader)

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print(f"📁 Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"📈 Best validation loss (masked): {trainer.checkpoint_callback.best_model_score:.6f}")
    print("=" * 70)

    wandb.finish()


if __name__ == '__main__':
    main()
