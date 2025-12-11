# resnet_mask_redshift_trainer.py (完整版本，支持可学习权重)

import argparse
import os
import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader
from datetime import datetime

from dataset_util.PairDataset import PairDataset
from imageutil.trans import CustomSmartCrop
from models.masked_autoencoder_with_redshift import MaskedAutoEncoderWithRedshift
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from root_path import ROOT_PATH
import torchvision.transforms as transforms

import wandb
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv

load_dotenv()

pl.seed_everything(42)


class FlattenAndReshape:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        x_flat = x.flatten()
        x_reshaped = x_flat.view(self.size, self.size, 5)
        x_final = x_reshaped.permute(2, 0, 1)
        return x_final


def parse_args():
    parser = argparse.ArgumentParser(description="MAE with Redshift Prediction")

    # 裁剪和掩码参数
    parser.add_argument('--crop_size', type=int, default=32)
    parser.add_argument('--core_size', type=int, default=10)
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    # 数据路径
    parser.add_argument('--train_data_path', type=str,
                        default=f'../data/data_g3_z/train_dataset')
    parser.add_argument('--test_data_path', type=str,
                        default=f'../data/data_g3_z/test_dataset')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=5)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)

    # 损失权重模式
    parser.add_argument('--learnable_weights', action='store_true',
                        help="Use learnable uncertainty weights (default: False)")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Fixed weight for reconstruction loss (ignored if learnable_weights=True)")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Fixed weight for redshift loss (ignored if learnable_weights=True)")

    # 优化器超参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--limit_val_batches', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)

    # 输出路径
    parser.add_argument('--output_dir', type=str,
                        default=f'{ROOT_PATH}/outputs/mae_redshift')

    # wandb 参数
    parser.add_argument('--wandb_project', type=str, default='astro-mae-redshift')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true')

    # GPU 设置
    parser.add_argument('--devices', type=int, nargs='+', default=[0])

    return parser.parse_args()


class MAERedshiftLightning(pl.LightningModule):
    """
    支持固定权重和可学习权重的训练模块
    """

    def __init__(self, in_channels=5, base_channels=64, hidden_dim=256,
                 dropout=0.3, lr=1e-3, weight_decay=1e-4, mask_ratio=0.75,
                 learnable_weights=True, alpha=1.0, beta=1.0):
        super().__init__()
        self.save_hyperparameters()

        # 初始化模型
        self.model = MaskedAutoEncoderWithRedshift(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_weights=learnable_weights
        )

        # 损失函数
        self.recon_criterion = torch.nn.MSELoss()
        self.redshift_criterion = torch.nn.MSELoss()

        # 权重模式
        self.learnable_weights = learnable_weights
        self.alpha = alpha  # 固定权重（仅在 learnable_weights=False 时使用）
        self.beta = beta

        self.mask_ratio = mask_ratio

    def generate_mask(self, images):
        B, C, H, W = images.shape
        mask = torch.rand(B, 1, H, W, device=images.device) < self.mask_ratio
        mask = mask.expand(-1, C, -1, -1).float()
        return mask

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        redshift_true = batch['z'].unsqueeze(1).float()

        mask = self.generate_mask(images)
        reconstructed, redshift_pred = self(images, mask)

        # 计算基础损失
        loss_recon_masked = self.recon_criterion(reconstructed * mask, images * mask)
        loss_recon_full = self.recon_criterion(reconstructed, images)
        loss_recon_visible = self.recon_criterion(
            reconstructed * (1 - mask),
            images * (1 - mask)
        )
        loss_redshift = self.redshift_criterion(redshift_pred, redshift_true)

        # 计算总损失
        if self.learnable_weights:
            # 使用可学习权重
            total_loss, weights = self.model.compute_weighted_loss(
                loss_recon_masked,
                loss_redshift
            )
            weight_recon, weight_redshift = weights[0].item(), weights[1].item()
        else:
            # 使用固定权重
            total_loss = self.alpha * loss_recon_masked + self.beta * loss_redshift
            weight_recon, weight_redshift = self.alpha, self.beta

        # 计算评估指标
        with torch.no_grad():
            mae_redshift = torch.abs(redshift_pred - redshift_true).mean()
            relative_error = (
                    torch.abs(redshift_pred - redshift_true) /
                    (redshift_true + 1e-8)
            ).mean()
            ss_res = ((redshift_true - redshift_pred) ** 2).sum()
            ss_tot = ((redshift_true - redshift_true.mean()) ** 2).sum()
            r2_score = 1 - ss_res / (ss_tot + 1e-8)

        # 记录日志
        self.log('train_loss', total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('train_mae_redshift', mae_redshift, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        # 记录损失权重
        self.log('train_weight_recon', weight_recon, on_epoch=True, sync_dist=True)
        self.log('train_weight_redshift', weight_redshift, on_epoch=True, sync_dist=True)

        # 如果使用可学习权重，记录 log_vars
        if self.learnable_weights:
            self.log('train_log_var_recon',
                     self.model.uncertainty_weights.log_vars[0].item(),
                     on_epoch=True, sync_dist=True)
            self.log('train_log_var_redshift',
                     self.model.uncertainty_weights.log_vars[1].item(),
                     on_epoch=True, sync_dist=True)

        # 详细损失
        self.log('train_loss_recon_masked', loss_recon_masked, on_epoch=True, sync_dist=True)
        self.log('train_loss_recon_full', loss_recon_full, on_epoch=True, sync_dist=True)
        self.log('train_loss_recon_visible', loss_recon_visible, on_epoch=True, sync_dist=True)
        self.log('train_loss_redshift', loss_redshift, on_epoch=True, sync_dist=True)
        self.log('train_relative_error', relative_error, on_epoch=True, sync_dist=True)
        self.log('train_r2_score', r2_score, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        redshift_true = batch['z'].unsqueeze(1).float()

        mask = self.generate_mask(images)
        reconstructed, redshift_pred = self(images, mask)

        loss_recon_masked = self.recon_criterion(reconstructed * mask, images * mask)
        loss_recon_full = self.recon_criterion(reconstructed, images)
        loss_recon_visible = self.recon_criterion(
            reconstructed * (1 - mask),
            images * (1 - mask)
        )
        loss_redshift = self.redshift_criterion(redshift_pred, redshift_true)

        if self.learnable_weights:
            total_loss, weights = self.model.compute_weighted_loss(
                loss_recon_masked,
                loss_redshift
            )
            weight_recon, weight_redshift = weights[0].item(), weights[1].item()
        else:
            total_loss = self.alpha * loss_recon_masked + self.beta * loss_redshift
            weight_recon, weight_redshift = self.alpha, self.beta

        mae_redshift = torch.abs(redshift_pred - redshift_true).mean()
        relative_error = (
                torch.abs(redshift_pred - redshift_true) /
                (redshift_true + 1e-8)
        ).mean()

        ss_res = ((redshift_true - redshift_pred) ** 2).sum()
        ss_tot = ((redshift_true - redshift_true.mean()) ** 2).sum()
        r2_score = 1 - ss_res / (ss_tot + 1e-8)

        # 记录日志
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae_redshift', mae_redshift, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_weight_recon', weight_recon, on_epoch=True, sync_dist=True)
        self.log('val_weight_redshift', weight_redshift, on_epoch=True, sync_dist=True)

        if self.learnable_weights:
            self.log('val_log_var_recon',
                     self.model.uncertainty_weights.log_vars[0].item(),
                     on_epoch=True, sync_dist=True)
            self.log('val_log_var_redshift',
                     self.model.uncertainty_weights.log_vars[1].item(),
                     on_epoch=True, sync_dist=True)

        self.log('val_loss_recon_masked', loss_recon_masked, on_epoch=True, sync_dist=True)
        self.log('val_loss_recon_full', loss_recon_full, on_epoch=True, sync_dist=True)
        self.log('val_loss_recon_visible', loss_recon_visible, on_epoch=True, sync_dist=True)
        self.log('val_loss_redshift', loss_redshift, on_epoch=True, sync_dist=True)
        self.log('val_relative_error', relative_error, on_epoch=True, sync_dist=True)
        self.log('val_r2_score', r2_score, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
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

    current_time = datetime.now().strftime("%m%d_%H%M")

    if args.wandb_name:
        run_name = f"{args.wandb_name}_{current_time}"
    else:
        if args.learnable_weights:
            run_name = f"mae_z_crop{args.crop_size}_mask{int(args.mask_ratio * 100)}_learnable_{current_time}"
        else:
            run_name = (f"mae_z_crop{args.crop_size}_mask{int(args.mask_ratio * 100)}_"
                        f"a{args.alpha}_b{args.beta}_{current_time}")

    print("=" * 80)
    print(f"🚀 Starting Experiment: {run_name}")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"  Architecture:")
    print(f"    - Crop Size: {args.crop_size}x{args.crop_size}")
    print(f"    - Core Size: {args.core_size}x{args.core_size}")
    print(f"    - Base Channels: {args.base_channels}")
    print(f"    - Hidden Dim: {args.hidden_dim}")
    print(f"    - Dropout: {args.dropout}")
    print(f"  Training:")
    print(f"    - Mask Ratio: {args.mask_ratio}")
    if args.learnable_weights:
        print(f"    - Loss Weights: LEARNABLE (uncertainty-based)")
    else:
        print(f"    - Loss Weights: FIXED α={args.alpha} (recon), β={args.beta} (redshift)")
    print(f"    - Batch Size: {args.batch_size}")
    print(f"    - Learning Rate: {args.lr}")
    print(f"    - Max Epochs: {args.max_epochs}")
    print(f"  Hardware:")
    print(f"    - Devices: {args.devices}")
    print("=" * 80)

    experiment_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"📁 Experiment Directory: {experiment_dir}\n")

    # 加载数据集
    print("📦 Loading datasets...")
    train_dataset_hf = load_from_disk(args.train_data_path)
    test_dataset_hf = load_from_disk(args.test_data_path)
    print(f"  - Train dataset size: {len(train_dataset_hf)}")
    print(f"  - Test dataset size: {len(test_dataset_hf)}")

    if 'z' not in train_dataset_hf.column_names:
        raise ValueError("❌ Dataset must contain 'z' (redshift) field!")

    print(f"  ✓ Dataset contains 'z' field")

    transform = transforms.Compose([
        CustomSmartCrop(crop_size=args.crop_size, core_size=args.core_size),
        FlattenAndReshape(size=args.crop_size),
    ])

    train_dataset = PairDataset(train_dataset_hf, transform=transform)
    test_dataset = PairDataset(test_dataset_hf, transform=transform)

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
    model = MAERedshiftLightning(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        learnable_weights=args.learnable_weights,
        alpha=args.alpha,
        beta=args.beta
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    if args.learnable_weights:
        print(f"  - Using LEARNABLE uncertainty weights")
        print(f"    Initial log_var_recon: {model.model.uncertainty_weights.log_vars[0].item():.4f}")
        print(f"    Initial log_var_redshift: {model.model.uncertainty_weights.log_vars[1].item():.4f}")
    else:
        print(f"  - Using FIXED weights: α={args.alpha}, β={args.beta}")
    print()

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
        "architecture": "MAE-ResNet-Redshift",
        "dataset": "astro-g3",
        "crop_size": args.crop_size,
        "core_size": args.core_size,
        "mask_ratio": args.mask_ratio,
        "learnable_weights": args.learnable_weights,
        "alpha": args.alpha if not args.learnable_weights else "learnable",
        "beta": args.beta if not args.learnable_weights else "learnable",
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "total_params": total_params,
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
                monitor="val_mae_redshift",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1,
                mode="min",
                filename='epoch_{epoch:03d}-val_mae_{val_mae_redshift:.4f}',
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
    print("\n" + "=" * 80)
    print("🎯 Starting training...")
    print("=" * 80 + "\n")

    trainer.fit(model, train_loader, test_loader)

    print("\n" + "=" * 80)
    print("✅ Training completed!")

    if args.learnable_weights:
        final_log_vars = model.model.uncertainty_weights.log_vars
        final_weights = torch.exp(-final_log_vars)
        print(f"📊 Final learned weights:")
        print(f"    - Reconstruction: {final_weights[0].item():.4f} (log_var: {final_log_vars[0].item():.4f})")
        print(f"    - Redshift: {final_weights[1].item():.4f} (log_var: {final_log_vars[1].item():.4f})")

    print(f"📁 Checkpoints saved in: {os.path.join(experiment_dir, 'checkpoints')}")
    print("=" * 80)

    wandb.finish()


if __name__ == '__main__':
    main()
