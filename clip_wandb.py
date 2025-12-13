import argparse
import os
import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader
from datetime import datetime  # [新增] 用于生成时间戳

from dataset_util.PairDataset import PairDataset
from imageutil.trans import CustomRandomHorizontalFlip, CustomRandomRotation, CustomRandomVerticalFlip, CustomCenterCrop, CustomExpStretchWithOffset, CustomRandom
from models.clip_resnet18 import AstroClipModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from dataset_util.SpecDataset import SpecDataset
from root_path import ROOT_PATH
from specutil.scheduler import CosineAnnealingWithWarmupLR
import torchvision.transforms as transforms

# 导入 wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# 设置 Wandb API Key
# 从 .env 文件加载环境变量
from dotenv import load_dotenv
load_dotenv()
# 设置随机种子
pl.seed_everything(42)

# [修改] 改为接收 size 参数，以支持动态 Crop Size
class FlattenAndReshape:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        # original_shape = x.shape
        x_flat = x.flatten()
        # 使用 self.size 动态 reshape (例如 32x32x5 或 64x64x5)
        x_reshaped = x_flat.view(self.size, self.size, 5)
        x_final = x_reshaped.permute(2, 0, 1)
        return x_final

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Training Script")

    # [新增] 动态 Crop Size 参数，默认为 32
    parser.add_argument('--crop_size', type=int, default=32, help="Crop size for transforms")

    # 数据路径
    parser.add_argument('--train_data_path', type=str, default=f'../data/data_g3_z/train_dataset',
                        help="Path to the training dataset")
    parser.add_argument('--test_data_path', type=str, default=f'../data/data_g3_z/test_dataset',
                        help="Path to the test dataset")

    # 优化器超参数
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay")
    parser.add_argument('--T_max', type=int, default=10_000, help="T_max for cosine annealing scheduler")
    parser.add_argument('--T_warmup', type=int, default=1_000, help="T_warmup for cosine annealing scheduler")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--max_epochs', type=int, default=500, help="Number of epochs")
    parser.add_argument('--limit_val_batches', type=int, default=100, help="Limit validation batches")
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help="Gradient clipping value")

    # 输出路径 (作为所有实验的根目录)
    parser.add_argument('--output_dir', type=str, default=f'{ROOT_PATH}/outputs/clip/resnet_g3',
                        help="Root Output directory for checkpoints and logs")

    # 图像和光谱编码器权重路径
    parser.add_argument('--spec_weight_path', type=str,
                        default=f'../outputs/spec/spec_g3/logs/lightning_logs/version_0/checkpoints/last.ckpt',
                        help="Path to image encoder weights")
    
    # wandb 参数
    parser.add_argument('--wandb_project', type=str, default='astro-clip', help="W&B project name")
    parser.add_argument('--wandb_name', type=str, default=None, help="W&B run name")
    parser.add_argument('--wandb_offline', action='store_true', help="Run W&B in offline mode")

    return parser.parse_args()


# 训练模型
class ClipLightning(pl.LightningModule):
    def __init__(self, lr, weight_decay, T_max, T_warmup, spec_weight_path):
        super(ClipLightning, self).__init__()
        self.save_hyperparameters()

        self.model = AstroClipModel(spec_weight_path=spec_weight_path)

    def training_step(self, batch, batch_idx):
        loss_withlogit, loss_nologit, logit_scale = self.model.training_step(batch, batch_idx)
        self.log("train_loss_withlogit", loss_withlogit, on_epoch=True, prog_bar=True)
        self.log("train_loss_nologit", loss_nologit, on_epoch=True, prog_bar=True)
        self.log("scale", logit_scale)
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        val_loss_nologit, val_loss_withlogit = self.model.validation_step(batch, batch_idx)
        self.log("val_loss_nologit", val_loss_nologit, on_epoch=True, prog_bar=True)
        self.log("val_loss_withlogit", val_loss_withlogit, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.hparams.T_max,
            T_warmup=self.hparams.T_warmup,
            eta_min=self.hparams.lr / 500
        )

        return [optimizer], [scheduler]


def main():
    # 解析命令行参数
    args = parse_args()

    # ================= [逻辑修改: 生成唯一实验路径] =================
    # 获取当前时间
    current_time = datetime.now().strftime("%m%d_%H%M")
    
    # 构造实验名称 (Run Name)
    if args.wandb_name:
        run_name = f"{args.wandb_name}_{current_time}"
    else:
        # 自动命名: clip_crop32_lr0.001_1208_1830
        run_name = f"clip_crop{args.crop_size}_lr{args.lr}_{current_time}"
    
    print(f"🚀 Starting Experiment: {run_name}")

    # 构造该次实验的专属目录
    experiment_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    # ============================================================

    # 加载数据集
    train_dataset = load_from_disk(args.train_data_path)
    test_dataset = load_from_disk(args.test_data_path)

    # [修改] 应用动态 Crop Size
    transform = transforms.Compose([
        CustomCenterCrop(size=args.crop_size),   # 使用传入的 crop_size
        FlattenAndReshape(size=args.crop_size),  # 使用传入的 crop_size
    ])

    # 创建自定义数据集实例
    train_dataset = PairDataset(train_dataset, transform=transform)
    test_dataset = PairDataset(test_dataset, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 初始化模型
    model = ClipLightning(lr=args.lr, weight_decay=args.weight_decay,
                          T_max=args.T_max, T_warmup=args.T_warmup,
                          spec_weight_path=args.spec_weight_path)

    # 初始化 WandbLogger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,                # Wandb 网页显示的名称
        offline=args.wandb_offline,
        log_model=True,
        save_dir=experiment_dir,      # [关键] Wandb 日志将保存在专属文件夹内
        version=run_name              # 强制本地文件夹名与 run_name 一致
    )
    
    # 记录配置到 wandb
    wandb_logger.log_hyperparams({
        "architecture": "CLIP-ResNet18",
        "dataset": "astro-g3",
        "crop_size": args.crop_size,
        "experiment_dir": experiment_dir
    })

    # 设置训练器参数
    trainer = pl.Trainer(
        log_every_n_steps=16,
        default_root_dir=experiment_dir, # [关键] Checkpoint 默认根目录
        enable_checkpointing=True,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        limit_val_batches=args.limit_val_batches,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=os.path.join(experiment_dir, "checkpoints"), # [关键] 显式指定 ckpt 保存路径
                monitor="val_loss_nologit",
                save_top_k=2,
                save_last=True,
                every_n_epochs=1,
                mode="min",
                # [关键] 动态文件名: epoch_005-val_loss_0.1234.ckpt
                filename='epoch_{epoch:03d}-val_loss_{val_loss_nologit:.4f}',
                auto_insert_metric_name=False
            ),
        ],
        strategy='ddp',
        accelerator='gpu',
        devices=[4,5,6], # 保持原有的显卡 ID
    )

    # 使用 Trainer 进行训练
    trainer.fit(model, train_loader, test_loader)
    
    # 训练完成后关闭 wandb
    wandb.finish()

if __name__ == '__main__':
    main()
