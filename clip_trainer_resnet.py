import argparse
import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader

from dataset_util.PairDataset import PairDataset
from imageutil.trans import CustomRandomHorizontalFlip, CustomRandomRotation, CustomRandomVerticalFlip, \
    CustomCenterCrop, CustomExpStretchWithOffset, CustomRandom
from models.clip_resnet18 import AstroClipModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from dataset_util.SpecDataset import SpecDataset
from root_path import ROOT_PATH
from specutil.scheduler import CosineAnnealingWithWarmupLR
import torchvision.transforms as transforms
# 设置随机种子
pl.seed_everything(42)

class FlattenAndReshape:
    def __call__(self, x):
        # 假设输入是 (5, 32, 32)
        original_shape = x.shape
        # 展平张量
        x_flat = x.flatten()  # shape: (5120,)
        # 重新reshape为 (32, 32, 5)
        x_reshaped = x_flat.view(32, 32, 5)  # shape: (32, 32, 5)
        # 转换轴为 (5, 32, 32)
        x_final = x_reshaped.permute(2, 0, 1)  # shape: (5, 32, 32)
        return x_final

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Training Script")

    # 数据路径
    parser.add_argument('--train_data_path', type=str, default=f'{ROOT_PATH}/data/data_g3/train_dataset',
                        help="Path to the training dataset")
    parser.add_argument('--test_data_path', type=str, default=f'{ROOT_PATH}/data/data_g3/test_dataset',
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

    # 输出路径
    parser.add_argument('--log_dir', type=str, default=f'{ROOT_PATH}/outputs/clip/resnet_g3/logs',
                        help="TensorBoard log directory")
    parser.add_argument('--output_dir', type=str, default=f'{ROOT_PATH}/outputs/clip/resnet_g3',
                        help="Output directory for checkpoints")

    # 图像和光谱编码器权重路径
    parser.add_argument('--spec_weight_path', type=str,
                        default=f'{ROOT_PATH}/outputs/spec/spec_g3/logs/lightning_logs/version_0/checkpoints/last.ckpt',
                        help="Path to image encoder weights")

    return parser.parse_args()


# 训练模型
class ClipLightning(pl.LightningModule):
    def __init__(self, lr, weight_decay, T_max, T_warmup, spec_weight_path):
        super(ClipLightning, self).__init__()
        self.save_hyperparameters()  # 自动保存超参数到 self.hparams

        # # 初始化模型组件，传入权重路径
        # self.image_encoder = ImageHead(model_path=image_weight_path)
        # self.spectrum_encoder = SpectrumHead(model_path=spec_weight_path)
        self.model = AstroClipModel(spec_weight_path=spec_weight_path)

    def training_step(self, batch, batch_idx):
        loss_withlogit,loss_nologit, logit_scale = self.model.training_step(batch,batch_idx)
        self.log("train_loss_withlogit", loss_withlogit, on_epoch=True, prog_bar=True)
        self.log("train_loss_nologit", loss_nologit, on_epoch=True, prog_bar=True)
        self.log("scale", logit_scale)
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        # print(f"Validation Step: batch_idx={batch_idx}")
        # print(f"Keys in batch: {list(batch.keys())}")
        val_loss_nologit, val_loss_withlogit =self.model.validation_step(batch,batch_idx)
        self.log("val_loss_nologit", val_loss_nologit, on_epoch=True, prog_bar=True)
        self.log("val_loss_withlogit", val_loss_withlogit, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,  # 从 hparams 获取 lr
            weight_decay=self.hparams.weight_decay,  # 从 hparams 获取 weight_decay
        )

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.hparams.T_max,
            T_warmup=self.hparams.T_warmup,
            eta_min=self.hparams.lr / 500  # 1/100 的学习率作为 eta_min
        )

        return [optimizer], [scheduler]


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载数据集
    train_dataset = load_from_disk(args.train_data_path)
    test_dataset = load_from_disk(args.test_data_path)

    transform = transforms.Compose([
        CustomCenterCrop(size=32),
        FlattenAndReshape(),
        # CustomRandomHorizontalFlip(p=0.4),
        # CustomRandomVerticalFlip(p=0.4),
        # CustomRandomRotation(degrees=45, resample='bilinear', p=0.4),
        # CustomExpStretchWithOffset(a = 1, b = 100),
        # CustomRandom(),
    ])

        # transforms.Normalize(mean=[0.00386281, 0.00558842, 0.0086468,  0.01220661, 0.01972715],
        #                      std=[0.4165482, 0.27164888, 0.38365453, 0.476246, 1.3442839])])

    # 创建自定义数据集实例
    train_dataset = PairDataset(train_dataset, transform=transform)
    test_dataset = PairDataset(test_dataset, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    model = ClipLightning(lr=args.lr, weight_decay=args.weight_decay,
                          T_max=args.T_max, T_warmup=args.T_warmup,
                          spec_weight_path=args.spec_weight_path)

    # 设置训练器参数
    trainer = pl.Trainer(
        log_every_n_steps=16,
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        limit_val_batches=args.limit_val_batches,
        logger=pl_loggers.TensorBoardLogger(args.log_dir),
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                monitor="val_loss_nologit",  # 监控的指标
                save_top_k=2,  # 保存最好的两个模型
                save_last=True,  # 保存最后一个模型
                every_n_epochs=1,  # 每个 epoch 保存一次模型
                mode="min"  # 'min' 表示监控 val_loss_nologit 的最小值
            ),
        ],
        strategy='ddp',
        accelerator='gpu',
        devices=6,
        #enable_progress_bar=True,  # 禁用进度条刷新
    )

    # 使用 Trainer 进行训练
    trainer.fit(model, train_loader, test_loader)
if __name__ == '__main__':
    main()
