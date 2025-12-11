# 导入 wandb
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import numpy as np
import math
# 设置 Wandb API Key
# 从 .env 文件加载环境变量
from dotenv import load_dotenv

from dataset_util.PairDataset import PairDataset
from root_path import ROOT_PATH
from specutil.scheduler import CosineAnnealingWithWarmupLR

load_dotenv()


class IfeNet(nn.Module):
    """
    论文 Figure 3 的复现:
    输入: 64x64x5 (SDSS Images)
    输出: 32维特征向量
    """

    def __init__(self, input_channels=5):
        super().__init__()

        # Conv1: 64x64 -> 30x30 (Paper Table 1: k=5, s=1 -> pool k=2)
        # (64-5+1)/1 = 60 -> Pool(2) -> 30
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),  # Paper mentions CBAM, substituting with ReLU for base implementation
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv2: 30x30 -> 14x14 (Paper Table 1: k=3, s=1 -> pool k=2)
        # (30-3+1)/1 = 28 -> Pool(2) -> 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv3: 14x14 -> 6x6 (Paper Table 1: k=3, s=1 -> pool k=2)
        # (14-3+1)/1 = 12 -> Pool(2) -> 6
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flatten size calculation: 6 * 6 * 64 = 2304

        # Fully Connected Layers: 2304 -> 1028 -> 32
        self.fc = nn.Sequential(
            nn.Linear(2304, 1028),
            nn.Tanh(),  # 论文 Figure 5 的 Hidden Layers 用了 Tanh，这里 FC 也保持一致或用 ReLU
            nn.Dropout(0.5),  # 论文 Figure 3 包含 Dropout
            nn.Linear(1028, 32)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class RegNetZ(nn.Module):
    """
    论文 Figure 5 的复现: Mixture Density Network (MDN)
    输入: 融合特征 (Image 32 + Params 15 = 47)
    输出: 5个高斯分布的 参数 (mu, sigma, omega)
    """

    def __init__(self, input_dim, num_gaussians=5):
        super().__init__()
        self.num_gaussians = num_gaussians

        # Hidden Layers: Input -> 50 -> 100
        self.hidden1 = nn.Linear(input_dim, 50)
        self.hidden2 = nn.Linear(50, 100)

        # Output Layer: 100 -> 3 * num_gaussians (mu, sigma, omega)
        self.output_layer = nn.Linear(100, num_gaussians * 3)

    def forward(self, x):
        # Activation function: Tanh (Paper Figure 5)
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))

        output = self.output_layer(x)

        # Split output into parameters [cite: 187, 188, 190]
        mu = output[:, :self.num_gaussians]
        sigma = output[:, self.num_gaussians:2 * self.num_gaussians]
        omega = output[:, 2 * self.num_gaussians:]

        # Constraints:
        # Sigma must be positive: exp(sigma)
        sigma = torch.exp(sigma)

        # Omega must sum to 1: Softmax(omega) [cite: 188]
        omega = F.softmax(omega, dim=1)

        return mu, sigma, omega


class QPreNet(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4, T_max=10000, T_warmup=1000):
        super().__init__()
        self.save_hyperparameters()

        # 1. 图像特征提取 (Input: 5 channels)
        self.ife_net = IfeNet(input_channels=5)

        # 2. 融合特征长度
        # Image Features (32) + Photometric Data (15)
        self.fused_dim = 32 + 15

        # 3. 回归网络 (MDN)
        self.reg_net = RegNetZ(input_dim=self.fused_dim, num_gaussians=5)

    def forward(self, img, params):
        # 提取图像特征
        img_feat = self.ife_net(img)  # [Batch, 32]

        # 融合特征 [cite: 19] "concatenated to form fused features"
        # 确保 params 是 float32
        params = params.float()
        fused = torch.cat([img_feat, params], dim=1)  # [Batch, 47]

        # 预测 MDN 参数
        mu, sigma, omega = self.reg_net(fused)
        return mu, sigma, omega

    def mdn_loss(self, target, mu, sigma, omega):
        """
        计算负对数似然损失 (Negative Log Likelihood)
        Target: 真实红移 z
        """
        target = target.unsqueeze(1).expand_as(mu)

        # 计算高斯概率密度 N(y | mu, sigma) 的 Log 值
        # log( 1/sqrt(2pi*sigma^2) * exp(...) )
        # = -0.5*log(2pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
        log_gaussian = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((target - mu) / sigma) ** 2

        # 计算加权 Log 概率: log(omega) + log_gaussian
        log_prob = torch.log(omega) + log_gaussian

        # LogSumExp 技巧计算总概率的 Log: log( sum(exp(log_prob)) )
        log_likelihood = torch.logsumexp(log_prob, dim=1)

        # Loss = -Mean(LogLikelihood)
        return -torch.mean(log_likelihood)

    def training_step(self, batch, batch_idx):
        # 从 batch 字典中解包数据
        img = batch['image']
        params = batch['probs']
        z = batch['z']

        mu, sigma, omega = self(img, params)
        loss = self.mdn_loss(z, mu, sigma, omega)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch['image']
        params = batch['probs']
        z = batch['z']

        mu, sigma, omega = self(img, params)
        loss = self.mdn_loss(z, mu, sigma, omega)

        # 计算预测值：高斯混合模型的均值 [cite: 196]
        # Photo-z = sum(omega * mu)
        pred_z = torch.sum(omega * mu, dim=1)

        # 计算 MAE 方便观察
        mae = F.l1_loss(pred_z, z)

        # =============================================================
        # [新增] 统计 Delta z < 0.1 和 < 0.15 的比例
        # =============================================================
        # 根据论文公式：|Delta z| = |z_spec - z_photo| / (1 + z_spec) [cite: 22, 244]

        # 1. 计算归一化误差
        # 注意：分母是 (1 + z_spec)，即 (1 + z)
        normalized_error = torch.abs(z - pred_z) / (1 + z)

        # 2. 统计比例 (Accuracy)
        # 将布尔 tensor 转换为 float (True->1.0, False->0.0) 然后求均值
        acc_0_1 = (normalized_error < 0.1).float().mean()
        acc_0_15 = (normalized_error < 0.15).float().mean()

        # 3. 记录日志 (prog_bar=True 会在训练进度条直接显示)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_acc_0.1', acc_0_1, on_epoch=True, prog_bar=True)
        self.log('val_acc_0.15', acc_0_15, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # 如果你有 scheduler 模块就用，没有就用标准的
        try:
            scheduler = CosineAnnealingWithWarmupLR(
                optimizer,
                T_max=self.hparams.T_max,
                T_warmup=self.hparams.T_warmup,
                eta_min=1e-6
            )
            return [optimizer], [scheduler]
        except:
            return optimizer


# =================================================================
# 4. 主程序
# =================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Q-PreNet Training")

    # 路径配置
    parser.add_argument('--train_data_path', type=str, default=f'../../data/data_g3_z/train_dataset')
    parser.add_argument('--test_data_path', type=str, default=f'../../data/data_g3_z/test_dataset')
    parser.add_argument('--output_dir', type=str, default=f'{ROOT_PATH}/outputs/q_prenet')

    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=50)

    # Wandb
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 实验命名与路径
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = args.wandb_name if args.wandb_name else f"qprenet_sdss_15dim_{current_time}"
    experiment_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"🚀 Starting Experiment: {run_name}")
    print(f"📂 Output Dir: {experiment_dir}")

    # 2. 加载数据
    train_data = load_from_disk(args.train_data_path)
    test_data = load_from_disk(args.test_data_path)

    # 初始化 Dataset: 这里的关键是把变换参数全设为 None/False，直接用原始数据
    train_dataset = PairDataset(train_data, transform=None, extinction=False, probsTrans=False)
    test_dataset = PairDataset(test_data, transform=None, extinction=False, probsTrans=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 初始化模型
    model = QPreNet(lr=args.lr, weight_decay=1e-4)

    # 4. Logger 配置
    wandb_logger = WandbLogger(
        project="astro-qprenet",
        name=run_name,
        save_dir=experiment_dir,
        offline=args.wandb_offline,
        log_model=True,
        version=run_name
    )

    # 5. Trainer 配置
    # 你可以根据 val_acc_0.15 来保存最佳模型
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        monitor="val_acc_0.15",  # 改为监控 0.15 准确率
        save_top_k=2,
        mode="max",  # 准确率越高越好
        filename='epoch_{epoch:03d}-acc015_{val_acc_0.15:.4f}',
        save_last=True
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=[4, 5, 6],  # 你的GPU设置
        strategy="ddp",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0
    )

    # 6. 开始训练
    trainer.fit(model, train_loader, test_loader)
    wandb.finish()


if __name__ == '__main__':
    main()