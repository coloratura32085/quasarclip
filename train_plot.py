# train_astro_clip.py
"""
Astro-CLIP Lightning 训练脚本
增加特征可视化功能
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
    # CustomCenterCrop,
    # CustomRandomHorizontalFlip,
    # CustomRandomVerticalFlip,
    # CustomRandomRotation,
    # CustomExpStretchWithOffset,
    # CustomRandom,
    PerChannelMinMaxNorm,
)
from specutil.scheduler import CosineAnnealingWithWarmupLR
from models.clip_resnet_pretrain import AstroClipModel
from root_path import ROOT_PATH

# 新增的可视化功能依赖
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

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
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--T_max", type=int, default=10_000)
    parser.add_argument("--T_warmup", type=int, default=1_000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=int, default=100)

    # 权重
    parser.add_argument("--spec_weight_path", type=str,
                        default=f"{ROOT_PATH}/outputs/spec/epoch=227-step=128820.ckpt")
    parser.add_argument("--image_weight_path", type=str,
                        default=f"{ROOT_PATH}/outputs/img/version_0/checkpoints/epoch=11-step=13548.ckpt")

    # 输出
    parser.add_argument("--output_dir", type=str, default=f"{ROOT_PATH}/outputs/clip/resnet_pretrain")

    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="训练后执行特征可视化")
    parser.add_argument("--viz_num_samples", type=int, default=500, help="可视化样本数量")
    parser.add_argument("--viz_num_lines", type=int, default=20, help="可视化连线数量")

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


# ---------------- 特征可视化函数 ----------------

def visualize_features(model, dataloader, output_dir, num_samples=500, num_lines=20):
    """
    使用t-SNE可视化图像和光谱特征
    参数:
    model (AstroClipModel): 训练好的模型
    dataloader (DataLoader): 数据加载器
    output_dir (Path): 输出目录
    num_samples (int): 要可视化的样本数
    num_lines (int): 要画的对应点连线的数量
    """
    device = next(model.parameters()).device

    # 收集特征和标签
    image_features = []
    spectrum_features = []

    model.eval()
    print(f"正在收集{num_samples}个样本的特征...")
    with torch.no_grad():
        # 使用tqdm显示进度条
        for batch in tqdm(dataloader):
            if len(image_features) >= num_samples:
                break

            images = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)

            # 获取特征
            img_feat = model.image_encoder(images)
            spec_feat = model.spectrum_encoder(spectra)

            # 保留剩余所需样本数
            remaining = num_samples - len(image_features)
            img_feat = img_feat[:remaining]
            spec_feat = spec_feat[:remaining]

            image_features.append(img_feat.cpu())
            spectrum_features.append(spec_feat.cpu())

    # 合并所有特征
    image_features = torch.cat(image_features).numpy()
    spectrum_features = torch.cat(spectrum_features).numpy()

    print(f"收集完成: 图像特征 {image_features.shape}, 光谱特征 {spectrum_features.shape}")

    # 合并所有特征用于t-SNE
    all_features = np.concatenate([image_features, spectrum_features], axis=0)

    # 运行t-SNE降维
    print("运行t-SNE降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, num_samples - 1),  # perplexity不能大于样本数-1
        random_state=42,
        verbose=1
    )
    tsne_results = tsne.fit_transform(all_features)

    # 分离降维后的特征
    tsne_image = tsne_results[:num_samples]
    tsne_spectrum = tsne_results[num_samples:]

    print("创建可视化图表...")
    # 设置好看的绘图样式
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    plt.figure(figsize=(14, 10))

    # 创建散点图 - 图像特征
    img_scatter = plt.scatter(
        tsne_image[:, 0], tsne_image[:, 1],
        color='dodgerblue', alpha=0.7,
        s=60, edgecolor='w', linewidth=0.5,
        label='图像特征'
    )

    # 创建散点图 - 光谱特征
    spec_scatter = plt.scatter(
        tsne_spectrum[:, 0], tsne_spectrum[:, 1],
        color='tomato', alpha=0.7,
        s=60, edgecolor='w', linewidth=0.5,
        label='光谱特征'
    )

    # 添加对应点之间的连线（随机选取）
    if num_lines > 0:
        indices = np.random.choice(num_samples, min(num_lines, num_samples), replace=False)
        for i in indices:
            plt.plot(
                [tsne_image[i, 0], tsne_spectrum[i, 0]],
                [tsne_image[i, 1], tsne_spectrum[i, 1]],
                'k-', alpha=0.3, linewidth=1
            )

    # 添加标题和标签
    plt.title(f'图像和光谱特征分布（{num_samples}个样本）', fontsize=18, pad=15)
    plt.xlabel('t-SNE维度1', fontsize=14)
    plt.ylabel('t-SNE维度2', fontsize=14)

    # 添加图例
    plt.legend(loc='best', fontsize=12)

    # 添加网格
    plt.grid(alpha=0.3)

    # 添加解释性文本
    plt.figtext(0.5, 0.01,
                f"每个蓝点代表一个图像特征，红点代表对应的光谱特征，黑线连接同一对象的不同模态特征",
                ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.tight_layout()

    # 保存图像
    viz_path = output_dir / "feature_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ 特征可视化图已保存至: {viz_path}")

    # 显示图像
    plt.show()

    return tsne_image, tsne_spectrum


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

    train_ds = PairDataset(train_raw, transform=transform, rgb=False)
    val_ds = PairDataset(val_raw, transform=transform, rgb=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ----------- 模型 & Trainer -----------
    model = AstroClipLightning(
        args.spec_weight_path, args.image_weight_path, args.lr, args.weight_decay, args.T_max, args.T_warmup
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # ----------- 训练模型 -----------
    trainer.fit(model, train_loader, val_loader)

    # ----------- 特征可视化 -----------
    if args.visualize and trainer.is_global_zero:
        print("=" * 60)
        print("开始特征可视化...")

        # 加载最佳模型
        if trainer.checkpoint_callback.best_model_path:
            print(f"加载最佳模型: {trainer.checkpoint_callback.best_model_path}")
            model = AstroClipLightning.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                spec_weight_path=args.spec_weight_path,
                image_weight_path=args.image_weight_path,
                lr=args.lr,
                weight_decay=args.weight_decay,
                T_max=args.T_max,
                T_warmup=args.T_warmup
            )
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            print("⚠️ 没有找到最佳模型，使用当前模型")
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 执行可视化
        visualize_features(
            model.model,
            val_loader,
            output_dir,
            num_samples=args.viz_num_samples,
            num_lines=args.viz_num_lines
        )


if __name__ == "__main__":
    main()