import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from dataset_util.PairDataset import PairDataset  # 你自定义的路径
from datasets import load_from_disk
from torchvision import transforms
import argparse

from root_path import ROOT_PATH


# ====== ViT 模型定义（回归） ======
class ViTRegressor(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, img_size = 144)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.vit(x)


# ====== Lightning 模块定义 ======
class LitViT(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = ViTRegressor()
        self.lr = lr
        self.loss_fn = nn.L1Loss()  # MAE
        self.mse_fn = nn.MSELoss()  # Metric

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["z"]
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["z"]
        y_hat = self(x).squeeze()
        val_loss = self.loss_fn(y_hat, y)
        val_mse = self.mse_fn(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mse", val_mse)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ====== 主训练入口 ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default = f"F:/database/filterWave/data_rgb_z/train_dataset")
    parser.add_argument("--test_data_path", type=str, default= f"F:/database/filterWave/data_rgb_z/test_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    # 加载 datasets.Dataset
    raw_train_dataset = load_from_disk(args.train_data_path)
    raw_test_dataset = load_from_disk(args.test_data_path)


    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop((144, 144)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        transforms.RandomRotation(degrees=10),  # 旋转 ±10 度
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # 自定义 PairDataset 包装
    train_dataset = PairDataset(raw_train_dataset, transform=transform)
    test_dataset = PairDataset(raw_test_dataset, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化模型、日志器和回调
    model = LitViT(lr=args.lr)
    logger = TensorBoardLogger(f"{ROOT_PATH}/outputs/rgb_vit_regression/logs", name="vit_regression")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        save_weights_only=False,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu",  # 明确使用 GPU
        devices=-1,  # 自动使用所有可用 GPU
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
