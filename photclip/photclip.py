import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from datasets import load_from_disk
import argparse

from dataset_util.PairDataset import PairDataset  # 你自己的 Dataset：返回 {'spectrum': Tensor, 'z': Tensor}


# ============================================================
#  Attention Block：固定输入 [B, C, L]，内部转为 [B, L, C] 计算后再转回
# ============================================================
class AttBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.filters = filters
        self.fc = nn.Linear(filters * 2, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L] （卷积的标准输出）
        返回: [B, C, L]
        """
        if x.ndim != 3:
            raise ValueError(f"AttBlock expects 3D tensor [B, C, L], got {tuple(x.shape)}")

        B, C, L = x.shape
        if C != self.filters:
            raise ValueError(f"AttBlock(filters={self.filters}) got x with C={C}. "
                             f"Make sure previous Conv1d out_channels == filters.")

        # 转为 [B, L, C] 便于按序列长度做池化
        x_lc = x.transpose(1, 2)          # [B, L, C]
        gp = torch.max(x_lc, dim=1)[0]    # [B, C]
        ap = torch.mean(x_lc, dim=1)      # [B, C]
        p = torch.cat([gp, ap], dim=1)    # [B, 2C]
        att = F.relu(self.fc(p))          # [B, C]
        att = att.unsqueeze(-1)           # [B, C, 1]
        x_out = x * att                   # [B, C, L]，通道注意力（随 C 变化，沿 L 广播）
        return x_out


# ============================================================
#  残差卷积 Block（全程使用 [B, C, L]）
# ============================================================
class Block(nn.Module):
    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels, filters, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(filters, filters, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(filters, filters, 3, padding=4, dilation=4)
        self.att1 = AttBlock(filters)
        self.att2 = AttBlock(filters)
        self.att3 = AttBlock(filters)
        self.match = nn.Conv1d(in_channels, filters, 1) if in_channels != filters else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全部保持 [B, C, L]
        out = F.relu(self.c1(x))
        out = self.att1(out)

        out = F.relu(self.c2(out))
        out = self.att2(out)

        out = F.relu(self.c3(out))
        out = self.att3(out)

        if self.match is not None:
            x = self.match(x)
        return out + x


# ============================================================
#  Lightning 模型
# ============================================================
class CNNLightningModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        # 卷积堆叠（输入最终会成为 [B, 1, L]）
        self.block1 = Block(1, 128)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = Block(128, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = Block(64, 32)
        self.pool3 = nn.MaxPool1d(2)
        self.block4 = Block(32, 16)
        self.pool4 = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.5)

        # 首层线性层动态创建（适配任意 L）
        self.fc_layers = nn.Sequential(
            nn.Identity(),                 # 占位：第一次 forward 时替换成 Linear(flatten_dim -> 2048)
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )
        self._flatten_dim = None

        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] 或 [B, L, 1]
        返回: [B, 1]
        """
        print("CNN input x.shape:", x.shape)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)          # [B, L]
        x = x.unsqueeze(1)             # [B, 1, L]
        print("CNN input x.shape:", x.shape)

        x = self.block1(x)
        print("CNN block1 output x.shape:", x.shape)
        x = self.pool1(x)   # [B, 128, L/2]
        print("CNN pool1 output x.shape:", x.shape)
        x = self.block2(x)
        print("CNN block2 output x.shape:", x.shape)
        x = self.pool2(x)   # [B,  64, L/4]
        x = self.block3(x)
        x = self.pool3(x)   # [B,  32, L/8]
        x = self.block4(x)
        x = self.pool4(x)   # [B,  16, L/16]
        x = self.drop(x)
        x = torch.flatten(x, 1)
        print("CNN flatten output x.shape:", x.shape)# [B, 16*(L/16)]

        if self._flatten_dim is None:
            self._flatten_dim = x.shape[1]
            print(f"[Init] flatten_dim = {self._flatten_dim}")
            self.fc_layers[0] = nn.Linear(self._flatten_dim, 2048).to(x.device)

        return self.fc_layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['spectrum'], batch['z']
        y_hat = self(x).squeeze(1)
        loss = F.l1_loss(y_hat, y)  # MAE
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['spectrum'], batch['z']
        y_hat = self(x).squeeze(1)
        loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", self.val_mae(y_hat, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse", self.val_mse(y_hat, y), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)


# ============================================================
#  主入口
# ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_path", type=str, default='F:/database/filterWave/data_g3_z/train_dataset')
#     parser.add_argument("--test_path", type=str,  default='F:/database/filterWave/data_g3_z/test_dataset')
#     parser.add_argument("--epochs", type=int, default=30)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--batch_size", type=int, default=512)
#     parser.add_argument("--num_workers", type=int, default=4)
#     args = parser.parse_args()
#
#     print(f"Loading datasets from {args.train_path} and {args.test_path} ...")
#     train_ds = load_from_disk(args.train_path)
#     test_ds = load_from_disk(args.test_path)
#
#     train_dataset = PairDataset(train_ds, transform=None)
#     val_dataset = PairDataset(test_ds, transform=None) # 输出如 ['spectrum', 'z']
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         persistent_workers=(args.num_workers > 0),
#         pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         persistent_workers=(args.num_workers > 0),
#         pin_memory=True
#     )
#
#     model = CNNLightningModel(lr=args.lr)
#
#     trainer = pl.Trainer(
#         max_epochs=args.epochs,
#         accelerator="auto",
#         log_every_n_steps=10,
#         callbacks=[
#             pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
#             pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=2, dirpath='./phot_check', filename="{epoch}-{val_loss:.4f}")
#         ]
#     )
#
#     trainer.fit(model, train_loader, val_loader)
