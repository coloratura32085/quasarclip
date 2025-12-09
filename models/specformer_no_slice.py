import math

import lightning as L
# import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.modules import LayerNorm, TransformerBlock, _init_by_depth


class SpecFormer(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    def forward(self, x: Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor):
        """Forward pass through the model."""
        t = x.shape[1]
        if t > self.hparams.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.hparams.max_len}"
            )

        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        # print("Input shape:", x.shape)  # 调试：输入形状 (batch_size, seq_length, input_dim)

        data_emb = self.data_embed(x)  # shape (batch_size, seq_length, embed_dim)
        # print("Data embed shape:", data_emb.shape)

        pos_emb = self.position_embed(pos)  # shape (seq_length, embed_dim)
        # print("Position embed shape:", pos_emb.shape)

        x = self.dropout(data_emb + pos_emb)  # shape (batch_size, seq_length, embed_dim)
        # print("After dropout shape:", x.shape)

        for block in self.blocks:
            x = block(x)  # shape (batch_size, seq_length, embed_dim)
        # print("After transformer block shape:", x.shape)

        x = self.final_layernorm(x)  # shape (batch_size, seq_length, embed_dim)
        # print("After layernorm shape:", x.shape)
        reconstructions = self.head(x)  # shape (batch_size, seq_length, input_dim)
        # print("Final output shape:", reconstructions.shape)

        return {"reconstructions": reconstructions, "embedding": x}

    def training_step(self, batch):
        # slice the input and copy
        input = self.preprocess(batch["spectrum"])
        target = torch.clone(input)

        # mask parts of the input
        input = self.mask_sequence(input)
        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]

        # find the mask locations
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        # self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        # slice the input and copy
        input = self.preprocess(batch["spectrum"])
        target = torch.clone(input)

        # mask parts of the input
        input = self.mask_sequence(input)

        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]

        # find the mask locations
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        # self.log("val_training_loss", loss, prog_bar=True)
        return loss

    def mask_sequence(self, x: Tensor):
        """Mask batched sequence"""
        return torch.stack([self._mask_seq(el) for el in x])

    def preprocess(self, x):
        std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
        x = (x - mean) / std
        # print("Shape after std and mean calculation:", x.shape)
        #x = self._slice(x)
        # print("Shape after slice", x.shape)
        x = x.transpose(1, 2)
        # x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
        # print("Shape after padding:", x.shape)
        # x[:, 0, 0] = ((mean.squeeze() - 2) / 2)
        # x[:, 0, 1] = ((std.squeeze() - 2) / 8)
        # print("preprocess shape: " + str(x.shape))
        # 在第 3 维（sequence_length）最前面加 2 个零的 padding
        x = F.pad(x, pad=(2, 0, 0, 0), mode="constant", value=0)
        # print("Shape after padding:", x.shape)
        # 继续后续操作
        x[:, 0, 0] = ((mean.squeeze() - 2) / 2)
        x[:, 0, 1] = ((std.squeeze() - 2) / 8)
        return x

    def _reset_parameters_datapt(self):
        # not scaling the initial embeddngs.
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        # transformer block weights
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

    def _slice(self, x):
        start_indices = np.arange(
            0,
            x.shape[1] - self.hparams.slice_overlap,
            self.hparams.slice_section_length - self.hparams.slice_overlap,
        )
        # print(x.shape)
        sections = [
            x[:, start : start + self.hparams.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]

        # If the last section is not of length 'section_length', you can decide whether to keep or discard it
        if sections[-1].shape[1] < self.hparams.slice_section_length:
            sections.pop(-1)  # Discard the last section

        return torch.cat(sections, 1)

    def _mask_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly masks contiguous sections of an unbatched sequence,
        ensuring separation between chunks is at least chunk_width."""
        len_ = seq.shape[1]
        num_chunks = self.hparams.mask_num_chunks
        chunk_width = self.hparams.mask_chunk_width

        # Ensure there's enough space for the chunks and separations
        total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width
        if total_width_needed > len_:
            raise ValueError("Sequence is too short to mask")
        masked_seq = seq.clone()

        for i in range(num_chunks):
            start = (i * len_) // num_chunks
            loc = torch.randint(0, len_ // num_chunks - chunk_width, (1,)).item()
            masked_seq[:,loc + start : loc + start + chunk_width] = 0

        return masked_seq


# model = SpecFormer(
#     input_dim=22,
#     embed_dim=768,
#     num_layers=6,
#     num_heads=6,
#     max_len=5000,
#     mask_num_chunks=6,
#     mask_chunk_width=50,
#     slice_section_length=20,
#     slice_overlap=10,
#     dropout=0.,
#     norm_first=False,
# )
#
# x =torch.randn(10,4501,1)
# out = model(x)