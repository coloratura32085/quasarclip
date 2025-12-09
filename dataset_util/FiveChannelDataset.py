import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FiveChannelDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]['image']
        if self.transform:
            img = self.transform(img)  # Apply the transformation
        return img
    # def __init__(self, data_path, transform=None):
    #     """
    #     初始化数据集，支持按需加载 .npy 文件
    #     :param data_path: .npy 数据文件路径
    #     :param transform: 数据增强或转换函数
    #     """
    #     self.data_path = data_path
    #     self.transform = transform
    #
    #     # 使用内存映射加载 .npy 文件，避免一次性加载到内存
    #     self.data = np.load(data_path, mmap_mode='r')
    #     self.data_shape = self.data.shape  # (N, C, H, W)
    #
    # def __len__(self):
    #     """
    #     返回数据集的大小（样本数）
    #     """
    #     return self.data_shape[0]
    #
    # def __getitem__(self, idx):
    #     """
    #     根据索引返回单个样本
    #     :param idx: 数据索引
    #     """
    #     sample = self.data[idx]  # 动态读取单个样本 (C, H, W)
    #     sample = torch.tensor(sample)  # 转换为 PyTorch Tensor
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample
