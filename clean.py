import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm

from dataset_util.SpecDataset import SpecDataset
from models.specformer import SpecFormer
from root_path import ROOT_PATH
from task.get_embed import remove_prefix
import os

# 下面是 unshape 函数
def unshape(x):
    out = []
    for j in range(x.shape[0]):
        t = x[j, :, :]
        result = []
        # 转换为 NumPy 数组
        t = np.array(t)
        result.append(t[0, 0:10])
        for i in range(1, t.shape[0] - 1):
            overlap = (t[i, 10:] + t[i + 1, 0:10]) / 2
            result.append(overlap)
        result.append(t[t.shape[0] - 1 , 10:])
        result = np.array(result).flatten()
        out.append(result)
    return out

# 预训练权重路径
pretrained_weights = {
    "specformer": '/home/kongxiao/data45/kx/dongbixin/outputs/spec/spec_g3/logs/lightning_logs/version_0/checkpoints/epoch=227-step=128820.ckpt',
}

# 加载预训练模型
checkpoint = torch.load(pretrained_weights["specformer"])
params = checkpoint["hyper_parameters"]
del params['lr'], params['weight_decay'], params['T_max'], params['T_warmup']
specformer = SpecFormer(**params)
state_dict = remove_prefix(checkpoint["state_dict"], 'model.')
specformer.load_state_dict(state_dict, strict=False)

# 检查是否有可用的 GPU
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
specformer.to(device)

# 加载数据
data = datasets.load_from_disk(f'{ROOT_PATH}/data/data_g3_z/test_dataset')

# 创建数据加载器
batch_size = 64  # 设置每批数据的大小
dataset = SpecDataset(data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 结果保存路径
reconstruction_file_path = f'{ROOT_PATH}/data/testPre/reconstruction_{{}}.npy'
embedding_file_path = f'{ROOT_PATH}/data/testPre/embedding_{{}}.npy'

# 初始化输出和嵌入结果列表
reconstruction_list = []  # 用于保存每批次的 reconstruction 数据
embedding_list = []  # 用于保存每批次的 embedding 数据

# 分批次处理数据，添加进度条
for batch_idx, spectra_batch in enumerate(tqdm(data_loader, desc="Processing batches", ncols=100)):
    # 确保数据在正确的设备上
    spectra_batch = {key: value.to(device) for key, value in spectra_batch.items()}

    # 将数据传递给模型
    output_batch = specformer(spectra_batch['spectrum'])['reconstructions']
    embedding_batch = specformer(spectra_batch['spectrum'])['embedding']

    # 获取模型输出并进行后处理
    output_batch = output_batch.detach().cpu().numpy()
    output_batch = output_batch[:, :, 2:]

    embedding_batch = embedding_batch.detach().cpu().numpy()

    # 处理批次输出
    output_batch = unshape(output_batch)

    # 将每批数据添加到列表中
    reconstruction_list.append(output_batch)
    embedding_list.append(embedding_batch)

    # 每个批次处理完后，将保存部分结果
    if (batch_idx + 1) % 500 == 0 or batch_idx == len(data_loader) - 1:  # 每 500 个批次保存一次
        # 处理并保存 reconstruction 数据
        if reconstruction_list:
            # 将每个批次保存为一个单独的文件
            reconstruction_filename = reconstruction_file_path.format(batch_idx + 1)
            np.save(reconstruction_filename, np.concatenate(reconstruction_list, axis=0), allow_pickle=False)
            reconstruction_list = []  # 清空列表

        # 处理并保存 embedding 数据
        if embedding_list:
            # 将每个批次保存为一个单独的文件
            embedding_filename = embedding_file_path.format(batch_idx + 1)
            np.save(embedding_filename, np.concatenate(embedding_list, axis=0), allow_pickle=False)
            embedding_list = []  # 清空列表

# 最终输出
print(f"Reconstruction files saved as {ROOT_PATH}/data/testPre/reconstruction_{{}}.npy")
print(f"Embedding files saved as {ROOT_PATH}/data/testPre/embedding_{{}}.npy")
