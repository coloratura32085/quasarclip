import numpy as np
import pandas as pd
import torch  # 导入 PyTorch 用于张量操作
from datasets import load_from_disk
from scipy.ndimage import gaussian_filter1d  # 导入高斯滤波函数
from root_path import ROOT_PATH
dataset = load_from_disk(f'{ROOT_PATH}/data/data_ivar/train_dataset')

# 2. 定义高斯滤波并转换为张量的函数
def apply_gaussian_filter(example, idx, sigma=3):
    """
    对样本中的 'spectrum' 应用高斯滤波，并将滤波后的 'spectrum' 转换为张量。

    参数:
        example (dict): 数据集中的单个样本。
        idx (int): 样本的索引。
        sigma (float): 高斯滤波的标准差。

    返回:
        dict: 更新后的样本。
    """
    spectrum = example['spectrum']
    # print(spectrum.shape)

    # 将 spectrum 转换为 NumPy 数组（如果它是张量）
    if hasattr(spectrum, 'numpy'):
        spectrum = spectrum.numpy()
    elif isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()
    else:
        # 如果 spectrum 已经是 NumPy 数组，则无需转换
        spectrum = np.array(spectrum)

    # 进行高斯滤波
    spectrum_filtered = gaussian_filter1d(spectrum, sigma=sigma)
    # print(spectrum_filtered[0:10])
    # 将滤波后的 spectrum 转换回张量
    filtered_tensor = torch.from_numpy(spectrum_filtered).float()  # 根据需要调整数据类型
    # print(filtered_tensor.shape)
    # 更新样本中的 'spectrum'
    example['spectrum'] = filtered_tensor
    return example

# 3. 将高斯滤波函数应用到整个数据集
# 使用 with_indices=True 以便传递索引到函数
# 根据系统的 CPU 核心数调整 num_proc 以优化性能
print("开始高斯滤波")
filtered_dataset = dataset.map(
    apply_gaussian_filter,
    with_indices=True,
    num_proc=1  # 根据你的系统调整这个数字
)
print("高斯滤波完成")
# 4. 保存滤波后的数据集
filtered_dataset.save_to_disk(f'{ROOT_PATH}/data/data_g3/train_dataset')

print("高斯滤波完成，数据集已成功保存。")
