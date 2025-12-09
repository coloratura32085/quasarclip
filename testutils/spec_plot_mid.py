from dataset_util.PairDataset import PairDataset
from root_path import ROOT_PATH
from datasets import load_from_disk
import matplotlib.pyplot as plt
import random
import numpy as np  # 用于 Tensor 转换
from scipy.signal import medfilt  # 导入中值滤波函数

# 数据路径
path = f'{ROOT_PATH}/data/data_ivar/train_dataset'

# 加载数据
spec = load_from_disk(path)

# 转换为 PairDataset 格式
spec = PairDataset(spec)

# 生成随机索引
random_indices = random.sample(range(len(spec)), 5)  # 随机选择 5 个索引

# 创建 5 行 * 1 列的子图
fig, axes = plt.subplots(5, 1, figsize=(25, 25))
fig.suptitle("Spectrum and Filtered Comparison (Median)", fontsize=16)

# 滤波窗口大小
kernel_size = 9  # 中值滤波的窗口大小

# 遍历随机索引并绘制到子图中
for i, idx in enumerate(random_indices):
    # 获取当前随机索引的数据，并将 Tensor 转为 NumPy 数组
    print(type(spec[idx]['spectrum']))
    print(spec[idx]['spectrum'].shape)
    spectrum = spec[idx]['spectrum'].numpy().squeeze()  # Tensor 转换为 NumPy
    print(spectrum.shape)

    # 对光谱进行中值滤波
    spectrum_filtered = medfilt(spectrum, kernel_size=kernel_size)

    # 绘制原始光谱和滤波后的光谱
    axes[i].plot(spectrum, label=f"Spectrum (Index: {idx})", color='blue', alpha=0.7)
    axes[i].plot(spectrum_filtered, label=f"Filtered Spectrum (Index: {idx})", color='red', alpha=0.7)

    # 设置子图的标题、坐标轴标签和网格
    axes[i].set_title(f"Spectrum Comparison (Index: {idx})")
    axes[i].set_xlabel("Feature Index")
    axes[i].set_ylabel("Spectrum Value")
    axes[i].legend()  # 添加图例
    axes[i].grid(True)

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 防止标题与子图重叠
plt.show()
