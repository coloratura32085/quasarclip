from datasets import load_from_disk

from root_path import ROOT_PATH

import matplotlib.pyplot as plt
import torch

# 假设 g3 和 iv 已经加载为 Tensor 类型
g3 = load_from_disk(f"{ROOT_PATH}/data/data_g3/train_dataset")[1000]['spectrum']
iv = load_from_disk(f"{ROOT_PATH}/data/data_ivar/train_dataset")[1000]['spectrum']

print(g3 == iv)
print(g3)
print(iv)
# 确保 g3 和 iv 是 tensor，如果不是，可以转换成 tensor
if not isinstance(g3, torch.Tensor):
    g3 = torch.tensor(g3)

if not isinstance(iv, torch.Tensor):
    iv = torch.tensor(iv)

# 创建一个图形
plt.figure(figsize=(25, 10))

# 绘制 g3 和 iv 的光谱在同一图中
plt.plot(iv.numpy(), label="iv Spectrum", color='orange')
plt.plot(g3.numpy(), label="g3 Spectrum", color='purple')


# 设置图标题和标签
plt.title("Comparison of g3 and iv Spectra")
plt.xlabel("Index")
plt.ylabel("Value")

# 显示图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
