import os
import datasets
import pandas as pd
from datasets import Image

# ===== 路径 =====
dataset_path     = r'F:/database/filterWave/data_64_pair/ivar_z'
csv_path         = r'D:/database/final_index.csv'
image_dir        = r'F:/database/sourcedata/rgb_224'   # 图片目录
new_dataset_path = r'F:/database/filterWave/img_z_all'  # 输出目录

# 1. 载入数据
dataset = datasets.load_from_disk(dataset_path)
df      = pd.read_csv(csv_path, dtype={"objID": str})
assert len(dataset) == len(df), "Dataset 与 CSV 行数不一致！"

# 2. 删除旧 image(s) 列
for col in ("images", "image"):
    if col in dataset.column_names:
        dataset = dataset.remove_columns(col)

objids = df['objID'].astype(str)

# 3. 映射函数 —— 仅添加图片
def add_image(example, idx):
    oid      = objids.iloc[idx]
    img_path = os.path.join(image_dir, f"{oid}.jpg")
    with open(img_path, 'rb') as f:
        example["image"] = {"path": img_path, "bytes": f.read()}
    return example

# 开启 tqdm 进度条；desc 可以自定义显示文字
dataset = dataset.map(
    add_image,
    with_indices=True,
    desc="🔄 正在嵌入 JPEG 到数据集…"        # ← 进度条标题
)
dataset = dataset.cast_column("image", Image(decode=True))

# 4. 保存
dataset.save_to_disk(new_dataset_path)
print(f"✅ 处理完成，数据集已保存到：{new_dataset_path}")
import datasets
data = datasets.load_from_disk(r'F:/database/filterWave/img_z_all')
print(data.column_names)
print(data[0])


# import datasets
#
# # 加载数据集
# dataset_path = 'F:/database/filterWave/img_z_all'
# dataset = datasets.load_from_disk(dataset_path)
# print("Data loaded.")
# # 检查数据集中是否已经有 train 和 test 分割
# if 'train' not in dataset and 'test' not in dataset:
#     # 进行分割
#     train_test_split = dataset.train_test_split(test_size=0.2, seed=66)
#     # 保存分割后的数据集
#     train_test_split['train'].save_to_disk('F:/database/filterWave/data_rgb_z/train_dataset')
#     train_test_split['test'].save_to_disk('F:/database/filterWave/data_rgb_z/test_dataset')
#     print("Data split and saved.")
# else:
#     print("数据集已经包含 train 和 test 分割，无需重新分割。")