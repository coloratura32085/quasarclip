import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from tqdm import tqdm

# ========================================
#  GPU 设置
# ========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("✅ Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

# ========================================
#  加载完整模型
# ========================================
base_model = tf.keras.models.load_model('specmodel-att_best.h5', compile=False)
test = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_17').output)
print(test.summary())

# ========================================
#  分批读取 + 分批预测
# ========================================
csv_path = '../data/data_spectrum.csv'
chunk_size = 256   # 每次读入 1000 条，可根据内存调整
batch_size = 256    # 每次送入 GPU 的 batch

results = []

print(f"✅ 开始分批读取文件: {csv_path}")

for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="推理中", ncols=100):
    # 转 numpy（注意：不做 astype）
    batch_data = chunk.values
    y_pred = test.predict(batch_data, batch_size=batch_size, verbose=0)
    results.append(y_pred)

# 合并所有批次预测结果
y_all = np.vstack(results)

# ========================================
#  保存结果
# ========================================
np.savetxt('data128_att.csv', y_all, delimiter=',', fmt='%s')
print("✅ 推理完成，共生成特征:", y_all.shape)
print("✅ 文件已保存为 data128_att.csv")
