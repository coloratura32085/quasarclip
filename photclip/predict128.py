import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import argparse
import random
import tensorflow as tf
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D, GlobalAveragePooling1D
import keras.backend as K
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *


# --- 1. 数据加载与准备（已修改） ---

# 原始代码中的 testdata (15 维) 被移除
# testdata=pd.read_csv('data_mag.csv')
# testdata=np.array(testdata)
# print(testdata.shape)

gen_128=pd.read_csv('gen_data128.csv',header=None)
gen_128=np.array(gen_128)
print(f"gen_128 shape: {gen_128.shape}")

testred=pd.read_csv('redshift.csv')
testred=np.array(testred)
print(f"testred shape: {testred.shape}")

# 原始代码中的 testdata 标准化被移除
# def scale_minmax(data):
#     return (data-data.min())/(data.max()-data.min())
# for i in range(0,15):
#     testdata[:,i]=scale_minmax(testdata[:,i])

# 只使用 128 维数据作为输入 X
X = gen_128
print(f"Model input X shape: {X.shape}")


# --- 2. 模型结构定义（已修改为单输入 128 维） ---

def BLOCK(seq, filters): # 定义网络的Block
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(int(cnn.shape[2]), 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq

# 定义单输入层，维度为 128
input_dims = 128
single_input = Input(shape=(input_dims,), name='single_input')
x = single_input

# Reshape 层：从 (128,) 变为 (128, 1)
reshape = Reshape((128, 1), input_shape=(128,))(x)
seq = reshape

# 1D CNN Blocks
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = Dropout(0.5)(seq)

# MLP 部分
seq = Flatten()(seq)
seq = Dense(64, activation='relu')(seq)
seq = Dense(32, activation='relu')(seq)
seq = Dense(16, activation='relu')(seq)
output_tensor = Dense(1)(seq)

# 定义单输入模型
model2 = Model(inputs=single_input, outputs=output_tensor)
print(model2.summary())


# --- 3. 训练和预测（已修改为单输入） ---

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, testred,test_size = 0.2, random_state=42)

# 定义回调函数
my_callbacks = [EarlyStopping(patience=4),
             ModelCheckpoint('model/blue.h5', save_best_only=True,save_weights_only = False)]

# 编译模型
model2.compile(loss='mean_absolute_error',
              optimizer=Adam(1e-3, amsgrad=True),
              metrics=['mse'])

# 拟合模型（使用 X_train）
H = model2.fit(X_train, Y_train, epochs=100, validation_data = (X_test, Y_test), workers=4, use_multiprocessing=True,
                      batch_size = 512,
                    callbacks=my_callbacks)

# 进行预测（使用 X_test）
pred = model2.predict(X_test)

# 保存结果
np.savetxt('./tmp/prediction128.csv', pred, delimiter=',', fmt='%s')
np.savetxt('./tmp/redshift128.csv', Y_test, delimiter=',', fmt='%s')
model2.save('model128.h5')

pred = np.array(pred)
Y_test = np.array(Y_test)

# 计算 delta 值
delta1 = np.abs((pred - Y_test) / (1 + Y_test))

# 统计不同区间的数量
count1 = np.sum(delta1 < 0.1)
count2 = np.sum((delta1 >= 0.1) & (delta1 < 0.2))
count3 = np.sum((delta1 >= 0.2) & (delta1 < 0.3))

# 计算各区间占 pred 长度的比重
ratio_01 = count1 / len(pred)
ratio_02 = count2 / len(pred)
ratio_03 = count3 / len(pred)
print("0.1占比:", ratio_01)
print("0.2占比:", ratio_02)
print("0.3占比:", ratio_03)

# 将结果写入 txt 日志文件
with open('evaluation_results_128_only.txt', 'a') as f:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n[{timestamp}] - 128维单输入模型\n")
    f.write(f"0.1占比: {ratio_01:.4f}\n")
    f.write(f"0.2占比: {ratio_02:.4f}\n")
    f.write(f"0.3占比: {ratio_03:.4f}\n")
    f.write("="*30 + "\n")