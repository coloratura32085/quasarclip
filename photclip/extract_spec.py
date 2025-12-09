import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, RepeatVector, Reshape, add, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ========================================
#  GPU 设置：只使用第6、7号GPU
# ========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("✅ Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

# ========================================

#  读取数据
# ========================================
testred = pd.read_csv('../data/redshift.csv')
testdata = pd.read_csv('../data/data_spectrum.csv')
print(testred.shape)
print(testdata.shape)

# ========================================
#  定义模块
# ========================================
def attblock(cnn, filters):
    gp = GlobalMaxPooling1D()(cnn)
    ap = GlobalAveragePooling1D()(cnn)
    p = tf.keras.layers.concatenate([gp, ap])
    att = Dense(filters, activation='relu')(p)
    att = RepeatVector(cnn.shape[1])(att)
    cnn = Multiply()([att, cnn])
    return cnn

def BLOCK(seq, filters):
    cnn = Conv1D(filters, 3, padding='same', activation='relu')(seq)
    cnn = attblock(cnn, filters)
    cnn = Conv1D(filters, 3, padding='same', dilation_rate=2, activation='relu')(cnn)
    cnn = attblock(cnn, filters)
    cnn = Conv1D(filters, 3, padding='same', dilation_rate=4, activation='relu')(cnn)
    cnn = attblock(cnn, filters)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='same')(seq)
    seq = add([seq, cnn])
    return seq

# ========================================
#  构建模型
# ========================================
input_dims = 3600
inputs = Input(shape=(input_dims,))
x = Reshape((3600, 1))(inputs)

x = BLOCK(x, 128)
x = MaxPooling1D(2)(x)
x = BLOCK(x, 64)
x = MaxPooling1D(2)(x)
x = BLOCK(x, 32)
x = MaxPooling1D(2)(x)
x = BLOCK(x, 16)
x = MaxPooling1D(2)(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs, outputs)
# #model.summary()
#
# # ========================================
# #  数据集划分
# # ========================================
# trainX, testX, trainY, testY = train_test_split(testdata, testred, test_size=0.2, random_state=42)
#
# # ========================================
# #  编译模型
# # ========================================
# model.compile(
#     loss='mean_absolute_error',
#     optimizer=Adam(1e-3, amsgrad=True),
#     metrics=['mse']
# )
#
# # ========================================
# #  回调函数设置：EarlyStopping + 最优保存 + 日志
# # ========================================
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
#     ModelCheckpoint('specmodel-att_best.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1),
#     CSVLogger('train_log.csv', append=False)
# ]
#
# # ========================================
# #  训练模型
# # ========================================
# H = model.fit(
#     trainX, trainY,
#     validation_data=(testX, testY),
#     epochs=30,
#     batch_size=512,
#     workers=1,
#     use_multiprocessing=False,
#     callbacks=callbacks,
#     verbose=1
# )
#
# # ========================================
# #  保存最终权重（防止被覆盖）
# # ========================================
# model.save_weights('specmodel-att_last_weights.h5')
# print("✅ 训练完成，最优模型已保存为 specmodel-att_best.h5")

