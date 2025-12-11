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


testdata=pd.read_csv('data_mag.csv')
testdata=np.array(testdata)
print(testdata.shape)
gen_128=pd.read_csv('gen_data128.csv',header=None)
gen_128=np.array(gen_128)
print(gen_128.shape)
testred=pd.read_csv('redshift.csv')
testred=np.array(testred)
print(testred.shape)

#实验加入星等数据的模型
input_dims = 15
main_inputs = Input(shape=(input_dims,),name='main_input')
auxiliary_input=Input(shape=(128,),name='auxiliary_input')

x=concatenate([main_inputs,auxiliary_input])
'''
dense1 = Dense(64, activation='relu')(x)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(1)(dense3)
model = Model(inputs = [main_inputs,auxiliary_input],outputs = dense4)
print(model.summary())
'''

def BLOCK(seq, filters): # 定义网络的Block
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)   #filters*3
    #cnn = attblock(cnn,filters)
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    #cnn = attblock(cnn,filters)
    cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    #cnn = attblock(cnn,filters)
    #print('cnn',int(cnn.shape[2]))
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(int(cnn.shape[2]), 1, padding='SAME')(seq)
    #print('seq',int(seq.shape[2]))
    seq = add([seq, cnn])
    return seq

#input_dims = 3600
#inputs = Input(shape=(input_dims,))
reshape = Reshape((143, 1), input_shape=(143,))(x)
seq = reshape

#seq = BLOCK(seq, 128)
#seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
#seq = BLOCK(seq, 32)
#seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = Dropout(0.5)(seq)

seq = Flatten()(seq)
seq = Dense(64, activation='relu')(seq)
#seq = Dropout(0.5)(seq)
seq = Dense(32, activation='relu')(seq)
#seq = Dropout(0.5)(seq)
seq = Dense(16, activation='relu')(seq)
#seq = Dropout(0.5)(seq)
#seq = Dense(256, activation='relu')(seq)
#seq = Dropout(0.5)(seq)
#seq = Dense(128, activation='relu')(seq)
#seq = Dense(64, activation='relu')(seq)
#seq = Dense(32, activation='relu')(seq)
#seq = Dense(16, activation='relu')(seq)
#seq = Dense(8, activation='relu')(seq)
output_tensor = Dense(1)(seq)
model2 = Model(inputs=[main_inputs,auxiliary_input], outputs=output_tensor)
print(model2.summary())


#normalization
def scale_minmax(data):
    return (data-data.min())/(data.max()-data.min())

for i in range(0,15):
    testdata[:,i]=scale_minmax(testdata[:,i])

X=np.hstack((testdata,gen_128))

X_train, X_test, Y_train, Y_test = train_test_split(X, testred,test_size = 0.2, random_state=42)

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
my_callbacks = [EarlyStopping(patience=4),
             ModelCheckpoint('model/blue.h5', save_best_only=True,save_weights_only = False)]

model2.compile(loss='mean_absolute_error',
              optimizer=Adam(1e-3, amsgrad=True),
              metrics=['mse'])
H = model2.fit([X_train[:,:15],X_train[:,15:]],Y_train, epochs=100,validation_data = ([X_test[:,:15],X_test[:,15:]],Y_test), workers=4, use_multiprocessing=True,
                      batch_size = 512,
                    callbacks=my_callbacks)

pred = model2.predict([X_test[:,:15],X_test[:,15:]])
np.savetxt('./tmp/prediction143.csv', pred, delimiter=',', fmt='%s')
np.savetxt('./tmp/redshift143.csv', Y_test, delimiter=',', fmt='%s')
model2.save('model143.h5')

pred = np.array(pred)
Y_test = np.array(Y_test)

# 计算delta值
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

# 将结果写入txt日志文件
with open('evaluation_results.txt', 'a') as f:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n[{timestamp}]\n")
    f.write(f"0.1占比: {ratio_01:.4f}\n")
    f.write(f"0.2占比: {ratio_02:.4f}\n")
    f.write(f"0.3占比: {ratio_03:.4f}\n")
    f.write("="*30 + "\n")