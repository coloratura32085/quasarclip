import pandas as pd
import numpy as np
testdata=pd.read_csv('data_phot.csv', header=None)
testdata=np.array(testdata)
data_128=pd.read_csv('data128_att.csv',header=None)
data_128=np.array(data_128)

#对星等数据进行归一化
def scale_minmax(data):
    return (data-data.min())/(data.max()-data.min())

for i in range(0,15):
    testdata[:,i]=scale_minmax(testdata[:,i])


X = testdata

Y = data_128

#对数据进行归一化
import math
feature=128

def nom_flow(data):
    for i in range(data.shape[0]):
        sum=0
        for j in range(feature):
            sum=sum+data[i,j]**2
        for j in range(feature):
            data[i,j]=data[i,j]/math.sqrt(sum)
nom_flow(Y)

#分割数据集
from sklearn.model_selection import train_test_split
#Y = data_16
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state=42)



import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'
import math

from keras.layers import *
from keras.models import Model



input_dims = 15
inputs = Input(shape=(input_dims,))
#inputs=Flatten()(inputs)
dense1=Dense(8,activation = 'relu')(inputs)
dense2=Dense(16,activation = 'relu')(dense1)
dense3=Dense(32,activation = 'relu')(dense2)
dense4 = Dense(64,activation = 'relu')(dense3)
dense5 = Dense(128,activation = 'relu')(dense4)
model = Model(inputs = inputs,outputs = dense5)
print(model.summary())

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *

my_callbacks = [EarlyStopping(patience=10),
             ModelCheckpoint('model/gen128.h5', save_best_only=True,save_weights_only = False)]
model.compile(loss='logcosh',
              optimizer=Adam(1e-3, amsgrad=True))
model.fit(X_train,Y_train, epochs=100,validation_data = (X_test,Y_test), workers=4, use_multiprocessing=True,
                      batch_size = 512,
                    callbacks=my_callbacks)

zz=model.predict(X)

np.savetxt('gen_data128.csv',zz,delimiter=',')