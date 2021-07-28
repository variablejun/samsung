import pandas as  pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
import decimal as dc
import tensorflow as tf

#1.data 
dataframe1 = pd.read_csv('../_data/SK주가 20210721.csv',encoding='cp949')
dataframe2 = pd.read_csv('../_data/삼성전자 주가 20210721.csv',encoding='cp949')
#datasets1 =dataframe1.values
#datasets2 =dataframe2.values
datasets1 = dataframe1.to_numpy()
datasets2 = dataframe2.to_numpy()
#datasets1 = nd.array(datasets1)
#datasets2 = nd.array(datasets2)

datasets1 = datasets1[:,[1,2,3,4,10]]
datasets2 = datasets2[:,[1,2,3,4,10]]
#ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
datasets1 = datasets1[0:2601,:]
datasets2 = datasets2[0:2601,:]
datasets1 = np.flip(datasets1,axis=0)
datasets2 = np.flip(datasets2,axis=0)

print("=======MODEL========")
print(datasets1)
print("=========sam========")
print(datasets2)
print("=======SPLIT========")
size = 6

def split_x(dataset, size):
     aaa=[]
     for i in range(len(dataset) - size + 1):
          subset = dataset[i : (i + size)].astype(int)
          aaa.append(subset)

     return np.array(aaa)


datasets1 = split_x(datasets1, size)
datasets2 = split_x(datasets2, size)
  


x1 = datasets1[:, :5]
x2 = datasets2[:, :5]
y1 = datasets2[:,[-1]]


from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train,x2_test, y1_train, y1_test = train_test_split(x1,x2,y1,train_size = 0.7) # train_size 0.7



#(2596, 5, 5)
#(2596, 5)
#(2596, 5, 5)
#(2596, 5)
#(1817, 5, 5) (779, 5, 5) (1817, 5, 5) (779, 5, 5) (1817, 5) (779, 5) (1817, 5) (779, 5)

#2.모델 구성

'''
 ValueError: Dimensions must be equal, but are 4 and 5 for '{{node mean_squared_error/SquaredDifference}} = SquaredDifference[T=DT_FLOAT](mean_squared_error/remove_squeezable_dimensions/Squeeze, mean_squared_error/Cast)' with input shapes: [?,4], [?,5].

'''

input1 = Input(shape=(5,5))
xx = Conv1D(128,2,activation = 'relu')(input1)
xx = Flatten()(xx)
xx = Dense(1024)(xx)
xx = Dropout(0.2)(xx)
xx = Dense(512)(xx)
xx = Dense(256)(xx)
xx = Dense(128)(xx)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output1 = Dense(2)(xx)

input2 = Input(shape=(5,5))
xx = Conv1D(128,2,activation = 'relu')(input2)
xx = Flatten()(xx)

xx = Dense(1024)(xx)
xx = Dropout(0.2)(xx)
xx = Dense(512)(xx)
xx = Dense(256)(xx)
xx = Dense(128)(xx)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output2 = Dense(2)(xx)

from tensorflow.keras.layers import concatenate, Concatenate  #  소문자 메소드,  대문자 클래스
merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
last_output1 = Dense(1)(merge3)

model = Model(inputs = [input1, input2], outputs = last_output1)

#3.complie/훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True,mode='auto', filepath='./_save/MCP_test1.hdf5')

import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit([x1_train,x2_train], y1_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es, cp]) 
model.save('./_save/Modelsave_test1.h5')

loss = model.evaluate([x1_test,x2_test], y1_test) 
end = time.time()- starttime
y_pred = model.predict([x1_test,x2_test]) 

print(y_pred[778])
#4.평가/예측
print("걸린시간", end)
print('loss : ', loss)




'''
??
25/25 [==============================] - 0s 6ms/step - loss: 5645876068352.0000
걸린시간 120.96780347824097
loss :  5645876068352.0

25/25 [==============================] - 0s 4ms/step - loss: 5547162075136.0000
걸린시간 26.344221353530884
loss :  5547162075136.0

y1 제거후

25/25 [==============================] - 0s 4ms/step - loss: 44474407321600.0000
걸린시간 52.47992491722107
loss :  44474407321600.0

연산수 증가
25/25 [==============================] - 0s 4ms/step - loss: 44126640799744.0000
걸린시간 136.2484200000763
loss :  44126640799744.0

[6593642.5]
걸린시간 233.87167286872864
loss :  45036783796224.0

[4322416.5]
걸린시간 49.82314085960388
loss :  48825813172224.0
'''
#'../_data/winequality-white.csv'