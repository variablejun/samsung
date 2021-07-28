import pandas as  pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
import decimal as dc
import tensorflow as tf
dataframe1 = pd.read_csv('../_data/SK주가 20210721.csv',encoding='cp949')
dataframe2 = pd.read_csv('../_data/삼성전자 주가 20210721.csv',encoding='cp949')

x1 =dataframe1.values
x2 =dataframe2.values

x1 = x1[:,[1,2,3,4,10]]
x2 = x2[:,[1,2,3,4,10]]

x1 = x1[0:2601,:].astype(np.float)
x2 = x2[0:2601,:].astype(np.float)
x1 = np.flip(x1,axis=0)
x2 = np.flip(x2,axis=0)
print('==================')
print(x1)
print('==================')
print(x2)

size = 5
def split_x(dataset, size):
     aaa=[]
     for i in range(len(dataset) - size + 1):
          subset = dataset[i : (i + size)].astype(np.int32)
          aaa.append(subset)

     return np.array(aaa)
x1 = split_x(x1, size)
x2 = split_x(x2, size)
print('========split==========')
print(x1)
print('========split==========')
print(x2)

x1 = x1[:, :5]
x2 = x2[:, :5]
y1 = x1[:,[-1]]
y2 = x2[:,[-1]]


from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train,x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2,train_size = 0.7, random_state=66) # train_size 0.7
#(2596, 5, 5)
#(2596, 5)
#(2596, 5, 5)
#(2596, 5)
#(1817, 5, 5) (779, 5, 5) (1817, 5, 5) (779, 5, 5) (1817, 5) (779, 5) (1817, 5) (779, 5)
print('========train==========')
print(x1_train, x2_train)
print('========test==========')
print(x1_test, x2_test)
print('========train==========')
print(y1_train,y2_train)
print('========test==========')
print(y1_test, y2_test)

'''


'''
model = Sequential()
model.add(Conv1D(128,2,activation = 'relu',input_shape=(5,5)))
model.add(LSTM(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))

'''
#2.모델 구성
input1 = Input(shape=(5,5))
xx = LSTM(32,activation = 'relu')(input1)
xx = Dense(128)(xx)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output1 = Dense(1)(xx)

input2 = Input(shape=(5,5))
xx = LSTM(32,activation = 'relu')(input2)
xx = Dense(128)(xx)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output2 = Dense(1)(xx)

from tensorflow.keras.layers import concatenate, Concatenate  #  소문자 메소드,  대문자 클래스
merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
last_output1 = Dense(1)(merge3)

model = Model(inputs = [input1, input2], outputs = last_output1)

'''
#3.complie/훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True,mode='auto', filepath='_save/MCP_samsung.hdf5')

print('========train==========')
print(x1_train, x2_train)
print('========test==========')
print(x1_test, x2_test)
print('========train==========')
print(y1_train,y2_train)
print('========test==========')
print(y1_test, y2_test)

import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x1_train ,y1_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es,cp]) 
model.save('_save/Modelsave_samsung.h5')
print('========train==========')
print(x1_train, x2_train)
print('========test==========')
print(x1_test, x2_test)
print('========train==========')
print(y1_train,y2_train)
print('========test==========')
print(y1_test, y2_test)

loss = model.evaluate(x1_test,y1_test) 
end = time.time()- starttime

#4.평가/예측
y_pred = model.predict(x1_test) 

print(y_pred) # 779개 나옴
print("걸린시간", end)
print('loss : ', loss)

print('========train==========')
print(x1_train, x2_train)
print('========test==========')
print(x1_test, x2_test)
print('========train==========')
print(y1_train,y2_train)
print('========test==========')
print(y1_test, y2_test)



'''
??
걸린시간 120.96780347824097
loss :  5645876068352.0


걸린시간 168.537015914917
loss :  5557168111616.0


걸린시간 212.89712500572205
loss :  5558651322368.0



걸린시간 43.40117406845093
loss :  5166423605248.0


걸린시간 39.79009437561035
loss :  5133884719104.0

'''
