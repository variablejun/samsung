import pandas as  pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
import decimal as dc
import tensorflow as tf

#1.data 
dataframe1 = pd.read_csv('../_data/SK주가 20210721.csv',encoding='cp949')
dataframe2 = pd.read_csv('../_data/삼성전자 주가 20210721.csv',encoding='cp949')
x1 =dataframe1.to_numpy()
x2 =dataframe2.to_numpy()
#x1 =dataframe1.values
#x2 =dataframe2.values
#x1 = x1.to_numpy()
#x2 = x2.to_numpy()
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
          subset = dataset[i : (i + size)].astype(np.int)
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
y2 = x2[:,[-1]]

print('========slice==========')
print(x1)
print('========Yslice==========')
print(y1)

print('========slice==========')
print(x2)
print('========Yslice==========')
print(y2)
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

