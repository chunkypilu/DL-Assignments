#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:11:35 2018

@author: priyank
"""
from traindata_dict_one_hot import trainingdata_Xy,max_size
import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM


X,y=trainingdata_Xy()
l=len(X)
print(X.shape,y.shape)

X=X.reshape(X.shape[0],-1,1)
y=y.reshape(y.shape[0],y.shape[2],y.shape[1],y.shape[3])

print(X.shape,y.shape)

train_X, train_y = X[:l-5, :], y[:l-5, :]
test_X, test_y =X[l-5:, :], y[l-5:, :]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

max_size=max_size()
#max_size=50000

model = Sequential()  
model.add(LSTM(20, input_shape=(max_size,1),return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.fit(train_X, train_y, nb_epoch=2, batch_size=1, verbose=2,validation_data=(test_X, test_y ))

model.summary()

a=train_X[0,:,:]
b=train_y[0,:,:]

a=a.reshape(1,a.shape[0],a.shape[1])

predict = model.predict(a)
