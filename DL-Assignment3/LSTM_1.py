#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:11:35 2018

@author: priyank
"""
from traindata_new import trainingdata_Xy,max_size
import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.datasets import mnist








def myGenerator():
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    
#    y_train = np_utils.to_categorical(y_train,10)
#    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255
    
    X,y=trainingdata_Xy()
    X_train=X[:,:,:10]
    y_train=y[:,:,:10]
    while 1:
        for i in range(10): # 1875 * 32 = 60000 -> # of training samples
            if i%5==0:
                print ("i = " + str(i))
            yield X_train[i*2:(i+1)*2,:,:], y_train[i*2:(i+1)*2,:,:]














#
#X,y=trainingdata_Xy()
#l=len(X)
#
#train_X, train_y = X[:l-5, :], y[:l-5, :]
#test_X, test_y =X[l-5:, :], y[l-5:, :]
#
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#max_size=max_size()
max_size=10
model = Sequential()  
model.add(LSTM(max_size, input_shape=(1, max_size),return_sequences=True))
model.add(Dense(max_size))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
#model.fit(train_X, train_y, nb_epoch=100, batch_size=70, verbose=2,validation_data=(test_X, test_y ))







model.fit_generator(myGenerator(), samples_per_epoch = 20, nb_epoch = 1, verbose=2, callbacks=[], validation_data=None, class_weight=None)



X_train=X[0,:,50000:50010]

predict = model.predict(X_train)
