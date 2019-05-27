#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 01:30:59 2018

@author: emblab
"""

import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import load_model

from traindata_dict_one_hot import trainingdata_Xy,max_size

X,y=trainingdata_Xy()
l=len(X)
print(X.shape,y.shape)

yy=y[1]
xx=X[0]

X=X.reshape(X.shape[0],-1,1)
y=y.reshape(y.shape[0],y.shape[2],y.shape[3])

print(X.shape,y.shape)

yy1=y[1]
xx1=X[0]


train_X, train_y = X[:l-5, :], y[:l-5, :]
test_X, test_y =X[l-5:, :], y[l-5:, :]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



model=load_model('/home/emblab/Desktop/Assignment_3__/model_toy.h5')
model.summary()


a=train_X[0,:,:]
b=train_y[0,:,:]
a=a.reshape(1,a.shape[0],a.shape[1])

predict = model.predict(a)


aa=predict[0]

lll=aa.shape[1]


for xx in range (predict.shape[0]):    
    for i in range(predict[xx].shape[0]):
        max_=predict[xx][i][0] 
        index=0
        for j in range(predict[xx].shape[1]):
           if predict[xx][i][j]>max_:
               index=j
           else  :
              predict[xx][i][j]=0
        predict[xx][i][index]=1       

#y_classes = predict.argmax()



