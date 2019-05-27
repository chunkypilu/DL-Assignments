

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:04:44 2018

@author: priyank
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
from testdata_dict_final import *




X_test=np.load('/home/priyank/Desktop/Assignment_3__/attachments/test_X_dict_toy.npy')
y_test=np.load('/home/priyank/Desktop/Assignment_3__/attachments/test_y_dict_toy.npy')
print(X_test.shape,y_test.shape)

#X_test=X_test[:,:,0:118989]
#y_test=y_test[:,:,0:118989]


print('\n\n\n',X_test.shape,y_test.shape)





X_test=X_test.reshape(X_test.shape[0],-1,1)
y_test=y_test.reshape(y_test.shape[0],y_test.shape[2],y_test.shape[1])

print(X_test.shape,y_test.shape)
model=load_model('/home/priyank/Desktop/Assignment_3__/model_final_toy.h5')


#a=train_X[0,:,:]
#b=train_y[0,:,:]
#a=a.reshape(1,a.shape[0],a.shape[1])
#

predict = model.predict(y_test)
print(predict.shape)

a,b,boundaries=trainingdata_Xy()




flag=0
count=0
count1=0
for i in range(predict.shape[0]):
       for j in range(boundaries[i].shape[1]):
             
              start=boundaries[i][0][j]
              end=boundaries[i][1][j]
              print(start,end)
              
              for k in range(start,end+1):
                  if (predict[i][k+1][0]-predict[i][k][0]>=.0000000000000000000000000000000000000001):
                         flag=0
                         
                         #print('true')
                  else:    
                         flag=1
                         #print('false')
                         
              if (flag==0):
                    print('1 word')
                    count=count+1

              else:
                    print('more than 1 word')
                    count1=count1+1
