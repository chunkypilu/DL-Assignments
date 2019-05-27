#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:00:20 2018

@author: priyank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:11:35 2018

@author: priyank
"""
from traindata_dict_final import trainingdata_Xy,max_size
import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM


#X,y=trainingdata_Xy()

X=np.load('/home/priyank/Desktop/Assignment_3__/attachments/train_X_dict_toy.npy')
y=np.load('/home/priyank/Desktop/Assignment_3__/attachments/train_y_dict_toy.npy')

l=len(X)
print(X.shape,y.shape)

yy=y[1]
xx=X[0]

X=X.reshape(X.shape[0],-1,1)
y=y.reshape(y.shape[0],y.shape[2],y.shape[1])

print(X.shape,y.shape)

yy1=y[1]
xx1=X[0]


train_X, train_y = X[:l-5, :], y[:l-5, :]
test_X, test_y =X[l-5:, :], y[l-5:, :]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

max_size1=max_size()
#max_size=50000

model = Sequential()  
model.add(LSTM(20, input_shape=(max_size1,1),return_sequences=True))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])


model.fit(train_X, train_y, nb_epoch=2, batch_size=1, verbose=2,validation_data=(test_X, test_y ))

model.summary()

model.save('/home/priyank/Desktop/Assignment_3__/model_final_toy.h5')
#
#a=train_X[0,:,:]
#b=train_y[0,:,:]
#a=a.reshape(1,a.shape[0],a.shape[1])
#
#predict = model.predict(a)
#
#
#aa=predict[0]
#
#lll=aa.shape[1]
#
#
#for i in range(aa.shape[0]):
#    max_=aa[i][0] 
#    
#    for j in range(aa.shape[1]):
#       if aa[i][j]>max_:
#           index=j
#       else  :
#          aa[i][j]=0
#    aa[i][index]=1       
#
##y_classes = predict.argmax()
