#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:04:10 2018

@author: priyank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:12:05 2018

@author: emblab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:29:39 2018

@author: emblab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 05:06:17 2018

@author: emblab
"""

import numpy as np
from keras.preprocessing.text import one_hot
import librosa
import librosa.display
import os
from keras.preprocessing.sequence import pad_sequences
import pickle






with open("mySavedDict_train_test.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)
#print (myNewPulledInDictionary)   



PATH='/media/priyank/6442175942172F741/DL_ASS_3_data_toy'
PATH_='/home/priyank/Desktop/Assignment_3__/Assignment_3_test_toy'


y=[]
docs=[] 

def max_size():
    docs=[]   
    folders=os.listdir(PATH)       
    for i in folders:
           PATH1=PATH+'/'+i
           folders1=os.listdir(PATH1)
           for j in folders1:
                PATH2=PATH1+'/'+j
                folders2=os.listdir(PATH2)
                for k in folders2:
                      if 'wav' in k:
                             PATH3=PATH2+'/'+k
                             #print(k)
                             y_val, sr = librosa.load(PATH3,sr=None)
                             docs.append(y_val)



    folders=os.listdir(PATH_)       
    for i in folders:
           PATH1=PATH_+'/'+i
           folders1=os.listdir(PATH1)
           for j in folders1:
                PATH2=PATH1+'/'+j
                folders2=os.listdir(PATH2)
                for k in folders2:
                      if 'wav' in k:
                             PATH3=PATH2+'/'+k
                             #print(k)
                             y_val, sr = librosa.load(PATH3,sr=None)
                             docs.append(y_val)
                         
   
    max_size=docs[0].size
    for i in range(len(docs)):
           if docs[i].size>max_size:
                  max_size=docs[i].size
                  #print(i)
    
    return max_size


def func(PATH3):
    
      m=max_size()
      with open (PATH3, "r") as myfile:
                               a=[]
                               b=[]
                               c=[]
                               for line in myfile:
                                      #print(line)
                                      for l in range(len(line)):
                                             
                                             if line[l]==' ':
                                                 a.append(line[0:l])
                                                 k=l
                                                 #print('k=====',k)
                                                 break
                                                 
                                      for j in range(k+1,len(line)) :           
                                             if line[j]==' ':    
                                                 b.append(line[k+1:j])
                                                 #print('j=======',j)
                                                 t=j
                                                 break
                                             
                                            
                                      c.append(line[t+1:len(line)-1])              
                                                
                                             
                               #print(a)
                               #print(b)
                               #print(c)
                               a=[int(i) for i in a]
                               b=[int(i) for i in b]
                               
                               a=np.array(a)
                               b=np.array(b)
                               #print(a)
                               l=len(a)
                               y_temp=np.zeros(m)
                               
                               y_temp[0:a[0]]=0    
                                   
                               for j in range(l-1):
                                   y_temp[b[j]+1:a[j+1]]=0

                               y_temp[b[l-1]+1:m]=0
                               
                               for m in range(l):   
                                   tt=c[m]
                                   #print(myNewPulledInDictionary[tt])
                                   y_temp[a[m]:b[m]+1]=myNewPulledInDictionary[tt]
                                
                               a=np.array([a])
                               b=np.array([b])
                               c=np.concatenate((a,b),axis=0) 
                               
                               print('\n\n',c.shape)
                               return y_temp,c




def trainingdata_Xy():
    
    start=0
    end=1
    mid=2
    space=3
    #max_size=89191
    y=[]
    docs=[]
    boundaries=[]
    folders=os.listdir(PATH_)       
    for i in folders:
           PATH1=PATH_+'/'+i
           folders1=os.listdir(PATH1)
           for j in folders1:
                PATH2=PATH1+'/'+j
                folders2=os.listdir(PATH2)
                for k in folders2:
                      if 'wav' in k:
                             PATH3=PATH2+'/'+k
                             print(PATH3)
                             y_val, sr = librosa.load(PATH3,sr=None)
                             
                             
                             y_val1=np.array([y_val[:,]])
                             docs.append(y_val1)
                             
                             k1=k.split('.')[0]
                      #if 'wrd' in k:
                             PATH3=PATH2+'/'+k1+'.wrd'
                             print(PATH3)
                             val,c=func(PATH3)
                             boundaries.append(c)
                             
                             #a=a.reshape(1,a.shape)
                             #b=b.reshape(1,b.shape)
                             print('\n\n')
                             
                             val1=np.array([val[:,]])
                             y.append(val1)
                      
                        
                        
   
    #max_size=docs[0].size
    #for i in range(len(docs)):
     #      if docs[i].size>max_size:
      #            max_size=docs[i].size
                  #print(i)
    
    max_size1=max_size()
    
    for j in range(len(docs)):
      diff=max_size1- (docs[j].shape[1])
      z = np.zeros((1,diff), dtype=docs[j].dtype)
      docs[j]=np.concatenate((docs[j],z), axis=1)
      
    X=np.array(docs) 
    y=np.array(y)
    
    return X,y,boundaries 
 

if __name__== "__main__":
   X,y,boundaries=trainingdata_Xy()
   
   #print(boundaries.shape)
   np.save('test_X_dict_toy',X)
   np.save('test_y_dict_toy',y)
   
   
