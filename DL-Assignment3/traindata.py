#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 05:06:17 2018

@author: emblab
"""

import numpy as np
from keras.preprocessing.text import one_hot
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from keras.preprocessing.sequence import pad_sequences

PATH='/home/emblab/Desktop/Assignment_3/DL_ASS_3_data_toy'

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
   
    max_size=docs[0].size
    for i in range(len(docs)):
           if docs[i].size>max_size:
                  max_size=docs[i].size
                  #print(i)
    
    return max_size


def func(PATH3):
    
      with open (PATH3, "r") as myfile:
                               a=[]
                               b=[]
                               for line in myfile:
                                      #line=line[8:]
                                      print(line)
                                      for l in range(len(line)):
                                             
                                             if line[l]==' ':
                                                 a.append(line[0:l])
                                                 k=l
                                                 break
                                                 
                                      for j in range(k+1,len(line)) :           
                                             if line[j]==' ':    
                                                 b.append(line[k+1:j])
                                                 break
                               a=[int(i) for i in a]
                               b=[int(i) for i in b]
                               
                               a=np.array(a)
                               b=np.array(b)
                               print(a)
                               l=len(a)
                               y_temp=np.zeros(max_size())
                               
                               for i in range(a[0]):
                                   y_temp[i]=3
                               for j in range(l-1):
                                   for k in range(b[j]+1,a[j+1]):
                                       y_temp[k]=3
                               for i in range(b[l-1]+1,max_size()):
                                   y_temp[i]=3        
                               for m in range(l):
                                   for n in range(a[m],b[m]+1):
                                       y_temp[n]=2
                               for o in range(l):
                                   y_temp[a[o]]=0
                                   y_temp[b[o]]=1
                                       
                                       
                               #y.append(y_temp)
                               return y_temp




def trainingdata_Xy():
    
    start=0
    end=1
    mid=2
    space=3
    #max_size=89191
    y=[]
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
                             print(PATH3)
                             y_val, sr = librosa.load(PATH3,sr=None)
                             docs.append(y_val)
                             
                             k1=k.split('.')[0]
                      #if 'wrd' in k:
                             PATH3=PATH2+'/'+k1+'.wrd'
                             print(PATH3)
                             y.append(func(PATH3))
                      
                        
                        
   
    max_size=docs[0].size
    for i in range(len(docs)):
           if docs[i].size>max_size:
                  max_size=docs[i].size
                  print(i)
    
    
    
    for j in range(len(docs)):
      diff=max_size- len(docs[j])
      z = np.zeros(diff, dtype=docs[j].dtype)
      docs[j]=np.concatenate((docs[j],z), axis=0)
      
    X=np.array(docs) 
    y=np.array(y)
    return X,y   
 

if __name__== "__main__":
   X,y=trainingdata_Xy()