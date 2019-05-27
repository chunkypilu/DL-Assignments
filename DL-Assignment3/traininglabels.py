#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:01:28 2018

@author: emblab
"""
import numpy as np
import os


PATH='/home/emblab/Desktop/Assignment_3/DL_ASS_3_data_toy'

start=0
end=1
mid=2
space=3

max_size=89191
y=[]

        
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
                               y_temp=np.zeros(max_size)
                               
                               for i in range(a[0]):
                                   y_temp[i]=3
                               for j in range(l-1):
                                   for k in range(b[j]+1,a[j+1]):
                                       y_temp[k]=3
                               for i in range(b[l-1]+1,max_size):
                                   y_temp[i]=3        
                               for m in range(l):
                                   for n in range(a[m],b[m]+1):
                                       y_temp[n]=2
                               for o in range(l):
                                   y_temp[a[o]]=0
                                   y_temp[b[o]]=1
                                       
                                       
                               y.append(y_temp)


docs=[]   
folders=os.listdir(PATH)       
for i in folders:
           PATH1=PATH+'/'+i
           folders1=os.listdir(PATH1)
           for j in folders1:
                PATH2=PATH1+'/'+j
                folders2=os.listdir(PATH2)
                for k in folders2:
                      if 'wrd' in k:
                             PATH3=PATH2+'/'+k
                             func(PATH3)
                             
#                             with open (PATH3, "r") as myfile:
#                               a=[]
#                               b=[]
#                               for line in myfile:
#                                      #line=line[8:]
#                                      print(line)
#                                      x=0
#                                      for l in range(len(line)):
#                                             
#                                             if line[l]==' ':
#                                                 a.append(line[0:l])
#                                                 k=l
#                                                 break
#                                                 
#                                      for j in range(k+1,len(line)) :           
#                                             if line[j]==' ':    
#                                                 b.append(line[k+1:j])
#                                                 break
#                                      #line=line.split(" ")[2:]
#                                      #docs.append(line[l+1:])
                              
    
    
                                       