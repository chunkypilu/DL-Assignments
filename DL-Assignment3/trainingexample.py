import numpy as np
from keras.preprocessing.text import one_hot
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from keras.preprocessing.sequence import pad_sequences

PATH='/home/emblab/Desktop/Assignment_3/DL_ASS_3_data_toy'

def trainingexample_X():
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
                             print(k)
                             y_val, sr = librosa.load(PATH3,sr=None)
                             docs.append(y_val)
   
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
    return X   
      
      
      
      
      