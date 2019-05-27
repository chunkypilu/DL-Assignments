
import numpy as np
from keras.preprocessing.text import one_hot
import librosa
import librosa.display
import os
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.utils import np_utils





with open("mySavedDict_train_test.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)
#print (myNewPulledInDictionary)   

dict_size=len(myNewPulledInDictionary)    



PATH='/media/priyank/6442175942172F741/DL_ASS_3_data_toy'
PATH_='/media/priyank/6442175942172F741/DL_ASS_3_data_toy'

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
                                
                               num_classes=dict_size+1
                               y_temp = np_utils.to_categorical(y_temp, num_classes) 
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
                             
                             
                             y_val1=np.array([y_val[:,]])
                             docs.append(y_val1)
                             
                             k1=k.split('.')[0]
                      #if 'wrd' in k:
                             PATH3=PATH2+'/'+k1+'.wrd'
                             print(PATH3)
                             val=func(PATH3)
                             val1=np.array([val[:,]])
                             y.append(val1)
                      
                        
                        
   
    #max_size=docs[0].size
    #for i in range(len(docs)):
    #      if docs[i].size>max_size:
    #              max_size=docs[i].size
                  #print(i)
    max_size=max_size()
    
    
    for j in range(len(docs)):
      diff=max_size- (docs[j].shape[1])
      z = np.zeros((1,diff), dtype=docs[j].dtype)
      docs[j]=np.concatenate((docs[j],z), axis=1)
      
    X=np.array(docs) 
    y=np.array(y)

    X=X.reshape(X.shape[0],-1,1)
    y=y.reshape(y.shape[0],y.shape[2],y.shape[3])

    yield X,y 
 

if __name__== "__main__":
   X,y=trainingdata_Xy()

   np.save('train_X.npy', X)
   np.save('train_y.npy', y)
   #X.shape
   #y.shape
   #yy=y[0]


   
   
   
