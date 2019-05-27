#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:21:05 2018

@author: emblab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:25:18 2018

@author: priyank
"""
#
import os
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import pickle

## I am making up a dictionary here to show you how this works...
## Because I want to store this outside of this single run, it could be that this
## dictionary is dynamic and user based - so persistance beyond this run has
## meaning for me.  
#myMadeUpDictionary = {"one": "banana", "two": "banana", "three": "banana", "four": "no-more"}

#with open("mySavedDict.txt", "wb") as myFile:
#   pickle.dump(myMadeUpDictionary, myFile)









PATH='/home/emblab/Desktop/Assignment_3/DL_ASS_3_data_toy'

#def read_of_words( ):

docs=[]   
folders=os.listdir(PATH)       
for i in folders:
   PATH1=PATH+'/'+i
   folders1=os.listdir(PATH1)
   for j in folders1:
                PATH2=PATH1+'/'+j
                folders2=os.listdir(PATH2)
                for k in folders2:
                      if 'txt' in k:
                             PATH3=PATH2+'/'+k
                             
                             print(k)
                             with open (PATH3, "r") as myfile:
                                   
                               for line in myfile:
                                      #line=line[8:]
                                      print(line)
                                      x=0
                                      for l in range(len(line)):
                                             
                                             if line[l]==' ':
                                               x=x+1
                                             if x==2:
                                                    break       
                                      #line=line.split(" ")[2:]
                                      docs.append(line[l+1:])
                                      
                                      
print(docs)
t = Tokenizer()
print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',t)
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
    
print(t.word_index['me'])

#encoded_docs = t.texts_to_sequences(docs)
#print(encoded_docs)
   





#if __name__== "__main__":
#   print('hi')
#   #print(sum(1,2))
#   read_of_words( )
   
with open("mySavedDict.txt", "wb") as myFile:
   pickle.dump(myMadeUpDictionary, myFile)
