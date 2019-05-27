#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:03:56 2018

@author: priyank
"""

import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to input image", required=True)
ap.add_argument("-p", "--pivot-point", help="Pivot point coordinates x, y separated by comma (,)", required=True)
ap.add_argument("-s", "--scale", help="Scale to zoom", type=int, required=True)
args = vars(ap.parse_args())

image_path = args["image"]
x, y = map(int, args["pivot_point"].split(","))
scale = args["scale"]
image1 = cv2.imread(image_path)
image = image1.tolist()

"""
WRITE YOUR CODE HERE
"""
######################################################################
#######################################################################
a=len(image)
b=len(image[0])
num_channels=len(image[0][0])

dct = {}
for i in range(num_channels):
       dct['R_matrix_%s' % i] = []


for i in range(0,a):
       
       dct1 = {}
       for k in range(num_channels):
         dct1['R_internal_%s' % k] = []

        
       for j in range(0,b):
              
        for k in range(num_channels):
         dct1['R_internal_%s' % k].append(image[i][j][k])
         
       for k in range(num_channels):
              dct['R_matrix_%s' % k].append(dct1['R_internal_%s' % k])
       
if(x>b-1 or y>a-1):
  print("Pivot entered is out of bound, please try again !!")
  exit()    

       
a_new=round(a/scale)
b_new=round(b/scale)

a_new_by2=round(a_new/2)
b_new_by2=round(b_new/2)


###########################################################################333
dct3={} 
for k in range(num_channels):
     dct3['R_matrix_shrinked1_%s' % k] = []
 


for i in range(y-a_new_by2,y+a_new_by2):
       
       dct4={} 
       for k in range(num_channels):
         dct4['R_matrix_shrinked_internal_%s' % k] = []

       for j in range(x-b_new_by2,x+b_new_by2):
          try:
           if (i>=0 and j>=0):
              for k in range(num_channels):
                dct4['R_matrix_shrinked_internal_%s' % k].append(dct['R_matrix_%s' % k][i][j])
           else:
                for k in range(num_channels):
                 dct4['R_matrix_shrinked_internal_%s' % k].append(255)  
                  
          except:
               for k in range(num_channels):
                 dct4['R_matrix_shrinked_internal_%s' % k].append(255)  
       for k in range(num_channels):
        dct3['R_matrix_shrinked1_%s' % k].append(dct4['R_matrix_shrinked_internal_%s' % k])

####For handing  corner cases###########################

dct5={}
for k in range(num_channels):
 dct5['R_matrix_shrinked_%s' % k]=dct3['R_matrix_shrinked1_%s' % k]      
######## K- times zooming algorithm implementation###################
       
m=len(dct5['R_matrix_shrinked_0'])
n=len(dct5['R_matrix_shrinked_0'][0])       
       
#matrix_shrinked_B(m, (n-1)*scale+1)

dct6={}
for k in range(num_channels):
  dct6['R_matrix_shrinked_B%s' % k] = []


for i in range(0,m):
       dct7={}
       for k in range(num_channels):
         dct7['R_matrix_shrinked_B_%s' % k] = []

       
       for j in range(0,(n-1)*scale+1):
              for k in range(num_channels):
                dct7['R_matrix_shrinked_B_%s' % k].append(0)

       for k in range(num_channels):
                dct6['R_matrix_shrinked_B%s' % k].append(dct7['R_matrix_shrinked_B_%s' % k])
         


for i in range(0,m):
       j=0
       for k in range(num_channels):
          dct6['R_matrix_shrinked_B%s' % k][i][j]=dct5['R_matrix_shrinked_%s' % k][i][j]    
       for j in range(1,n):
           for k in range(num_channels):
            dct6['R_matrix_shrinked_B%s' % k][i][j*scale]=dct5['R_matrix_shrinked_%s' % k][i][j]    
          
           dct_s={}
           for k in range(num_channels):
            if(dct6['R_matrix_shrinked_B%s' % k][i][j*scale]>dct6['R_matrix_shrinked_B%s' % k][i][(j-1)*scale]) :    
               dct_s['s_%s' % k ]=(dct6['R_matrix_shrinked_B%s' % k][i][j*scale]-dct6['R_matrix_shrinked_B%s' % k][i][(j-1)*scale])/scale
               dct_s['small_%s' % k ]=dct6['R_matrix_shrinked_B%s' % k][i][(j-1)*scale]
            else:
               dct_s['s_%s' % k ]=(dct6['R_matrix_shrinked_B%s' % k][i][(j-1)*scale]-dct6['R_matrix_shrinked_B%s' % k][i][j*scale])/scale    
               dct_s['small_%s' % k ]=dct6['R_matrix_shrinked_B%s' % k][i][j*scale]
          
          
           for l in range(scale-1,0,-1):
            for k in range(num_channels):
             dct6['R_matrix_shrinked_B%s' % k][i][j*scale-l]=round(dct_s['small_%s' % k ]+dct_s['s_%s' % k ])  
             dct_s['small_%s' % k ]=dct6['R_matrix_shrinked_B%s' % k][i][j*scale-l]
                


#matrix_shrinked_C((m-­1)*scale+1 ,(n-1)*scale+1)           
dct7={}             
for k in range(num_channels):
  dct7['R_matrix_shrinked_C%s' % k]  =[]   
                        
y=(m-1)*scale+1
z=(n-1)*scale+1
for i in range(0,y):
       dct8={}             
       for k in range(num_channels):
        dct8['R_matrix_shrinked_C_%s' % k]  =[]   

       for j in range(0,z):
              for k in range(num_channels):
                dct8['R_matrix_shrinked_C_%s' % k].append(0);
       for k in range(num_channels):
          dct7['R_matrix_shrinked_C%s' % k].append(dct8['R_matrix_shrinked_C_%s' % k])        



for i in range(0,z):
       j=0
       
       for k in range(num_channels):
          dct7['R_matrix_shrinked_C%s' % k][j][i]=dct6['R_matrix_shrinked_B%s' % k][j][i]    
       for j in range(1,m):
              for k in range(num_channels):
               dct7['R_matrix_shrinked_C%s' % k][j*scale][i]=dct6['R_matrix_shrinked_B%s' % k][j][i]
              
               
              dct_s1={}
              for k in range(num_channels):
               if(dct7['R_matrix_shrinked_C%s' % k][j*scale][i]>dct7['R_matrix_shrinked_C%s' % k][(j-1)*scale][i]):      
                  dct_s1['s1_%s' % k]= (dct7['R_matrix_shrinked_C%s' % k][j*scale][i]-dct7['R_matrix_shrinked_C%s' % k][(j-1)*scale][i])/scale
                  dct_s1['s1mall_%s' % k]=dct7['R_matrix_shrinked_C%s' % k][(j-1)*scale][i]
               else:
                  dct_s1['s1_%s' % k]= (dct7['R_matrix_shrinked_C%s' % k][(j-1)*scale][i]-dct7['R_matrix_shrinked_C%s' % k][j*scale][i])/scale     
                  dct_s1['s1mall_%s' % k]= dct7['R_matrix_shrinked_C%s' % k][j*scale][i]
              for l in range(scale-1,0,-1):
                  for k in range(num_channels):
                         dct7['R_matrix_shrinked_C%s' % k][(j*scale)-l][i]=round(dct_s1['s1mall_%s' % k]+dct_s1['s1_%s' % k])
                         dct_s1['s1mall_%s' % k]=dct7['R_matrix_shrinked_C%s' % k][(j*scale)-l][i]

#if(len(dct7['R_matrix_shrinked_C0'])<a)

a_=len(dct7['R_matrix_shrinked_C0'])

dict_last={}
if(a_<a):
       
   for k in range(num_channels):    
        dict_last['last_%s' % k]=dct7['R_matrix_shrinked_C%s' % k][-1] 
   for i in range(a-a_):
       for k in range(num_channels):
               dct7['R_matrix_shrinked_C%s' % k].append(dict_last['last_%s' % k])
              
b_=len(dct7['R_matrix_shrinked_C0'][0])
if(b_<b):
     for i in range(len(dct7['R_matrix_shrinked_C0'])): 
        for k in range(num_channels):
               dict_last['lastc_%s' % k]=dct7['R_matrix_shrinked_C%s' % k][i][-1]
               
        for j in range(b-b_):
              for k in range(num_channels):
                     dct7['R_matrix_shrinked_C%s' % k][i].append(dict_last['lastc_%s' % k])
                     

    


a_=len(dct7['R_matrix_shrinked_C0'])
b_=len(dct7['R_matrix_shrinked_C0'][0])

zoomed_image=[]
for i in range(0,a_):
       Final_matrix_shrinked_C_internal=[]
       for j in range (0,b_):
              llist=[]
              for k in range(num_channels):
                  llist.append(dct7['R_matrix_shrinked_C%s' % k][i][j])   
              
              Final_matrix_shrinked_C_internal.append(llist)
       zoomed_image.append(Final_matrix_shrinked_C_internal)
 
       
#################################################################################
################################################################################       

cv2.imwrite("zoomed_image.png", np.array(zoomed_image, dtype="uint8"))


