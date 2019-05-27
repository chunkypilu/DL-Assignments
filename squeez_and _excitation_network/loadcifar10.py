#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:31:31 2018

@author: priyank
"""

import os
import urllib.request
import tarfile
import pickle
from keras.utils import np_utils
import numpy as np
from keras.utils import np_utils


# dataset path
#home = os.path.expanduser('~')
#data_path = os.path.join(home, "data/CIFAR-10/")
#data_path = os.path.join(home, "/home/priyank/data/CIFAR-10/")

data_path ="/home/ee/mtech/eet162639/data/CIFAR-10/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# CIFAR-10 constants
img_size = 32
img_channels = 3
nb_classes = 10
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 5
images_per_file = 10000
# number of all the images in the training dataset
nb_images_train = nb_files_train * images_per_file


def download_and_extract_cifar():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
        file_path, _ = urllib.request.urlretrieve(url=data_url,
                                                  filename=os.path.join(data_path, 'cifar-10-python.tar.gz'))
        print('\nExtracting... ', end='')
        tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
        print('done')
    else:
        print("Data has already been downloaded and unpacked.")





def load_data(file_name):
    file_path = os.path.join(data_path, "cifar-10-batches-py/", file_name)
    
    print('Loading ' + file_name)
    with open(file_path, mode='rb') as file:    
        data = pickle.load(file)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    
    images = raw_images.reshape([-1, img_channels, img_size, img_size])    
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)
    
    return images, cls

def load_training_data():    
    # pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[nb_images_train, img_size, img_size, img_channels], 
                      dtype=int)
    cls = np.zeros(shape=[nb_images_train], dtype=int)
    
    begin = 0
    for i in range(nb_files_train):
        images_batch, cls_batch = load_data(file_name="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
        
    return images, np_utils.to_categorical(cls, nb_classes)

def load_test_data():
    images, cls = load_data(file_name="test_batch")
    
    return images, np_utils.to_categorical(cls, nb_classes)

def load_cifar():
    X_train, Y_train = load_training_data()
    X_test, Y_test = load_test_data()
    
    return X_train, Y_train, X_test, Y_test


