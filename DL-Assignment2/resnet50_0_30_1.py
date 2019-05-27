import numpy as np
np.random.seed(2017)

import os
import time
from resnet50_ import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


# Loading the training data
PATH = os.getcwd()
# Define data path
data_path =  '/home/ee/mtech/eet162639/data_crop'
data_dir_list = os.listdir(data_path)
data_dir_list=sorted(data_dir_list)


img_data_list=[]

a=[]
count=0


for dataset in data_dir_list:
         img_list=os.listdir(data_path+'/'+ dataset)
       	 count+=len(img_list)
         a.append(count)
       	 print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
       	 for img in img_list:
            try:      
              img_path = data_path + '/'+ dataset + '/'+ img
              img = image.load_img(img_path, target_size=(224,224))
              x = image.img_to_array(img)
              x = np.expand_dims(x, axis=0)
              x = preprocess_input(x)
              print('Input image shape:', x.shape)    
              img_data_list.append(x)
            except:
               pass    
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 7
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:a[0]]=0
labels[a[0]:a[1]]=1
labels[a[1]:a[2]]=2
labels[a[2]:a[3]]=3
labels[a[3]:a[4]]=4
labels[a[4]:a[5]]=5
labels[a[5]:a[6]]=6


names = ['atul','flute','sadhguru','sandeep','saurabh','shailendra','xnoise']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

###########################################################################################################################
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

custom_resnet_model.save('/home/ee/mtech/eet162639/resnet50_0_30_1.h5')


###########################################################################################################################
###########################################################################################
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(30)

fig=plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
fig.savefig('/home/ee/mtech/eet162639/resnet50_0_30_11.png')


fig1=plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
fig1.savefig('/home/ee/mtech/eet162639/resnet50_0_30_12.png')


