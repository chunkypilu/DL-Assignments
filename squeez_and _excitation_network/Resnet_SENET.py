
import os
import numpy as np
from keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import multiply
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from squeez_excitation import squeeze_excite_block
from loadcifar10  import *


def SEResNet_model(input_shape=None,
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,weight_decay=1e-4,
             include_top=True,
             pooling=None,
             classes=1000):
   
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),include_top=False)
    
    img_input = Input(shape=input_shape)
    x = func_create_seresnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, weight_decay, pooling)

    inputs = img_input
    model = Model(inputs, x, name='resnet')
    return model




def func_resnet_block(input, filters, k=1, strides=(1, 1)):
    shortcut = input
    
    channel_axis = -1
    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or shortcut._keras_shape[channel_axis] != filters * k:
        shortcut = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, shortcut])
    return m



def func_create_seresnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, weight_decay, pooling):
    channel_axis = -1
    N = list(depth)
    
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    
    for i in range(N[0]):
            x = func_resnet_block(x, filters[0], width)

    
    for k in range(1, len(N)):
         x = func_resnet_block(x, filters[k], width, strides=(2, 2))

         for i in range(N[k] - 1):
                x = func_resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x


if __name__ == '__main__':
   
    x_train, y_train,x_test,y_test=load_cifar() 
#normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
 




      
#define input format
    
    
    input_image = Input(shape = (32, 32, 3))
    model=SEResNet_model(input_shape=(32, 32, 3),
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,
             weight_decay=1e-4,
             include_top=True,
             
             pooling='avg',
             classes=10)
    
    
    epochs = 30
    '''
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.8, decay=decay, nesterov=False)
    '''
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    hist=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    acc = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%",(acc[1])*100)

    
    acc = model.evaluate(x_unseen, y_unseen, verbose=0)
    print("\n\nAccuracy(unseen): %.2f%",(acc[1])*100)
    
    model.save('seresnet.h5')

########################################################################################33

    
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
    fig.savefig('/home/ee/mtech/eet162639/loss_30.png')


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
    fig1.savefig('/home/ee/mtech/eet162639/acc_30.png')

        
