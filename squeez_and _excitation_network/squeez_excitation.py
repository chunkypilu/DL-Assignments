from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
   
    original = input
    channel_axis =-1
    filters = original._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se_ = GlobalAveragePooling2D()(original)
    se_ = Reshape(se_shape)(se_)
    se_ = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_)
    se_ = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_)

    x = multiply([original, se_])
    return x
