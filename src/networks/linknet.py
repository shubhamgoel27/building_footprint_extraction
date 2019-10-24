"""
https://www.kaggle.com/kmader/keras-linknet
https://github.com/okotaku/kaggle_dsbowl/blob/master/model/linknet.py
"""

from keras.models import Model
from keras.layers import Input, Conv2D, Deconv2D, MaxPool2D, concatenate, AvgPool2D
from keras.layers import BatchNormalization, Activation
from keras.layers.core import Dropout
from keras import backend as K
from keras.regularizers import l2
from keras.layers import add


s_c2 = lambda fc, k, s = 1, activation='elu', **kwargs: Conv2D(fc, kernel_size = (k,k), strides= (s,s),
                                       padding = 'same', activation = activation,
                                       **kwargs)


s_d2 = lambda fc, k, s = 1, activation='elu', **kwargs: Deconv2D(fc, kernel_size=(k,k), strides=(s,s), 
                                                       padding = 'same', activation=activation,
                                                       **kwargs)


c2 = lambda fc, k, s = 1, **kwargs: lambda x: Activation('elu')(BatchNormalization()(
    Conv2D(fc, kernel_size = (k,k), strides= (s,s),
           padding = 'same', activation = 'linear', **kwargs)(x)))


d2 = lambda fc, k, s = 1, **kwargs: lambda x: Activation('elu')(BatchNormalization()(
    Deconv2D(fc, kernel_size=(k,k), strides=(s,s), 
             padding = 'same', activation='linear', **kwargs)(x)))

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def enc_block(m, n):
    def block_func(x):
        cx = c2(n, 3)(c2(n, 3, 2)(x))
        cs1 = concatenate([AvgPool2D((2,2))(x), 
                           cx])
        cs2 = c2(n, 3)(c2(n, 3)(cs1))
        return concatenate([cs2, cs1])
    return block_func


def dec_block(m, n):
    def block_func(x):
        cx1 = c2(m//4, 1)(x)
        cx2 = d2(m//4, 3, 2)(cx1)
        return Dropout(0.1)(c2(n, 1)(cx2))
    return block_func


def build(size=256, chs=3, summary=False):
    
    start_in = Input((size, size, chs), name = 'Input')
    in_filt = c2(64, 7, 2)(start_in)
    in_mp = MaxPool2D((3,3), strides = (2,2), padding = 'same')(in_filt)

    enc1 = enc_block(64, 64)(in_mp)
    enc2 = enc_block(64, 128)(enc1)

    dec2 = dec_block(64, 128)(enc2)
    dec2_cat = _shortcut(enc1, dec2)
    dec1 = dec_block(64, 64)(dec2_cat)

    last_out = _shortcut(dec1, in_mp)
    
    out_upconv = d2(32, 3, 2)(last_out)
    out_conv = c2(32, 3)(out_upconv)
    out = s_d2(1, 2, 2, activation = 'sigmoid')(out_conv)

    model = Model(inputs = [start_in], outputs = [out]) #Trainable params: 1,151,297

    return model