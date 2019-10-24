"""
https://github.com/okotaku/kaggle_dsbowl/blob/master/model/tiramisu56.py
"""

from keras.layers import Input
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

def _denseBlock(x, layers, filters):
    for i in range(layers):
        x = BatchNormalization(gamma_regularizer=l2(0.0001),
                               beta_regularizer=l2(0.0001))(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same',
                   kernel_initializer="he_uniform")(x)
        x = Dropout(0.2)(x)

    return x


def _transitionDown(x, filters):
    x = BatchNormalization(gamma_regularizer=l2(0.0001),
                           beta_regularizer=l2(0.0001))(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_initializer="he_uniform")(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return x


def _transitionUp(x, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same',
                        kernel_initializer="he_uniform")(x)

    return x


def build(size=256, chs=3, summary=False):
    
    #Input
    inp = Input(shape=(size, size, chs))
    
    # Encoder
    x = Conv2D(48, kernel_size=(3, 3), padding='same',
               input_shape=(size, size, chs),
               kernel_initializer="he_uniform",
               kernel_regularizer=l2(0.0001))(inp)

    x = _denseBlock(x, 4, 96) # 4*12 = 48 + 48 = 96
    x = _transitionDown(x, 96)
    x = _denseBlock(x, 4, 144) # 4*12 = 48 + 96 = 144
    x = _transitionDown(x, 144)
    x = _denseBlock(x, 4, 192) # 4*12 = 48 + 144 = 192
    x = _transitionDown(x, 192)
    x = _denseBlock(x, 4, 240)# 4*12 = 48 + 192 = 240
    x = _transitionDown(x, 240)
    x = _denseBlock(x, 4, 288) # 4*12 = 48 + 288 = 336
    x = _transitionDown(x, 288)
    
    #Center
    x = _denseBlock(x, 15, 336) # 4 * 12 = 48 + 288 = 336

    #Decoder
    x = _transitionUp(x, 384)  # m = 288 + 4x12 + 4x12 = 384.
    x = _denseBlock(x, 4, 384)

    x = _transitionUp(x, 336) #m = 240 + 4x12 + 4x12 = 336
    x = _denseBlock(x, 4, 336)

    x = _transitionUp(x, 288) # m = 192 + 4x12 + 4x12 = 288
    x = _denseBlock(x, 4, 288)

    x = _transitionUp(x, 240) # m = 144 + 4x12 + 4x12 = 240
    x = _denseBlock(x, 4, 240)

    x = _transitionUp(x, 192) # m = 96 + 4x12 + 4x12 = 192
    x = _denseBlock(x, 4, 192)

    #Output
    x = Conv2D(1, kernel_size=(1, 1), padding='same',
              kernel_initializer="he_uniform",
              kernel_regularizer=l2(0.0001))(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inp, outputs=x) #Trainable params: 42,392,113
    
    if summary:
        model.summary()

    return model


