"""
orignal: https://github.com/okotaku/kaggle_dsbowl/blob/master/model/pspnet.py

similar: https://github.com/ykamikawa/tf-keras-PSPNet/blob/master/model.py
"""

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Reshape, Permute, Dense, Activation
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import multiply, add, concatenate
from keras.engine.topology import Layer
from keras.engine import InputSpec


def build(size = 512, chs = 3, summary=False):
        
    n_labels=1
    output_stride=16
    num_blocks=4
    
    levels=[6,3,2,1]
    
    use_se=True 
    output_mode='sigmoid'
    upsample_type='duc'

    input_shape = (size, size, chs)    
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, strides=(1, 1), use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, use_se=use_se)

    x = conv_block(x, 3, [128, 128, 512], stage=3, use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, use_se=use_se)

    if output_stride == 8:
        rate_scale = 2
    elif output_stride == 16:
        rate_scale = 1

    x = conv_block(x, 3, [256, 256, 1024], stage=4, dilation_rate=1*rate_scale,
                   multigrid=[1,1,1], use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       dilation_rate=1*rate_scale, multigrid=[1, 1, 1],
                       use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       dilation_rate=1*rate_scale, multigrid=[1, 1, 1],
                       use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       dilation_rate=1*rate_scale, multigrid=[1, 1, 1],
                       use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       dilation_rate=1*rate_scale, multigrid=[1, 1, 1],
                       use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,
                       dilation_rate=1*rate_scale, multigrid=[1, 1, 1],
                       use_se=use_se)

    init_rate = 2
    for block in range(4, num_blocks+1):
        x = conv_block(x, 3, [512, 512, 2048], stage=5,
                       dilation_rate=init_rate*rate_scale,
                       multigrid=[1, 1, 1], use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5,
                           dilation_rate=init_rate*rate_scale,
                           multigrid=[1, 1, 1], use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5,
                           dilation_rate=init_rate*rate_scale,
                           multigrid=[1, 1, 1], use_se=use_se)
        init_rate *= 2

    x = pyramid_pooling_module(x, 512, input_shape, output_stride, levels)

    if upsample_type == 'duc':
        x = duc(x, factor=output_stride,    
                output_shape=(input_shape[0], input_shape[1], n_labels))
        out = Conv2D(n_labels, (1, 1), padding='same',
                     kernel_initializer="he_normal")(x)

    elif upsample_type == 'bilinear':
        x = Conv2D(n_labels, (1, 1), padding='same',
                   kernel_initializer="he_normal")(x)
        out = BilinearUpSampling2D((n_labels, input_shape[0], input_shape[1]),
                                   factor=output_stride)(x)

    out = Activation(output_mode)(out)

    model = Model(inputs=img_input, outputs=out)

    return model


def conv_block(input_tensor, kernel_size, filters, stage, strides=(2, 2),
               dilation_rate=1, multigrid=[1,2,1], use_se=True):
    
    filters1, filters2, filters3 = filters

    if dilation_rate > 1:
        strides = (1, 1)
    else:
        multigrid = [1, 1, 1]

    x = Conv2D(filters1, (1, 1), strides=strides,
               dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1)
        x = multiply([x, se])
    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def _squeeze_excite_block(init, filters, k=1):
    se_shape = (1, 1, filters * k)

    se = GlobalAveragePooling2D()(init)
    se = Dense((filters * k) // 16, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters * k, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    return se


def identity_block(input_tensor, kernel_size, filters, stage, dilation_rate=1,
                   multigrid=[1, 2, 1], use_se=True):
    filters1, filters2, filters3 = filters

    if dilation_rate < 2:
        multigrid = [1, 1, 1]

    x = Conv2D(filters1, (1, 1),
               dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization()(x)

    # stage 5 after
    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1)
        x = multiply([x, se])
    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def interp_block(x, num_filters, input_shape, output_stride, level):
    feature_map_shape = (input_shape[0]/output_stride,
                         input_shape[1]/output_stride)

    if output_stride == 16:
        scale = 5
    elif output_stride == 8:
        scale = 10

    kernel = (level*scale, level*scale)
    strides = (level*scale, level*scale)
    global_feat = AveragePooling2D(kernel, strides=strides)(x)
    global_feat = Conv2D(num_filters, (1, 1), padding='same',
                         kernel_initializer="he_normal")(global_feat)
    global_feat = BatchNormalization()(global_feat)
    global_feat = Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)

    return global_feat


def pyramid_pooling_module(x, num_filters, input_shape, output_stride, levels):
    pyramid_pooling_blocks = [x]
    for level in levels:
        pyramid_pooling_blocks.append(interp_block(x, num_filters, input_shape,
                                                   output_stride, level))

    y = concatenate(pyramid_pooling_blocks)
    y = Conv2D(num_filters, (3, 3), padding='same',
               kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    return y


def duc(x, factor=8, output_shape=(512,512,1)):
    H,W,c,r = output_shape[0],output_shape[1],output_shape[2],factor
    h = H/r
    w = W/r
    x = Conv2D(c*r*r,
            (3, 3),
            padding='same',
            name='conv_duc_%s'%factor)(x)
    x = BatchNormalization(name='bn_duc_%s'%factor)(x)
    x = Activation('relu')(x)
    x = Permute((3,1,2))(x)
    x = Reshape((c,r,r,h,w))(x)
    x = Permute((1,4,2,5,3))(x)
    x = Reshape((c,H,W))(x)
    x = Permute((2,3,1))(x)

    return x


class BilinearUpSampling2D(Layer):
    def __init__(self, target_shape=None, factor=None, data_format=None,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}

        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return K.resize_images(inputs, self.factor, self.factor, self.data_format)

    def get_config(self):
        config = {'target_shape': self.target_shape,
                  'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [int(new_height), int(new_width)], align_corners=True)
    return resized