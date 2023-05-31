# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:34:00 2023

@author: Olive
"""

import pdb
import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import (MaxPooling3D,AveragePooling3D,
                                     BatchNormalization,Dense,Flatten,
                                     Conv2D,Conv1D,Add)

def _cstm(input):
    # input.shape = (None,32,224,224,3) N,T,H,W,C
    x = tf.keras.layers.Reshape((-1,input.shape[4] ,input.shape[1]))(input)# NHW,C,T fuse temporal info
    x = _bn_relu(x)
    x = Conv1D(filters=input.shape[1], kernel_size=3,strides=1, padding='same')(x)
    
    x = tf.keras.layers.Reshape((input.shape[1],
                                 input.shape[2],
                                 input.shape[3],
                                 input.shape[4]))(x) # N,T,H,W,C
    x = _bn_relu(x)
    x = Conv2D(filters=input.shape[4], kernel_size=3,strides=1, padding='same')(x)
    
    return x # N,T,H,W,C
    
def _cmm(input):#N,T,H,W,C
    if input.shape[4]>=16*2:
        filters = input.shape[4]/16
    else:
        filters = input.shape[4]
    x = _bn_relu(input)    
    x = Conv2D(filters=filters, kernel_size=1,strides=1, padding='same')(x)
    ''' Padding at starting and ending T '''
    x_padding = tf.keras.layers.ZeroPadding3D(padding=((2, 2), (0, 0),(0,0)))(x)
    f1 = x
    f2 = x_padding[:,3:-1,...]#None,T+1,...
    f2 = _bn_relu(f2)
    f2 = Conv2D(filters=filters, kernel_size=3,strides=1, padding='same')(f2)
    f3 = x_padding[:,4:,...]#None,T+2,...
    f3 = _bn_relu(f3)
    f3= Conv2D(filters=filters, kernel_size=3,strides=1, padding='same')(f3)
    
    x1 = tf.keras.layers.Subtract()([f1, f2])
    x2 = tf.keras.layers.Subtract()([f2, f3])
    x = tf.keras.layers.Concatenate(axis=-1)([x1,x2])
    x = _bn_relu(x)
    x = Conv2D(filters=input.shape[4], kernel_size=1,strides=1, padding='same')(x)
    return x # N,T,H,W,C
    
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",  tf.keras.regularizers.l2(1.e-4))

    def f(input):
        # pdb.set_trace()
        x = _bn_relu(input)
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(x)
        '''--- previous resnet for downsampling--- '''
        # pdb.set_trace()
        residual = x
        # print(residual.shape)
        x = _bn_relu(x)
        x = Conv2D(filters=filters, kernel_size=1,
                      strides=1, padding=padding)(x)
        x1 = _cstm(x)
        x2 = _cmm(x)
        x = tf.keras.layers.Add()([x1, x2])
        x = _bn_relu(x)
        x = Conv2D(filters=filters, kernel_size=1,
                      strides=1, padding=padding)(x)
        # print(x.shape)
        return Add()([residual,x])

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer= tf.keras.regularizers.l2(0.0001))(input)
        shortcut = BatchNormalization(axis=-1)(shortcut)

    return Add()([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            # pdb.set_trace()
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        # pdb.set_trace()
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer= tf.keras.regularizers.l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer= tf.keras.regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global TEM_AXIS
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if tf.keras.backend.image_data_format() == 'channels_last':#'channels_last'
        TEM_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        TEM_AXIS = 2
        ROW_AXIS = 3
        COL_AXIS = 4
    

# def _get_block(identifier):
#     pdb.set_trace()
#     if isinstance(identifier, six.string_types):
        
#         res = globals().get(identifier)
#         if not res:
#             raise ValueError('Invalid {}'.format(identifier))
#         return res
#     return identifier

class STMBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple ( nb_rows, nb_cols,nb_channels,)")
        

        # Permute dimension order if necessary
        if tf.keras.backend.image_data_format() == 'channels_last':
            input_shape = (input_shape[0],input_shape[1], input_shape[2], input_shape[3])



        input = tf.keras.Input(shape=input_shape)
        # pdb.set_trace()
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling3D(pool_size=(1, 3,3), strides=(1,2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = tf.keras.backend.int_shape(block)
        pool2 = AveragePooling3D(pool_size=(block_shape[TEM_AXIS],block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                  strides=(1,1,1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="sigmoid")(flatten1)

        model = tf.keras.Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return STMBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return STMBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return STMBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return STMBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return STMBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
    
# Resnet = STMBuilder()
# INPUT_SHAPE = (32,224,224,3)
# model = Resnet.build_resnet_18(input_shape=INPUT_SHAPE, num_outputs=1)
# model.summary(line_length =120)
# tf.keras.utils.plot_model(
#     model,
#     to_file=r'C:\Users\Olive\Desktop\model.png')