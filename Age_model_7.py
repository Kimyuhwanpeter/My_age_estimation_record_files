# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

regularization = tf.keras.regularizers.l2

class Conv2D_bn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, dilation_rate=1):
        super(Conv2D_bn, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            use_bias=use_bias,
                                            kernel_regularizer=regularization(weight_decay))
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, input):
        h = self.conv1(input)
        h = self.bn(h)
        return h

class depth_wise_Conv2D_bn(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, padding, use_bias, weight_decay):
        super(depth_wise_Conv2D_bn, self).__init__()
        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                     strides=strides,
                                                     padding=padding,
                                                     use_bias=use_bias,
                                                     depthwise_regularizer=regularization(weight_decay))
        self.bn = tf.keras.layers.BatchNormalization()
    def call(self, inputs):
        h = self.conv1(inputs)
        h = self.bn(h)
        return h


def Recurrent_CNN(input_shape=(224, 224, 3), num_classes=100, weight_decay=1e-4):
    
    # 224 x 224 x 3
    h = inputs = tf.keras.Input(input_shape)

    # 224 x 224 x 3
    h1 = Conv2D_bn(32, 3, 1, 'same', False, weight_decay)(h)        # 이 부분부터 내일 디버깅을 해보고 shape를 맞춰보자!!
    
    max1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h1)   # 112 x 112 x 32
    avg1 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(h1)   # 112 x 112 x 32
    h1 = tf.keras.layers.ReLU()(h1)
    h1 = Conv2D_bn(32, 3, 2, "same", False, weight_decay)(h1)   # 112 x 112 x 32
    h1 = tf.keras.layers.ReLU()(h1)
    
    ############################
    h = (max1 + avg1 + h1) / 3.
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)                           # 112 x 112 x 32
    ############################

    ############################
    h = Conv2D_bn(32, 3, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(32, 1, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)                           # 56 x 56 x 64
    ############################

    ############################
    max2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h)    # 28 x 28 x 64
    avg2 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(h)    # 28 x 28 x 64
    h1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         depthwise_regularizer=regularization(weight_decay))(h) # 28 x 28 x 64
    ############################
    
    ############################
    h = (max2 + avg2 + h1) / 3.
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)                           # 28 x 28 x 64
    ############################
    
    ############################
    h = Conv2D_bn(64, 3, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(64, 1, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(128, 3, 2, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)                           # 14 x 14 x 128
    ############################

    ############################
    max3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h)    # 7 x 7 x 128
    avg3 = tf.keras.layers.AvgPool2D(pool_size=(2,2))(h)    # 7 x 7 x 128
    h1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         use_bias=False,
                                         depthwise_regularizer=regularization(weight_decay))(h) # 7 x 7 x 128
    ############################

    ############################
    h = (max3 + avg3 + h1) / 3.
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)                           # 7 x 7 x 128
    ############################

    ############################
    h = Conv2D_bn(128, 3, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(128, 1, 1, "same", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)
    h = Conv2D_bn(256, 3, 2, "valid", False, weight_decay)(h)
    h = tf.keras.layers.ReLU()(h)                           # 1 x 1 x 256
    ############################

    h = Conv2D_bn(256, 3, 1, "valid", False, weight_decay)(h)

    h = tf.squeeze(h, [1,2])

    #h = tf.keras.layers.Dense(512, kernel_regularizer=regularization(weight_decay))(h)     # 4.01
    h = tf.keras.layers.Dense(num_classes)(h)

    # 현재 이 모델은 의식의 흐름대로 코딩해본것이라 디버깅을 필수적으로 해야한다!!

    return tf.keras.Model(inputs=inputs, outputs=h)
