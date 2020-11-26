# -*- coding: utf-8 -*-
import tensorflow as tf

regularization = tf.keras.regularizers.l2

class Conv2D_bn_ac(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay):
        super(Conv2D_bn_ac, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding,
                                           use_bias=self.use_bias,
                                           kernel_regularizer=regularization(weight_decay))
        self.bn = tf.keras.layers.BatchNormalization()
        self.ac = tf.keras.layers.ReLU()
    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.ac(out)
        return out

class Conv2D_bn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay):
        super(Conv2D_bn, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding,
                                           use_bias=self.use_bias,
                                           kernel_regularizer=regularization(weight_decay))
        self.bn = tf.keras.layers.BatchNormalization()
    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

def compact_model(input_shape=(224, 224, 3), weight_decay=1e-4, num_classes=1000):

    h = inputs = tf.keras.Input(input_shape)    # 224 x 224 x 3
    h1 = tf.image.resize(h, [112, 112])
    h2 = tf.image.resize(h1, [56, 56])
    print(h1.shape)
    #################################################################################
    h = Conv2D_bn_ac(64, 3, 1, "same", False, weight_decay)(h)      # 224 x 224 x 64
    h = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h)      # 112 x 112 x 64
    h1 = Conv2D_bn(64, 3, 1, "same", False, weight_decay)(h1)    # 112 x 112 x 64
    print(h1.shape)
    h = tf.keras.layers.ReLU()(h1 + h)  # 112 x 112 x 64 
    h = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h)      # 56 x 56 x 64

    h2 = Conv2D_bn_ac(64, 3, 1, "same", False, weight_decay)(h2)    # 56 x 56 x 64
    h1 = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h1)    # 56 x 56 x 64
    h = tf.keras.layers.ReLU()(h1 + h)      # 56 x 56 x 64
    h1 = tf.keras.layers.ReLU()(h2 + h1)    # 56 x 56 x 64

    h = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h)      # 28 x 28 x 64
    h1 = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h1)    # 28 x 28 x 64
    h2 = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h2)    # 28 x 28 x 64
    h = tf.keras.layers.ReLU()(h1 + h)      # 28 x 28 x 64
    h1 = tf.keras.layers.ReLU()(h1 + h2)    # 28 x 28 x 64
    
    h = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h)      # 14 x 14 x 64
    h1 = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h1)    # 14 x 14 x 64
    h2 = tf.keras.layers.ReLU()(h2)
    h2 = Conv2D_bn(64, 3, 2, "same", False, weight_decay)(h2)    # 14 x 14 x 64
    h = tf.keras.layers.ReLU()(h1 + h)  # 14 x 14 x 64
    h1 = tf.keras.layers.ReLU()(h1 + h2)    # 14 x 14 x 64

    ##########################################      version 1   filter 64   --> MAE: 4.16
    #h = tf.keras.layers.GlobalAveragePooling2D()(h)
    #h1 = tf.keras.layers.GlobalAveragePooling2D()(h1)
    #h2 = tf.keras.layers.GlobalAveragePooling2D()(h2)

    #fully = tf.concat([h, h1, h2], 1)
    #fully = tf.keras.layers.Dense(1024, use_bias=False, kernel_regularizer=regularization(weight_decay))(fully)
    #fully = tf.keras.layers.BatchNormalization()(fully)
    #fully = tf.keras.layers.ReLU()(fully)

    #fully = tf.keras.layers.Dense(num_classes)(fully)
    ##########################################

    ##########################################      version 2   filter 32    --> not complete
    #fully = h + h1 + h2
    #fully = tf.keras.layers.GlobalAveragePooling2D()(fully)

    #fully = tf.keras.layers.Dense(1024, use_bias=False, kernel_regularizer=regularization(weight_decay))(fully)
    #fully = tf.keras.layers.BatchNormalization()(fully)
    #fully = tf.keras.layers.ReLU()(fully)

    #fully = tf.keras.layers.Dense(num_classes)(fully)
    ##########################################
    
    ##########################################      version 2   filter 32--> 3.5, filter 64 --> 3.62
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h1 = tf.keras.layers.GlobalAveragePooling2D()(h1)
    h2 = tf.keras.layers.GlobalAveragePooling2D()(h2)

    h = tf.keras.layers.Dense(512, kernel_regularizer=regularization(weight_decay))(h)

    h1 = tf.keras.layers.Dense(512, kernel_regularizer=regularization(weight_decay))(h1)

    h2 = tf.keras.layers.Dense(512, kernel_regularizer=regularization(weight_decay))(h2)

    fully = tf.keras.layers.ReLU()(h + h1 + h2)

    fully = tf.keras.layers.Dense(num_classes)(fully)
    ##########################################

    #################################################################################

    return tf.keras.Model(inputs=inputs, outputs=fully)