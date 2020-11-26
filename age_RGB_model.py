# -*- coding: utf-8 -*-
import tensorflow as tf

def split_channels(inputs):
    channels = inputs.shape[-1]
    h = channels // 3
    return inputs[:,:,:,0:h], inputs[:,:,:,h:h*2], inputs[:,:,:,h*2:h*3]

def branch_model(input, weight_decay):
    # input shape --> batch x 224 x 224 x 1
    input = tf.expand_dims(input, 3)
    x = tf.keras.layers.Conv2D(filters=9,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)       # batch x 112 x 112 x 9

    x1, x2, x3 = split_channels(x)      # each --> batch x 112 x 112 x 3

    x1 = tf.keras.layers.Conv2D(filters=27,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)     # batch x 56 x 56 x 27 (x1)

    x2 = tf.keras.layers.Conv2D(filters=27,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)     # batch x 56 x 56 x 27 (x2)

    x3 = tf.keras.layers.Conv2D(filters=27,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)     # batch x 56 x 56 x 27 (x3)

    x1_1, x1_2, x1_3 = split_channels(x1)   # each --> batch x 64 x 64 x 9
    x2_1, x2_2, x2_3 = split_channels(x2)   # each --> batch x 64 x 64 x 9
    x3_1, x3_2, x3_3 = split_channels(x3)   # each --> batch x 64 x 64 x 9

    x1_1 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1)
    x1_1 = tf.keras.layers.BatchNormalization()(x1_1)
    x1_1 = tf.keras.layers.ReLU()(x1_1)     # batch x 28 x 28 x 81

    x1_2 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2)
    x1_2 = tf.keras.layers.BatchNormalization()(x1_2)
    x1_2 = tf.keras.layers.ReLU()(x1_2)     # batch x 28 x 28 x 81

    x1_3 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3)
    x1_3 = tf.keras.layers.BatchNormalization()(x1_3)
    x1_3 = tf.keras.layers.ReLU()(x1_3)     # batch x 28 x 28 x 81

    x2_1 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1)
    x2_1 = tf.keras.layers.BatchNormalization()(x2_1)
    x2_1 = tf.keras.layers.ReLU()(x2_1)     # batch x 28 x 28 x 81

    x2_2 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2)
    x2_2 = tf.keras.layers.BatchNormalization()(x2_2)
    x2_2 = tf.keras.layers.ReLU()(x2_2)     # batch x 28 x 28 x 81

    x2_3 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3)
    x2_3 = tf.keras.layers.BatchNormalization()(x2_3)
    x2_3 = tf.keras.layers.ReLU()(x2_3)     # batch x 28 x 28 x 81

    x3_1 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1)
    x3_1 = tf.keras.layers.BatchNormalization()(x3_1)
    x3_1 = tf.keras.layers.ReLU()(x3_1)     # batch x 28 x 28 x 81

    x3_2 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2)
    x3_2 = tf.keras.layers.BatchNormalization()(x3_2)
    x3_2 = tf.keras.layers.ReLU()(x3_2)     # batch x 28 x 28 x 81

    x3_3 = tf.keras.layers.Conv2D(filters=81,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3)
    x3_3 = tf.keras.layers.BatchNormalization()(x3_3)
    x3_3 = tf.keras.layers.ReLU()(x3_3)     # batch x 28 x 28 x 81

    x1_1_1, x1_1_2, x1_1_3 = split_channels(x1_1)   # each --> batch x 28 x 28 x 27
    x1_2_1, x1_2_2, x1_2_3 = split_channels(x1_2)   # each --> batch x 28 x 28 x 27
    x1_3_1, x1_3_2, x1_3_3 = split_channels(x1_3)   # each --> batch x 28 x 28 x 27

    x2_1_1, x2_1_2, x2_1_3 = split_channels(x2_1)   # each --> batch x 28 x 28 x 27
    x2_2_1, x2_2_2, x2_2_3 = split_channels(x2_2)   # each --> batch x 28 x 28 x 27
    x2_3_1, x2_3_2, x2_3_3 = split_channels(x2_3)   # each --> batch x 28 x 28 x 27

    x3_1_1, x3_1_2, x3_1_3 = split_channels(x3_1)   # each --> batch x 28 x 28 x 27
    x3_2_1, x3_2_2, x3_2_3 = split_channels(x3_2)   # each --> batch x 28 x 28 x 27
    x3_3_1, x3_3_2, x3_3_3 = split_channels(x3_3)   # each --> batch x 28 x 28 x 27

    x1_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_1)
    x1_1_1 = tf.keras.layers.BatchNormalization()(x1_1_1)
    x1_1_1 = tf.keras.layers.ReLU()(x1_1_1)     # batch x 14 x 14 x 243
    x1_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_1)
    x1_1_1 = tf.keras.layers.BatchNormalization()(x1_1_1)
    x1_1_1 = tf.keras.layers.ReLU()(x1_1_1)     # batch x 7 x 7 x 243

    x1_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_2)
    x1_1_2 = tf.keras.layers.BatchNormalization()(x1_1_2)
    x1_1_2 = tf.keras.layers.ReLU()(x1_1_2)     # batch x 14 x 14 x 243
    x1_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_2)
    x1_1_2 = tf.keras.layers.BatchNormalization()(x1_1_2)
    x1_1_2 = tf.keras.layers.ReLU()(x1_1_2)     # batch x 7 x 7 x 243

    x1_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_3)
    x1_1_3 = tf.keras.layers.BatchNormalization()(x1_1_3)
    x1_1_3 = tf.keras.layers.ReLU()(x1_1_3)     # batch x 14 x 14 x 243
    x1_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_1_3)
    x1_1_3 = tf.keras.layers.BatchNormalization()(x1_1_3)
    x1_1_3 = tf.keras.layers.ReLU()(x1_1_3)     # batch x 7 x 7 x 243

    x1_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_1)
    x1_2_1 = tf.keras.layers.BatchNormalization()(x1_2_1)
    x1_2_1 = tf.keras.layers.ReLU()(x1_2_1)     # batch x 14 x 14 x 243
    x1_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_1)
    x1_2_1 = tf.keras.layers.BatchNormalization()(x1_2_1)
    x1_2_1 = tf.keras.layers.ReLU()(x1_2_1)     # batch x 7 x 7 x 243

    x1_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_2)
    x1_2_2 = tf.keras.layers.BatchNormalization()(x1_2_2)
    x1_2_2 = tf.keras.layers.ReLU()(x1_2_2)     # batch x 14 x 14 x 243
    x1_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_2)
    x1_2_2 = tf.keras.layers.BatchNormalization()(x1_2_2)
    x1_2_2 = tf.keras.layers.ReLU()(x1_2_2)     # batch x 7 x 7 x 243

    x1_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_3)
    x1_2_3 = tf.keras.layers.BatchNormalization()(x1_2_3)
    x1_2_3 = tf.keras.layers.ReLU()(x1_2_3)     # batch x 14 x 14 x 243
    x1_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_2_3)
    x1_2_3 = tf.keras.layers.BatchNormalization()(x1_2_3)
    x1_2_3 = tf.keras.layers.ReLU()(x1_2_3)     # batch x 7 x 7 x 243

    x1_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_1)
    x1_3_1 = tf.keras.layers.BatchNormalization()(x1_3_1)
    x1_3_1 = tf.keras.layers.ReLU()(x1_3_1)     # batch x 14 x 14 x 243
    x1_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_1)
    x1_3_1 = tf.keras.layers.BatchNormalization()(x1_3_1)
    x1_3_1 = tf.keras.layers.ReLU()(x1_3_1)     # batch x 7 x 7 x 243

    x1_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_2)
    x1_3_2 = tf.keras.layers.BatchNormalization()(x1_3_2)
    x1_3_2 = tf.keras.layers.ReLU()(x1_3_2)     # batch x 14 x 14 x 243
    x1_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_2)
    x1_3_2 = tf.keras.layers.BatchNormalization()(x1_3_2)
    x1_3_2 = tf.keras.layers.ReLU()(x1_3_2)     # batch x 7 x 7 x 243

    x1_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_3)
    x1_3_3 = tf.keras.layers.BatchNormalization()(x1_3_3)
    x1_3_3 = tf.keras.layers.ReLU()(x1_3_3)     # batch x 14 x 14 x 243
    x1_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1_3_3)
    x1_3_3 = tf.keras.layers.BatchNormalization()(x1_3_3)
    x1_3_3 = tf.keras.layers.ReLU()(x1_3_3)     # batch x 7 x 7 x 243

    x2_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_1)
    x2_1_1 = tf.keras.layers.BatchNormalization()(x2_1_1)
    x2_1_1 = tf.keras.layers.ReLU()(x2_1_1)     # batch x 14 x 14 x 243
    x2_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_1)
    x2_1_1 = tf.keras.layers.BatchNormalization()(x2_1_1)
    x2_1_1 = tf.keras.layers.ReLU()(x2_1_1)     # batch x 7 x 7 x 243

    x2_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_2)
    x2_1_2 = tf.keras.layers.BatchNormalization()(x2_1_2)
    x2_1_2 = tf.keras.layers.ReLU()(x2_1_2)     # batch x 14 x 14 x 243
    x2_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_2)
    x2_1_2 = tf.keras.layers.BatchNormalization()(x2_1_2)
    x2_1_2 = tf.keras.layers.ReLU()(x2_1_2)     # batch x 7 x 7 x 243

    x2_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_3)
    x2_1_3 = tf.keras.layers.BatchNormalization()(x2_1_3)
    x2_1_3 = tf.keras.layers.ReLU()(x2_1_3)     # batch x 14 x 14 x 243
    x2_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_1_3)
    x2_1_3 = tf.keras.layers.BatchNormalization()(x2_1_3)
    x2_1_3 = tf.keras.layers.ReLU()(x2_1_3)     # batch x 7 x 7 x 243

    x2_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_1)
    x2_2_1 = tf.keras.layers.BatchNormalization()(x2_2_1)
    x2_2_1 = tf.keras.layers.ReLU()(x2_2_1)     # batch x 14 x 14 x 243
    x2_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_1)
    x2_2_1 = tf.keras.layers.BatchNormalization()(x2_2_1)
    x2_2_1 = tf.keras.layers.ReLU()(x2_2_1)     # batch x 7 x 7 x 243

    x2_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_2)
    x2_2_2 = tf.keras.layers.BatchNormalization()(x2_2_2)
    x2_2_2 = tf.keras.layers.ReLU()(x2_2_2)     # batch x 14 x 14 x 243
    x2_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_2)
    x2_2_2 = tf.keras.layers.BatchNormalization()(x2_2_2)
    x2_2_2 = tf.keras.layers.ReLU()(x2_2_2)     # batch x 7 x 7 x 243

    x2_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_3)
    x2_2_3 = tf.keras.layers.BatchNormalization()(x2_2_3)
    x2_2_3 = tf.keras.layers.ReLU()(x2_2_3)     # batch x 14 x 14 x 243
    x2_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_2_3)
    x2_2_3 = tf.keras.layers.BatchNormalization()(x2_2_3)
    x2_2_3 = tf.keras.layers.ReLU()(x2_2_3)     # batch x 7 x 7 x 243

    x2_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_1)
    x2_3_1 = tf.keras.layers.BatchNormalization()(x2_3_1)
    x2_3_1 = tf.keras.layers.ReLU()(x2_3_1)     # batch x 14 x 14 x 243
    x2_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_1)
    x2_3_1 = tf.keras.layers.BatchNormalization()(x2_3_1)
    x2_3_1 = tf.keras.layers.ReLU()(x2_3_1)     # batch x 7 x 7 x 243

    x2_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_2)
    x2_3_2 = tf.keras.layers.BatchNormalization()(x2_3_2)
    x2_3_2 = tf.keras.layers.ReLU()(x2_3_2)     # batch x 14 x 14 x 243
    x2_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_2)
    x2_3_2 = tf.keras.layers.BatchNormalization()(x2_3_2)
    x2_3_2 = tf.keras.layers.ReLU()(x2_3_2)     # batch x 7 x 7 x 243

    x2_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_3)
    x2_3_3 = tf.keras.layers.BatchNormalization()(x2_3_3)
    x2_3_3 = tf.keras.layers.ReLU()(x2_3_3)     # batch x 14 x 14 x 243
    x2_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2_3_3)
    x2_3_3 = tf.keras.layers.BatchNormalization()(x2_3_3)
    x2_3_3 = tf.keras.layers.ReLU()(x2_3_3)     # batch x 7 x 7 x 243

    x3_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_1)
    x3_1_1 = tf.keras.layers.BatchNormalization()(x3_1_1)
    x3_1_1 = tf.keras.layers.ReLU()(x3_1_1)     # batch x 14 x 14 x 243
    x3_1_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_1)
    x3_1_1 = tf.keras.layers.BatchNormalization()(x3_1_1)
    x3_1_1 = tf.keras.layers.ReLU()(x3_1_1)     # batch x 7 x 7 x 243

    x3_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_2)
    x3_1_2 = tf.keras.layers.BatchNormalization()(x3_1_2)
    x3_1_2 = tf.keras.layers.ReLU()(x3_1_2)     # batch x 14 x 14 x 243
    x3_1_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_2)
    x3_1_2 = tf.keras.layers.BatchNormalization()(x3_1_2)
    x3_1_2 = tf.keras.layers.ReLU()(x3_1_2)     # batch x 7 x 7 x 243

    x3_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_3)
    x3_1_3 = tf.keras.layers.BatchNormalization()(x3_1_3)
    x3_1_3 = tf.keras.layers.ReLU()(x3_1_3)     # batch x 14 x 14 x 243
    x3_1_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_1_3)
    x3_1_3 = tf.keras.layers.BatchNormalization()(x3_1_3)
    x3_1_3 = tf.keras.layers.ReLU()(x3_1_3)     # batch x 7 x 7 x 243

    x3_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_1)
    x3_2_1 = tf.keras.layers.BatchNormalization()(x3_2_1)
    x3_2_1 = tf.keras.layers.ReLU()(x3_2_1)     # batch x 14 x 14 x 243
    x3_2_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_1)
    x3_2_1 = tf.keras.layers.BatchNormalization()(x3_2_1)
    x3_2_1 = tf.keras.layers.ReLU()(x3_2_1)     # batch x 7 x 7 x 243

    x3_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_2)
    x3_2_2 = tf.keras.layers.BatchNormalization()(x3_2_2)
    x3_2_2 = tf.keras.layers.ReLU()(x3_2_2)     # batch x 14 x 14 x 243
    x3_2_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_2)
    x3_2_2 = tf.keras.layers.BatchNormalization()(x3_2_2)
    x3_2_2 = tf.keras.layers.ReLU()(x3_2_2)     # batch x 7 x 7 x 243

    x3_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_3)
    x3_2_3 = tf.keras.layers.BatchNormalization()(x3_2_3)
    x3_2_3 = tf.keras.layers.ReLU()(x3_2_3)     # batch x 14 x 14 x 243
    x3_2_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_2_3)
    x3_2_3 = tf.keras.layers.BatchNormalization()(x3_2_3)
    x3_2_3 = tf.keras.layers.ReLU()(x3_2_3)     # batch x 7 x 7 x 243

    x3_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_1)
    x3_3_1 = tf.keras.layers.BatchNormalization()(x3_3_1)
    x3_3_1 = tf.keras.layers.ReLU()(x3_3_1)     # batch x 14 x 14 x 243
    x3_3_1 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_1)
    x3_3_1 = tf.keras.layers.BatchNormalization()(x3_3_1)
    x3_3_1 = tf.keras.layers.ReLU()(x3_3_1)     # batch x 7 x 7 x 243

    x3_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_2)
    x3_3_2 = tf.keras.layers.BatchNormalization()(x3_3_2)
    x3_3_2 = tf.keras.layers.ReLU()(x3_3_2)     # batch x 14 x 14 x 243
    x3_3_2 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_2)
    x3_3_2 = tf.keras.layers.BatchNormalization()(x3_3_2)
    x3_3_2 = tf.keras.layers.ReLU()(x3_3_2)     # batch x 7 x 7 x 243

    x3_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_3)
    x3_3_3 = tf.keras.layers.BatchNormalization()(x3_3_3)
    x3_3_3 = tf.keras.layers.ReLU()(x3_3_3)     # batch x 14 x 14 x 243
    x3_3_3 = tf.keras.layers.Conv2D(filters=243,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3_3_3)
    x3_3_3 = tf.keras.layers.BatchNormalization()(x3_3_3)
    x3_3_3 = tf.keras.layers.ReLU()(x3_3_3)     # batch x 7 x 7 x 243

    x1 = (x1_1_1 + x1_1_2 + x1_1_3) / 3
    x2 = (x1_2_1 + x1_2_2 + x1_2_3) / 3
    x3 = (x1_3_1 + x1_3_2 + x1_3_3) / 3

    x4 = (x2_1_1 + x2_1_2 + x2_1_3) / 3
    x5 = (x2_2_1 + x2_2_2 + x2_2_3) / 3
    x6 = (x2_3_1 + x2_3_2 + x2_3_3) / 3

    x7 = (x3_1_1 + x3_1_2 + x3_1_3) / 3
    x8 = (x3_2_1 + x3_2_2 + x3_2_3) / 3
    x9 = (x3_3_1 + x3_3_2 + x3_3_3) / 3

    output = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9

    return output

def RGB_model(input_shape=(224, 224, 3), weight_decay=0.0001, num_classes=100):

    h = inputs = tf.keras.Input(input_shape)
    
    R_output = branch_model(h[:,:,:,0], weight_decay)
    G_output = branch_model(h[:,:,:,1], weight_decay)
    B_output = branch_model(h[:,:,:,2], weight_decay)

    h = tf.concat([R_output, G_output, B_output], 3)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(num_classes)(h)
    
    return tf.keras.Model(inputs=inputs, outputs=h)