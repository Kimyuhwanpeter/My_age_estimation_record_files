#-*- coding: utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

def bottle_neck(input,filters,weight_decay,filter_factor=2, strides=1):

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters*filter_factor,
                               kernel_size=1,
                               strides=strides,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    return h

def model_fit(input_shape=(224, 224, 3), num_classes=60, weight_decay=1e-4):

    h = inputs = tf.keras.Input(input_shape)

    # 이전 layer에 대해서 계속 어탠션 형식으로 가중치를 더해주면 어떠한가??
    h = tf.keras.layers.ZeroPadding2D(padding=(3,3))(h)
    h = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=7,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)           # 112 x 112 x 16

    h = bottle_neck(h, 16, weight_decay, 3)    # 112 x 112 x 48
    h = h_ = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(h)   # 56 x 56 x 48

    h = bottle_neck(h, 48, weight_decay)    # 56 x 56 x 96
    h = tf.keras.layers.Conv2D(filters=96,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)  # 56 x 56 x 96
    h = tf.keras.layers.BatchNormalization()(h) # 56 x 56 x 96

    split_h1 = h[:,:,:,0:48]
    split_h2 = h[:,:,:,48:96]

    Add1 = tf.keras.layers.Add()([split_h1, h_])
    Add2 = tf.keras.layers.Add()([split_h2, h_])

    h = tf.keras.layers.Concatenate(axis=3)([Add1, Add2])
    h = tf.keras.layers.ReLU()(h)

    h = h_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(h)   # 28 x 28 x 96

    h = bottle_neck(h, 96, weight_decay)   # 28 x 28 x 192
    h = tf.keras.layers.Conv2D(filters=192,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)  # 28 x 28 x 192
    h = tf.keras.layers.BatchNormalization()(h) # 28 x 28 x 192

    split_h3 = h[:,:,:,0:96]
    split_h4 = h[:,:,:,96:192]

    Add3 = tf.keras.layers.Add()([split_h3, h_1])
    Add4 = tf.keras.layers.Add()([split_h4, h_1])

    h = tf.keras.layers.Concatenate(axis=3)([Add3, Add4])
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=192,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 14 x 14 x 192

    h = bottle_neck(h, 192, weight_decay, 2, 2)   # 7 x 7 x 288

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(num_classes)(h)


    return tf.keras.Model(inputs=inputs, outputs=h)