# -*- coding: utf-8 -*-
import tensorflow as tf

def model(input_shape=(224, 224, 3), filters=32, weight_decay=0.0001, num_classes=100):
    
    h = inputs = tf.keras.Input(input_shape)
    # 완전한 트리형식으로 접근해볼--> 마지막 layer 는 concat해서 사용

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=7,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 224 x 224 x 32

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               dilation_rate=2,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 224 x 224 x 32

    h = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(h) # 112 x 112 x 32

    h1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.ReLU()(h1)     # 56 x 56 x 32

    h2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h1)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    h2 = tf.keras.layers.ReLU()(h2)     # 28 x 28 x 32

    h3 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h2)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    h3 = tf.keras.layers.ReLU()(h3)     # 14 x 14 x 32

    h4 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h3)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    h4 = tf.keras.layers.ReLU()(h4)     # 7 x 7 x 32

    k1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    k1 = tf.keras.layers.BatchNormalization()(k1)
    k1 = tf.keras.layers.ReLU()(k1)     # 56 x 56 x 32

    ############################################
    k1_half = k1[:,:,:,0:filters // 2]
    h1_half = h1[:,:,:,filters // 2:filters]
    k1 = tf.concat([k1_half, h1_half], 3)
    ############################################

    k2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(k1)
    k2 = tf.keras.layers.BatchNormalization()(k2)
    k2 = tf.keras.layers.ReLU()(k2)     # 28 x 28 x 32

    ############################################
    k2_half = k2[:,:,:,0:filters // 2]
    h2_half = h2[:,:,:,filters // 2:filters]
    k2 = tf.concat([k2_half, h2_half], 3)
    ###########################################

    k3 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(k2)
    k3 = tf.keras.layers.BatchNormalization()(k3)
    k3 = tf.keras.layers.ReLU()(k3)     # 14 x 14 x 32

    ############################################
    k3_half = k3[:,:,:,0:filters // 2]
    h3_half = h3[:,:,:,filters // 2:filters]
    k3 = tf.concat([k3_half, h3_half], 3)
    ############################################

    k4 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(k3)
    k4 = tf.keras.layers.BatchNormalization()(k4)
    k4 = tf.keras.layers.ReLU()(k4)     # 7 x 7 x 32

    n1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(k1)
    n1 = tf.keras.layers.BatchNormalization()(n1)
    n1 = tf.keras.layers.ReLU()(n1)     # 28 x 28 x 32

    ############################################
    k2_half = k2[:,:,:,filters // 2:filters]
    n1_half = n1[:,:,:,0:filters // 2]
    n1 = tf.concat([n1_half, k2_half], 3)
    ###########################################

    n2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(n1)
    n2 = tf.keras.layers.BatchNormalization()(n2)
    n2 = tf.keras.layers.ReLU()(n2)     # 14 x 14 x 32

    ############################################
    k3_half = k3[:,:,:,filters // 2:filters]
    n2_half = n2[:,:,:,0:filters // 2]
    n2 = tf.concat([n2_half, k3_half], 3)
    ###########################################

    n3 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(n2)
    n3 = tf.keras.layers.BatchNormalization()(n3)
    n3 = tf.keras.layers.ReLU()(n3)     # 7 x 7 x 32

    m1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(n1)
    m1 = tf.keras.layers.BatchNormalization()(m1)
    m1 = tf.keras.layers.ReLU()(m1)     # 14 x 14 x 32

    ############################################
    m1_half = m1[:,:,:,0:filters // 2]
    n2_half = n2[:,:,:,filters // 2:filters]
    m1 = tf.concat([m1_half, n2_half], 3)
    ###########################################

    m2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(m1)
    m2 = tf.keras.layers.BatchNormalization()(m2)
    m2 = tf.keras.layers.ReLU()(m2)     # 7 x 7 x 32

    l = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(m1)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.ReLU()(l)       # 7 x 7 x 32

    fc = tf.concat([h4, k4, n3, m2, l], 3)
    fc = tf.keras.layers.GlobalAveragePooling2D()(fc)
    fc = tf.keras.layers.Dense(num_classes)(fc)
    # 코랩에서 실험해볼 경우의 수
    # 1. image size를 224로 설정한 뒤     --> 현재 실험중! batch size = 64
    # 2. image size를 64로 설정한 뒤
    # 3. image size를 224, 맨 뒤의 fc layer에 1024의 fully connected layer를 추가했을 때
    # 4. image size를 64, 맨 뒤의 fc layer에 1024의 fully connected layer를 추가했을 때

    return tf.keras.Model(inputs=inputs, outputs=fc)