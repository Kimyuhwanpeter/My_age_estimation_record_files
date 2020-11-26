#-*- coding: utf-8 -*-
import tensorflow as tf

weight_decay = 0.0001
filters = 64        # 현재 코랩에서 32로 실험중... 결과가 안좋으면 첫 번째 필터 사이즈 3으로 해서 코랩으로 돌리자
                    # 위에 마저도 다 안되면 이제 필터를 64로 하고 코랩에 실험 돌려보자
                    # 현재 필터 64, 앞단 필터 사이즈 3으로 코랩으로 돌리는중!    --> 일단보류!! 다음주에 돌리든지하자!!!!

def Model(input_shape=(224,224,3), num_classes=101):
    
    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.AveragePooling2D(2, padding='same')(h)

    h = tf.keras.layers.Conv2D(filters=int(filters * 1.5),
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.AveragePooling2D(2, padding='same')(h)

    h = tf.keras.layers.Conv2D(filters=int(filters * 2.5),
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.AveragePooling2D(2, padding='same')(h)

    h = tf.keras.layers.Conv2D(filters=int(filters * 3.5),
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.AveragePooling2D(2, padding='same')(h)

    h = tf.keras.layers.Conv2D(filters=int(filters * 4.5),
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.AveragePooling2D(2, padding='same')(h)

    h = tf.keras.layers.Conv2D(filters=int(filters * 5.5),
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    
    fully1 = tf.keras.layers.Dense(int(filters * 10.5), use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    fully1 = tf.keras.layers.BatchNormalization()(fully1)
    fully1 = tf.keras.layers.ReLU()(fully1)

    fully2 = tf.keras.layers.Dense(int(filters * 10.5), use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    fully2 = tf.keras.layers.BatchNormalization()(fully2)
    fully2 = tf.keras.layers.ReLU()(fully2)

    output1 = tf.keras.layers.Dense(num_classes, name='last_layer_1')(fully1)
    output2 = tf.keras.layers.Dense(num_classes, name='last_layer_2')(fully2)
    
    return tf.keras.Model(inputs=inputs, outputs=[output1, output2])