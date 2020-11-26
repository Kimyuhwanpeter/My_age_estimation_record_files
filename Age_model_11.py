# -*- coding: utf-8 -*-
import tensorflow as tf

class Custom_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, kernel_regularizer, name):
        super(Custom_Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.name_ = name
    
    def build(self, input):
        _,_,_,input_ch = tf.TensorShape(input)
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_ch, self.filters)

        self.kernel = self.add_weight(name="weights",
                                      shape=kernel_shape,
                                      initializer=tf.keras.initializers.glorot_uniform,
                                      regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                                      constraint=None,
                                      trainable=True,
                                      dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.filters,),
                                        initializer=tf.keras.initializers.Zeros,
                                        regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                                        constraint=None,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

    @tf.function
    def call(self, inputs):
        if self.use_bias:
            conv = tf.nn.conv2d(inputs, self.kernel, self.strides, self.padding, name=self.name_)
            output = tf.nn.bias_add(conv, self.bias)
        else:
            output = tf.nn.conv2d(inputs, self.kernel, self.strides, self.padding, name=self.name_)
        return output, self.kernel

class Custom_Conv2D_filters(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, kernel_regularizer, name):
        super(Custom_Conv2D_filters, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.name_ = name

    def build(self, input):
        _,_,_,input_ch = tf.TensorShape(input)
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_ch, self.filters)
        
        self.kernel = self.add_weight(name="weights",
                                      shape=kernel_shape,
                                      initializer=tf.keras.initializers.glorot_uniform,
                                      regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                                      constraint=None,
                                      trainable=True,
                                      dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.filters,),
                                        initializer=tf.keras.initializers.Zeros,
                                        regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                                        constraint=None,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
    
    def call(self, input, kernel):
        if self.use_bias:
            kernel = tf.concat([kernel, self.kernel], 3)
            #kernel = kernel + self.kernel
            conv = tf.nn.conv2d(input, kernel, self.strides, self.padding, name=self.name_)
            output = tf.nn.bias_add(conv, self.bias)
        else:
            kernel = tf.concat([kernel, self.kernel], 3)
            #kernel = kernel + self.kernel
            output = tf.nn.conv2d(input, kernel, self.strides, self.padding, name=self.name_)
        return output, kernel

def Block(input, filters, kernel_size, strides, padding, use_bias, weight_decay, name):

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)

    return h
    
def model_fix_v_2(input_shape=(224, 224, 3), num_classes=100, weight_decay=0.0005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_1 = Custom_Conv2D(filters=32,
                                kernel_size=3,
                                strides=2,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_2 = Custom_Conv2D_filters(filters=32,
                                        kernel_size=3,
                                        strides=1,
                                        padding="SAME",
                                        use_bias=False,
                                        kernel_regularizer=weight_decay,
                                        name="customConv2")(h, weight_1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPool2D((2,2), padding="same")(h)

    h, weight_3 = Custom_Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv3")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_4 = Custom_Conv2D_filters(filters=64,
                                        kernel_size=3,
                                        strides=1,
                                        padding="SAME",
                                        use_bias=False,
                                        kernel_regularizer=weight_decay,
                                        name="customConv4")(h, weight_3)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_5 = Custom_Conv2D(filters=64,
                                kernel_size=1,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv5")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_6 = Custom_Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_7 = Custom_Conv2D(filters=128,
                                kernel_size=1,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_8 = Custom_Conv2D(filters=128,
                                kernel_size=3,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_9 = Custom_Conv2D_filters(filters=128,
                                kernel_size=3,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv9")(h, weight_8)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPool2D((2,2), padding="same")(h)

    h, weight_10 = Custom_Conv2D(filters=256,
                                kernel_size=1,
                                strides=1,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv10")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h, weight_11 = Custom_Conv2D_filters(filters=256,
                                kernel_size=1,
                                strides=2,
                                padding="SAME",
                                use_bias=False,
                                kernel_regularizer=weight_decay,
                                name="customConv11")(h, weight_10)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)