#-*- coding:utf-8 -*-
import tensorflow as tf

# MobilenetV2 와 Resnetxt 를 결합해보자!
class GroupConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, n_groups, name):
        super(GroupConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.n_groups = n_groups
        self.name_ = name

    def build(self, inputs):
        #input_channels = tf.shape(inputs)
        _,_,_,input_channels = tf.TensorShape(inputs)
        #input_channels = inputs.shape[-1]
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_channels // self.n_groups, self.filters)

        self.kernel = self.add_weight(name="kernel",
                                      shape=kernel_shape,
                                      initializer=tf.keras.initializers.glorot_uniform,
                                      regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                      constraint=None,
                                      trainable=True,
                                      dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=tf.keras.initializers.Zeros,
                                        regularizer=None,
                                        constraint=None,
                                        trainable=True,
                                        dtype=self.dtype)

        else:
            self.bias = None

        self.groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=self.strides,
                        padding=self.padding, name=self.name_)

    @tf.function
    def call(self, inputs):
        if self.n_groups == 1:
            outputs = self.groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.n_groups, value=inputs)
            weightGroups = tf.split(axis=3, num_or_size_splits=self.n_groups, value=self.kernel)
            convGroup = [self.groupConv(i, k) for i,k in zip(inputGroups, weightGroups)]
            outputs = tf.concat(convGroup, 3)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs

####################################################################################
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def res_block(inputs, expansion, strides, alpha, filters, weight_decay, block_id, blocks="depth"):

    input_channels = tf.keras.backend.int_shape(inputs)[3]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    
    h = inputs
    prefix = "block_{}_".format(block_id)

    if block_id:
        h = tf.keras.layers.Conv2D(filters=expansion * input_channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=prefix + "expand")(h)
        h = tf.keras.layers.BatchNormalization(name=prefix + "expand_BN")(h)
        h = tf.keras.layers.ReLU(6., name=prefix + "expand_relu")(h)
    else:
        prefix = "expanded_conv_"

    if blocks == "depth":
        if strides == 2:
            h = tf.keras.layers.ZeroPadding2D(padding=(1,1), name=prefix + "pad")(h)
        h = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=strides,
                                            use_bias=False,
                                            padding="same" if strides == 1 else "valid",
                                            depthwise_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            name=prefix + "depthwise")(h)
        h = tf.keras.layers.BatchNormalization(name=prefix + "depthwise_BN")(h)
        h = tf.keras.layers.ReLU(6., name=prefix + "depthwise_relu")(h)
    
    else:
        h = GroupConv(filters=expansion * input_channels,
                      kernel_size=3,
                      strides=strides,
                      padding="SAME" if strides == 1 else "VALID",
                      use_bias=False,
                      weight_decay=weight_decay,
                      n_groups=32,
                      name=prefix + "groupconv")(h)
        h = tf.keras.layers.BatchNormalization(name=prefix + "groupconv_BN")(h)
        h = tf.keras.layers.ReLU(6., name=prefix + "groupconv_relu")(h)

    h = tf.keras.layers.Conv2D(filters=pointwise_filters,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name=prefix + "project")(h)
    h = tf.keras.layers.BatchNormalization(name=prefix + "project_BN")(h)

    if input_channels == pointwise_filters and strides == 1:
        return tf.keras.layers.Add(name=prefix + "add")([inputs, h])
    return h

def fix_model(input_shape=(224, 224, 3), num_classes=100, weight_decay=1e-4, alpha=1.0, depth='depth'):
    
    h = inputs = tf.keras.Input(input_shape)

    first_block_filters = make_divisible(32 * alpha, 8)
    h = tf.keras.layers.ZeroPadding2D(padding=(1,1), name="Conv1_pad")(h)
    h = tf.keras.layers.Conv2D(filters=first_block_filters,
                               kernel_size=3,
                               strides=2,
                               padding='valid',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name="Conv1")(h)
    h = tf.keras.layers.BatchNormalization(name="bn_Conv1")(h)
    h = tf.keras.layers.ReLU(6., name="Conv1_relu")(h)

    h = res_block(inputs=h, expansion=1, strides=1, alpha=alpha, filters=16, weight_decay=weight_decay, block_id=0, blocks='depth')

    h = res_block(inputs=h, expansion=6, strides=2, alpha=alpha, filters=24, weight_decay=weight_decay, block_id=1, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=24, weight_decay=weight_decay, block_id=2, blocks='depth')

    h = res_block(inputs=h, expansion=6, strides=2, alpha=alpha, filters=32, weight_decay=weight_decay, block_id=3, blocks='depth') 
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=32, weight_decay=weight_decay, block_id=4, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=32, weight_decay=weight_decay, block_id=5, blocks='depth')

    h = res_block(inputs=h, expansion=6, strides=2, alpha=alpha, filters=64, weight_decay=weight_decay, block_id=6, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=64, weight_decay=weight_decay, block_id=7, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=64, weight_decay=weight_decay, block_id=8, blocks='depth')

    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=96, weight_decay=weight_decay, block_id=9, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=96, weight_decay=weight_decay, block_id=10, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=96, weight_decay=weight_decay, block_id=11, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=96, weight_decay=weight_decay, block_id=12, blocks='depth')

    h = res_block(inputs=h, expansion=6, strides=2, alpha=alpha, filters=160, weight_decay=weight_decay, block_id=13, blocks='depth')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=160, weight_decay=weight_decay, block_id=14, blocks='groupconv')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=160, weight_decay=weight_decay, block_id=15, blocks='groupconv')
    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=160, weight_decay=weight_decay, block_id=16, blocks='groupconv')

    h = res_block(inputs=h, expansion=6, strides=1, alpha=alpha, filters=320, weight_decay=weight_decay, block_id=17, blocks='depth')

    if alpha > 1.0:
        last_block_filters = make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    h = tf.keras.layers.Conv2D(filters=last_block_filters,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name="Conv_1")(h)
    h = tf.keras.layers.BatchNormalization(name="Conv_1_bn")(h)
    h = tf.keras.layers.ReLU(6., name="out_relu")(h)

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)