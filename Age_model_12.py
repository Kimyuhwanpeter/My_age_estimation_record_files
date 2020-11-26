# -*- coding: utf-8 -*-
import tensorflow as tf

class layers():
    Conv2D = tf.keras.layers.Conv2D
    GloAvgPool2D = tf.keras.layers.GlobalAveragePooling2D
    AvgPool2D = tf.keras.layers.AvgPool2D
    MaxPool2D = tf.keras.layers.MaxPool2D
    BatchNorm = tf.keras.layers.BatchNormalization
    ReLU = tf.keras.layers.ReLU

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):   # 모바일넷에서 이용하던 block을 이용해서 짜보자!
  """Inverted ResNet block."""
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

  in_channels = backend.int_shape(inputs)[channel_axis]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  prefix = 'block_{}_'.format(block_id)

  if block_id:
    # Expand
    x = layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(
            x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
  else:
    prefix = 'expanded_conv_'

  # Depthwise
  if stride == 2:
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, 3),
        name=prefix + 'pad')(x)
  x = layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same' if stride == 1 else 'valid',
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(
          x)

  x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

  # Project
  x = layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project_BN')(
          x)

  if in_channels == pointwise_filters and stride == 1:
    return layers.Add(name=prefix + 'add')([inputs, x])
  return x

def ensemble(input_shape=(224, 224, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    # 앙상블 형식으로 만들자!

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=1,
                      padding="valid",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)

    h = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 112 x 112 x 128

    h = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 56 x 56 x 256
    
        
    return tf.keras.Model(inputs=inputs, outputs=h)

a = tf.keras.applications.MobileNetV2