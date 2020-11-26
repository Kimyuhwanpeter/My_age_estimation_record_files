#-*- coding: utf-8 -*-
import tensorflow as tf

def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=tf.float32))
  return numerator / denominator

weight_decay = 0.0001

def compact_model(input_shape=(224,224,3), num_classes=10):     # Non linear특성을 확실하게 살릴방법!!?

    # Convolution layer 마다 fully connected layer를 삽입하는것은 어떠한가?
    # 만일 이렇게 되면 파라미터 갯수가 급격하게 많아짐 --> 오버피팅의 위험이 더욱 커짐
    h = inputs = tf.keras.Input(input_shape)

    Conv1 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    print(Conv1.shape)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1)
    Conv1 = tf.keras.layers.ReLU()(Conv1)

    Conv2 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)

    print(Conv2.shape)
    Conv2 = tf.keras.layers.BatchNormalization()(Conv2)
    Conv2 = tf.keras.layers.ReLU()(Conv2)

    pixel_wise_sum = Conv1 + Conv2

    pixel_wise_sum = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(pixel_wise_sum)
    print(pixel_wise_sum.shape)

    Conv3 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(pixel_wise_sum)
    print(Conv3.shape)
    Conv3 = tf.keras.layers.BatchNormalization()(Conv3)
    Conv3 = tf.keras.layers.ReLU()(Conv3)

    Conv4 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv3)

    print(Conv4.shape)
    Conv4 = tf.keras.layers.BatchNormalization()(Conv4)
    Conv4 = tf.keras.layers.ReLU()(Conv4)

    Conv5 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv4)

    print(Conv5.shape)
    Conv5 = tf.keras.layers.BatchNormalization()(Conv5)
    Conv5 = tf.keras.layers.ReLU()(Conv5)
    
    Conv5 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(Conv5)
    print(Conv5.shape)

    Conv6 = tf.keras.layers.Conv2D(filters=8,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv5)

    print(Conv6.shape)
    Conv6 = tf.keras.layers.BatchNormalization()(Conv6)
    Conv6 = tf.keras.layers.ReLU()(Conv6)

    Conv7 = tf.keras.layers.Conv2D(filters=8,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv5)

    print(Conv7.shape)
    Conv7 = tf.keras.layers.BatchNormalization()(Conv7)
    Conv7 = tf.keras.layers.ReLU()(Conv7)

    Conv8 = tf.keras.layers.Conv2D(filters=8,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv5)

    print(Conv8.shape)
    Conv8 = tf.keras.layers.BatchNormalization()(Conv8)
    Conv8 = tf.keras.layers.ReLU()(Conv8)

    Conv9 = tf.keras.layers.Conv2D(filters=8,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv5)

    print(Conv9.shape)
    Conv9 = tf.keras.layers.BatchNormalization()(Conv9)
    Conv9 = tf.keras.layers.ReLU()(Conv9)

    concat_layer = tf.concat([Conv6, Conv7, Conv8, Conv9], 3)
    Conv10 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(concat_layer)
    print(Conv10.shape)

    Conv11 = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv10)

    print(Conv11.shape)
    Conv11 = tf.keras.layers.BatchNormalization()(Conv11)
    Conv11 = tf.keras.layers.ReLU()(Conv11)

    Conv12 = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv10)

    print(Conv12.shape)
    Conv12 = tf.keras.layers.BatchNormalization()(Conv12)
    Conv12 = tf.keras.layers.ReLU()(Conv12)

    Conv13 = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv10)

    print(Conv13.shape)
    Conv13 = tf.keras.layers.BatchNormalization()(Conv13)
    Conv13 = tf.keras.layers.ReLU()(Conv13)

    Conv14 = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv10)

    print(Conv14.shape)
    Conv14 = tf.keras.layers.BatchNormalization()(Conv14)
    Conv14 = tf.keras.layers.ReLU()(Conv14)

    concat_layer2 = tf.concat([Conv11, Conv12, Conv13, Conv14], 3)
    Conv15 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(concat_layer2)
    print(Conv15.shape)

    Conv16 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv15)

    print(Conv16.shape)
    Conv16 = tf.keras.layers.BatchNormalization()(Conv16)
    Conv16 = tf.keras.layers.ReLU()(Conv16)

    Conv17 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv15)

    print(Conv17.shape)
    Conv17 = tf.keras.layers.BatchNormalization()(Conv17)
    Conv17 = tf.keras.layers.ReLU()(Conv17)

    Conv18 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv15)

    print(Conv18.shape)
    Conv18 = tf.keras.layers.BatchNormalization()(Conv18)
    Conv18 = tf.keras.layers.ReLU()(Conv18)

    Conv19 = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Conv15)

    print(Conv19.shape)
    Conv19 = tf.keras.layers.BatchNormalization()(Conv19)
    Conv19 = tf.keras.layers.ReLU()(Conv19)

    SE0 = tf.keras.layers.GlobalMaxPool2D()(Conv15)
    SE0 = tf.expand_dims(SE0, 1)
    print(SE0.shape)
    SE0 = tf.keras.layers.Conv1D(filters=128 // 16,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(SE0)
    SE0 = tf.keras.layers.BatchNormalization()(SE0)
    SE0 = tf.math.sigmoid(SE0)

    SE0 = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(SE0)
    SE0 = tf.keras.layers.BatchNormalization()(SE0)
    SE0 = tf.math.sigmoid(SE0)
    SE0 = tf.expand_dims(SE0, 1)
    print(SE0.shape)

    concat_layer3 = tf.concat([Conv16, Conv17, Conv18, Conv19], 3)
    concat_layer3 = tf.multiply(concat_layer3, SE0)
    print(concat_layer3.shape)
    Conv20 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(concat_layer3)
    print(Conv20.shape)

    SE1 = tf.keras.layers.GlobalMaxPool2D()(Conv20)
    SE1 = tf.expand_dims(SE1, 1)
    print(SE1.shape)
    SE1 = tf.keras.layers.Conv1D(filters=128 // 16,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(SE1)
    SE1 = tf.keras.layers.BatchNormalization()(SE1)
    SE1 = tf.math.sigmoid(SE1)
    print(SE1.shape)
    SE1 = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(SE1)
    print(SE1.shape)
    SE1 = tf.keras.layers.BatchNormalization()(SE1)
    SE1 = tf.math.sigmoid(SE1)
    SE1 = tf.expand_dims(SE1, 1)
    print(SE1.shape)

    h = tf.keras.layers.ZeroPadding2D()(Conv20)         #######################
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding='valid',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.math.multiply(h, SE1)


    print(h.shape)
    h = tf.keras.layers.GlobalMaxPool2D()(h)
    h = tf.expand_dims(h, 1)
    print(h.shape)

    f1 = tf.keras.layers.Conv1D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    print(f1.shape)
    f1 = tf.keras.layers.BatchNormalization()(f1)
    f1 = tf.math.sigmoid(f1)
    f2 = tf.keras.layers.Conv1D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    print(f2.shape)
    f2 = tf.keras.layers.BatchNormalization()(f2)
    f2 = tf.math.sigmoid(f2)
    f3 = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    print(f3.shape)
    f3 = tf.keras.layers.BatchNormalization()(f3)
    f3 = tf.math.sigmoid(f3)
    f4 = tf.keras.layers.Conv1D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    print(f4.shape)
    f4 = tf.keras.layers.BatchNormalization()(f4)
    f4 = tf.math.sigmoid(f4)

    concat_layer4 = tf.concat([f1, f2, f3, f4], 2)
    concat_layer4 = tf.squeeze(concat_layer4, 1)
    print(concat_layer4.shape)

    SE2 = tf.keras.layers.Dense(736 // 2, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), use_bias=False)(concat_layer4)      # 여기를 다시 짜보자!
    SE2 = tf.keras.layers.BatchNormalization()(SE2)
    SE2 = tf.keras.layers.ReLU()(SE2)
    SE2 = tf.keras.layers.Dense(736, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), use_bias=False)(SE2)
    SE2 = tf.keras.layers.BatchNormalization()(SE2)
    SE2 = tf.keras.layers.ReLU()(SE2)

    concat_layer4 = tf.math.multiply(concat_layer4, SE2)

    h = tf.keras.layers.Dense(num_classes, name='last_layer')(concat_layer4)

    return tf.keras.Model(inputs=inputs, outputs=h)
# SE module을 추가해보자 --> 추가하고 지금 코랩에 돌리는중!

# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate))
        return self.current_learning_rate


#global_step,
#learning_rate_base,
#total_steps,
#warmup_learning_rate=0.0,
#warmup_steps=0,
#hold_base_rate_steps=0
class warmup_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps, hold_base_rate_steps):
        super(warmup_decay, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.current_learning_rate = tf.Variable(initial_value=learning_rate_base, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.learning_rate = 0.5 * self.learning_rate_base * (1 + tf.cos(
        3.14 *(tf.cast(step, tf.float32) - self.warmup_steps - self.hold_base_rate_steps
        ) / float(self.total_steps - self.warmup_steps - self.hold_base_rate_steps)))

        if self.hold_base_rate_steps > 0:
          self.learning_rate = tf.where(
              step > self.warmup_steps + self.hold_base_rate_steps,
              self.learning_rate, self.learning_rate_base)

        if self.warmup_steps > 0:
          if self.learning_rate_base < self.warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
          slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
          self.warmup_rate = slope * tf.cast(step,
                                        tf.float32) + self.warmup_learning_rate
          self.learning_rate = tf.where(step < self.warmup_steps, self.warmup_rate,
                                   self.learning_rate)

        self.current_learning_rate.assign(tf.where(
            step > self.total_steps,
            0.0, self.learning_rate))
        return self.current_learning_rate

#def eager_decay_rate():
#    """Callable to compute the learning rate."""
#    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
#        np.pi *
#        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
#        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
#    if hold_base_rate_steps > 0:
#        learning_rate = tf.where(
#            global_step > warmup_steps + hold_base_rate_steps,
#            learning_rate, learning_rate_base)
#    if warmup_steps > 0:
#        if learning_rate_base < warmup_learning_rate:
#            raise ValueError('learning_rate_base must be larger or equal to '
#                                'warmup_learning_rate.')
#        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
#        warmup_rate = slope * tf.cast(global_step,
#                                    tf.float32) + warmup_learning_rate
#        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
#                                learning_rate)
#    return tf.where(global_step > total_steps, 0.0, learning_rate,
#                    name='learning_rate')