#-*- coding: utf-8 -*-
import tensorflow as tf

def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=tf.float32))
  return numerator / denominator

weight_decay = 0.0001
filters=32

def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = tf.keras.layers.GlobalAveragePooling2D()(input)
    fc1 = tf.keras.layers.Dense(channel_nums//r_factor)(ga_pooling)
    scale = tf.keras.layers.Dense(channel_nums, activation='sigmoid')(tf.keras.layers.ReLU()(fc1))
    scale = tf.expand_dims(scale, 1)
    scale = tf.expand_dims(scale, 1)
    #print(scale.shape)
    #print(input.shape)
    
    return tf.math.multiply(input, scale)

def compact_model(input_shape=(224,224,3), num_classes=10):

    # C3AE에서 여러 입력을 설정한 것 처럼 접근하자!
    # 똑같이 따라하지는 말자 --> 이미 모델이 다르기 때문에 입력되는 부분도 달라야 논문의 가치가 있어보인다.

    h = inputs = tf.keras.Input(input_shape)
    
    #h = tf.reverse(inputs, [-1]) - tf.constant([103.939, 116.779, 123.68])
    #print(h.shape)

    # 첫 레이어 부분부터 tree 형식으로 레이어를 구성하자
    ##########################################################################################
    Conv1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1)
    Conv1 = tf.keras.layers.ReLU()(Conv1)
    ##########################################################################################

    ##########################################################################################
    Max1 = tf.keras.layers.MaxPool2D(2, padding='same')(Conv1)  # 112 x 112 x 32
    Max1 = SE_BLOCK(Max1)
    Tree_a_1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Max1)
    Tree_a_1 = tf.keras.layers.BatchNormalization()(Tree_a_1)
    Tree_a_1 = tf.keras.layers.ReLU()(Tree_a_1)

    Tree_a_2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Max1)
    Tree_a_2 = tf.keras.layers.BatchNormalization()(Tree_a_2)
    Tree_a_2 = tf.keras.layers.ReLU()(Tree_a_2)
    ##########################################################################################

    ##########################################################################################
    Avg1 = tf.keras.layers.AveragePooling2D(2, padding='same')(Conv1)   # 112 x 112 x 32
    Avg1 = SE_BLOCK(Avg1)
    Tree_b_1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg1)
    Tree_b_1 = tf.keras.layers.BatchNormalization()(Tree_b_1)
    Tree_b_1 = tf.keras.layers.ReLU()(Tree_b_1)

    Tree_b_2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg1)
    Tree_b_2 = tf.keras.layers.BatchNormalization()(Tree_b_2)
    Tree_b_2 = tf.keras.layers.ReLU()(Tree_b_2)
    ##########################################################################################

    Tree_3 = Tree_a_2 + Tree_b_2

    ##########################################################################################
    Avg2 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_a_1)
    Avg2 = SE_BLOCK(Avg2)
    Tree_a_3 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg2)
    Tree_a_3 = tf.keras.layers.BatchNormalization()(Tree_a_3)
    Tree_a_3 = tf.keras.layers.ReLU()(Tree_a_3)

    Tree_a_4 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg2)
    Tree_a_4 = tf.keras.layers.BatchNormalization()(Tree_a_4)
    Tree_a_4 = tf.keras.layers.ReLU()(Tree_a_4)
    ##########################################################################################

    ##########################################################################################
    Avg3 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_3)
    Avg3 = SE_BLOCK(Avg3)
    Tree_b_3 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg3)
    Tree_b_3 = tf.keras.layers.BatchNormalization()(Tree_b_3)
    Tree_b_3 = tf.keras.layers.ReLU()(Tree_b_3)

    Tree_b_4 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg3)
    Tree_b_4 = tf.keras.layers.BatchNormalization()(Tree_b_4)
    Tree_b_4 = tf.keras.layers.ReLU()(Tree_b_4)
    ##########################################################################################

    ##########################################################################################
    Avg4 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_b_2)
    Avg4 = SE_BLOCK(Avg4)
    Tree_c_1 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg4)
    Tree_c_1 = tf.keras.layers.BatchNormalization()(Tree_c_1)
    Tree_c_1 = tf.keras.layers.ReLU()(Tree_c_1)

    Tree_c_2 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg4)
    Tree_c_2 = tf.keras.layers.BatchNormalization()(Tree_c_2)
    Tree_c_2 = tf.keras.layers.ReLU()(Tree_c_2)
    ##########################################################################################

    Tree_b_5 = Tree_a_4 + Tree_b_3
    Tree_c_3 = Tree_b_4 + Tree_c_1

    ##########################################################################################
    Avg5 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_a_3)
    Avg5 = SE_BLOCK(Avg5)
    Tree_a_5 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg5)
    Tree_a_5 = tf.keras.layers.BatchNormalization()(Tree_a_5)
    Tree_a_5 = tf.keras.layers.ReLU()(Tree_a_5)

    Avg6 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_b_5)
    Avg6 = SE_BLOCK(Avg6)
    Tree_b_6 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg6)
    Tree_b_6 = tf.keras.layers.BatchNormalization()(Tree_b_6)
    Tree_b_6 = tf.keras.layers.ReLU()(Tree_b_6)

    Tree_a_6 = Tree_a_5 + Tree_b_6
    ##########################################################################################

    ##########################################################################################
    Tree_b_7 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg6)
    Tree_b_7 = tf.keras.layers.BatchNormalization()(Tree_b_7)
    Tree_b_7 = tf.keras.layers.ReLU()(Tree_b_7)

    Avg7 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_c_3)
    Avg7 = SE_BLOCK(Avg7)
    Tree_b_8 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg7)
    Tree_b_8 = tf.keras.layers.BatchNormalization()(Tree_b_8)
    Tree_b_8 = tf.keras.layers.ReLU()(Tree_b_8)

    Tree_b_9 = Tree_b_7 + Tree_b_8
    ##########################################################################################

    ##########################################################################################
    Tree_c_4 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg7)
    Tree_c_4 = tf.keras.layers.BatchNormalization()(Tree_c_4)
    Tree_c_4 = tf.keras.layers.ReLU()(Tree_c_4)

    Avg8 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_c_2)
    Avg8 = SE_BLOCK(Avg8)
    Tree_c_5 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg8)
    Tree_c_5 = tf.keras.layers.BatchNormalization()(Tree_c_5)
    Tree_c_5 = tf.keras.layers.ReLU()(Tree_c_5)

    Tree_c_6 = Tree_c_4 + Tree_c_5
    ##########################################################################################

    ##########################################################################################
    Avg9 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_a_6)
    Avg9 = SE_BLOCK(Avg9)
    Tree_a_7 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg9)
    Tree_a_7 = tf.keras.layers.BatchNormalization()(Tree_a_7)
    Tree_a_7 = tf.keras.layers.ReLU()(Tree_a_7)

    Avg10 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_b_9)
    Avg10 = SE_BLOCK(Avg10)
    Tree_a_8 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg10)
    Tree_a_8 = tf.keras.layers.BatchNormalization()(Tree_a_8)
    Tree_a_8 = tf.keras.layers.ReLU()(Tree_a_8)

    Tree_a_9 = Tree_a_7 + Tree_a_8
    ##########################################################################################

    ##########################################################################################
    Tree_b_10 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg10)
    Tree_b_10 = tf.keras.layers.BatchNormalization()(Tree_b_10)
    Tree_b_10 = tf.keras.layers.ReLU()(Tree_b_10)

    Avg11 = tf.keras.layers.AveragePooling2D(2, padding='same')(Tree_c_6)
    Avg11 = SE_BLOCK(Avg11)
    Tree_b_11 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Avg11)
    Tree_b_11 = tf.keras.layers.BatchNormalization()(Tree_b_11)
    Tree_b_11 = tf.keras.layers.ReLU()(Tree_b_11)
    ##########################################################################################

    Tree_b_11 = Tree_b_10 + Tree_b_11

    print(Tree_a_9)
    print(Tree_b_11)

    Tree_a_9 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Tree_a_9)
    Tree_a_9 = tf.keras.layers.BatchNormalization()(Tree_a_9)
    Tree_a_9 = tf.keras.layers.ReLU()(Tree_a_9)

    Tree_b_11 = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(Tree_b_11)
    Tree_b_11 = tf.keras.layers.BatchNormalization()(Tree_b_11)
    Tree_b_11 = tf.keras.layers.ReLU()(Tree_b_11)

    h = tf.concat([Tree_a_9, Tree_b_11], 3)     # 지금 잠깐 돌려보고있음
    h = SE_BLOCK(h)

    h = tf.keras.layers.Conv2D(filters=filters*5,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)

    print(h.shape)

    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.Dense(num_classes, name='last_layer')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

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
    