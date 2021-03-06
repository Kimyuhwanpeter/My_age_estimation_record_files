# -*- coding: utf-8 -*-
from tqdm import tqdm
from absl import flags, app
from random import shuffle, random
from collections import Counter
from Age_model_4 import *
from keras_radam.training import RAdamOptimizer

import os
import tensorflow as tf
import numpy as np
import sys
import datetime
import matplotlib.pylab as plt

flags.DEFINE_string('txt_path', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt', 'Training text path')

flags.DEFINE_string('img_path', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/', 'Training image path')

flags.DEFINE_string('val_txt', "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation (test) text path")

flags.DEFINE_string("val_img", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation (test) image path")

flags.DEFINE_integer('img_size', 224, 'height and width')

flags.DEFINE_integer('ch', 3, 'Channels')

flags.DEFINE_integer('batch_size', 50, 'batch size')

flags.DEFINE_integer("val_batch", 128, "")

flags.DEFINE_integer('epochs', 100, 'Epochs on training')

flags.DEFINE_integer("epoch_decay", 50, "Epochs decay")

flags.DEFINE_float('lr', 0.001, 'Learning rate')

flags.DEFINE_integer('num_classes', 60, 'Number of classes')

flags.DEFINE_bool('pre_checkpoint', False, 'True or False')

flags.DEFINE_string('pre_checkpoint_path', 'C:/Users/Yuhwan/Desktop/1249', '(saved) checkpoint path')

flags.DEFINE_string('save_checkpoint', 'D:/tensorflor2.0(New_age_estimation)/checkpoint', 'saving checkpoint path')

flags.DEFINE_bool('train', True, 'True of False')

flags.DEFINE_string('graphs', 'D:/tensorflow2.0(New_generator_CycleGAN)/graphs/', 'Directory of loss graphs')

####################################################################################################################################
flags.DEFINE_string('test_txt', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt', 'Test text path')

flags.DEFINE_string('test_img', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/', 'Test image path')
####################################################################################################################################

FLAGS = flags.FLAGS
FLAGS(sys.argv)

len_dataset = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)
len_dataset = len(len_dataset)
#scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset // FLAGS.batch_size, FLAGS.epoch_decay * len_dataset // FLAGS.batch_size)
#optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
#scheduler = warmup_decay(FLAGS.lr, FLAGS.epochs * len_dataset // FLAGS.batch_size, 0.001, FLAGS.epoch_decay * len_dataset // FLAGS.batch_size, 0)
#optimizer = tf.keras.optimizers.SGD(scheduler, momentum=0.9)
optimizer = RAdamOptimizer(FLAGS.lr)

def test_func(image, label):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    hsv = tf.image.rgb_to_hsv(img)
    yiq = tf.image.rgb_to_yiq(img)

    img = tf.image.per_image_standardization(img)
    hsv = tf.image.per_image_standardization(hsv)
    yiq = tf.image.per_image_standardization(yiq)

    return img, hsv, yiq, label

def _func(image, label):        # RGB, HSV, YIQ이들을 입력으로서 학습에 이용해보자!!!

    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])

    image_hsv = tf.image.rgb_to_hsv(image)
    image_yiq = tf.image.rgb_to_yiq(image)

    if random() > 0.5:
        image = tf.image.flip_left_right(image)
        image_hsv = tf.image.flip_left_right(image_hsv)
        image_yiq = tf.image.flip_left_right(image_yiq)

    image = tf.image.per_image_standardization(image)
    image_hsv = tf.image.per_image_standardization(image_hsv)
    image_yiq = tf.image.per_image_standardization(image_yiq)

    if label == 74:
        label = 72
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 75:
        label = 73
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 76:
        label = 74
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 77:
        label = 75
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    else:
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)

    return image, image_hsv, image_yiq, label

@tf.function
def run_model(model, images, hsv_images, yiq_images, trainable=True):
    logits = model([images, hsv_images, yiq_images], training=trainable)
    return logits

def cla_loss(logits, labels, weights, class_factor):
    loss = 0
    for batch in range(FLAGS.batch_size):

        logit = logits[batch]
        label = labels[batch]
        age = tf.argmax(label)

        #W = tf.reduce_mean(weights[0]*weights[0] / 2, 0)       # 54 shape
        #logit = tf.nn.softmax(class_factor[age] * W + logit)        # 이 부분이 만족스럽지 않다;;;;;

        W = tf.reduce_mean(weights[0]*weights[0] / 2)
        logit = tf.nn.softmax(logit)
        loss += tf.keras.losses.CategoricalCrossentropy(from_logits=True)(label, logit) + class_factor[age] * W

    return loss / FLAGS.batch_size

def train_step(model, images, hsv_images, yiq_images, labels, last_weights, class_factor):
    with tf.GradientTape() as tape:

        logits = run_model(model, images, hsv_images, yiq_images, True)
        #total_loss = cla_loss(logits, labels, last_weights, class_factor)
        total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)  # 이거 헷갈리지말고 기억해!!

    gradients = tape.gradient(total_loss, model.trainable_variables)
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

def main(argv=None):
    # 다음주 코랩에서 돌려보자!!!!! 기억해!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model = compact_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), num_classes=FLAGS.num_classes)

    model2 = compact_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), num_classes=FLAGS.num_classes)

    model3 = compact_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), num_classes=FLAGS.num_classes)

    h1 = model.output
    h2 = model2.output
    h3 = model3.output

    h = tf.concat([h1, h2, h3], 1)
    h = tf.keras.layers.Dense(FLAGS.num_classes, name='last_layer')(h)
    
    model = tf.keras.Model(inputs=[model.input, model2.input, model3.input], outputs=h)
    L = model.get_layer('last_layer')
    W = L.get_weights()
    model.summary()

    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 3)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!!! [*{}*]".format(ckpt_manager.latest_checkpoint))

    if FLAGS.train is True:
        count = 0;

        data_img = np.loadtxt(FLAGS.txt_path, dtype='<U100', skiprows=0, usecols=0)
        data_img = [FLAGS.img_path + data for data in data_img]
        data_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)
        lab = data_lab - 16
        # label에 대해서 가우시안 분포로 만든 뒤 적은 클래스에 대해 가중치를 조금 더 주는 형식으로 학습을 진행하면 어떻게 될까?
        # 우선 각 라벨에 해당하는 개수를 구함
        C = Counter(lab)
        C = [float(C[i]) for i in C]

        # 개수는 크기때문에 정규화 시킨다!
        max = np.argmax(C)
        C = np.array(C) / C[max]
        C = 1. - C

        val_img = np.loadtxt(FLAGS.val_txt, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img + data for data in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt, dtype=np.int32, skiprows=0, usecols=1)
        val_lab = [data - 2 if data > 71 else data for data in val_lab]
        val_lab = np.array(val_lab, np.int32) - 16

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)

        for ep in range(FLAGS.epochs):
            A = list(zip(data_img, data_lab))
            shuffle(A)
            data_img, data_lab = zip(*A)
            
            data_img = np.array(data_img)
            data_lab = np.array(data_lab)

            dataset = tf.data.Dataset.from_tensor_slices((data_img, data_lab))
            dataset = dataset.shuffle(len(data_img))
            dataset = dataset.map(_func)
            dataset = dataset.batch(FLAGS.batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            data_iter = iter(dataset)

            dataset_val = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
            dataset_val = dataset_val.map(test_func)
            dataset_val = dataset_val.batch(FLAGS.val_batch)
            dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)

            val_iter = iter(dataset_val)

            batch_idx = len(data_img) // FLAGS.batch_size
            val_idx = len(val_lab) // FLAGS.val_batch
            for step in range(batch_idx):

                batch_images, hsv_images, yiq_images, batch_labels = next(data_iter)

                total_loss = train_step(model, batch_images, hsv_images, yiq_images, batch_labels, W, C)

                if (count + 1) % (val_idx + 1) == 0:
                    AE = 0
                    val_iter = iter(dataset_val)
                    for i in range(val_idx):
                        val_images, hsv, yiq, val_labels = next(val_iter)
                        predict_age = run_model(model, val_images, hsv, yiq, False)
                        predict_age = tf.argmax(predict_age, 1)
                        AE += tf.reduce_sum(tf.math.abs(val_labels.numpy() - predict_age.numpy()))
                    print("===============================")
                    print("Epoch: {} [{}/{}] MAE = {}".format(ep + 1, step + 1, batch_idx, AE / len(val_img)))
                    print("===============================")
                    model_dir = FLAGS.save_checkpoint
                    folder_name = int(count/(len(val_lab) // FLAGS.val_batch))
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print("Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(model=model,
                                                    optimizer=optimizer)
                    checkpoint_dir = folder_neme_str + "/" + "new_age_estimation_{}_steps.ckpt".format(count)
                    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

                    manager.save()
                    with val_summary_writer.as_default():
                        tf.summary.scalar(u'MAE', AE / len(val_img), step=count)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(ep + 1, step + 1, batch_idx, total_loss))

                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', total_loss, step=count)

                count += 1


    else:
        data_img = np.loadtxt(FLAGS.test_txt, dtype='<U100', skiprows=0, usecols=0)
        data_img = [FLAGS.test_img + data for data in data_img]
        data_lab = np.loadtxt(FLAGS.test_txt, dtype=np.int32, skiprows=0, usecols=1)
        data_lab = [data - 2 if data > 71 else data for data in data_lab]
        data_lab = np.array(data_lab, np.int32) - 16
        print(len(data_lab))

        test_data = tf.data.Dataset.from_tensor_slices((data_img, data_lab))
        test_data = test_data.map(test_func)
        test_data = test_data.batch(FLAGS.batch_size)
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        count = 0
        for ep in range(2):
            it = iter(test_data)
            AE = 0
            for i in range(len(data_lab) // FLAGS.batch_size):

                image, gr_age = next(it)
                logits = run_model(model, image, False)
                pre_age = tf.argmax((logits), 1)
                #print(pre_age.numpy(), gr_age[0].numpy())

                AE += tf.reduce_sum(tf.math.abs(gr_age.numpy() - pre_age.numpy()))

                if i % 1000 == 0:
                    print("MAE per {} image(s)...".format(AE / (i + 1)))

                count += 1

            print("AE = {}".format(AE))
            print("MAE for total test images = {}".format( AE / len(data_img) ))


if __name__ == '__main__':
    app.run(main)