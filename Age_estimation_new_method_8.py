# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from Age_model_8 import *

import tensorflow as tf
import numpy as np
import os
import sys

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("val_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_string("val_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_integer("img_size", 224, "Image size (height and width)")

flags.DEFINE_integer("ch", 3, "Channels")

flags.DEFINE_integer("batch_size", 32, "Training batch size")

flags.DEFINE_integer("val_batch_size", 128, "Validation batch size")

flags.DEFINE_integer("num_classes", 60, "Number of classes")

flags.DEFINE_integer("epochs", 200, "Training total epochs")

flags.DEFINE_float("lr", 1e-4, "Learning rate")

flags.DEFINE_float("weight_decay", 1e-4, "Weight regularization")

flags.DEFINE_bool("train", True, "")

flags.DEFINE_bool("pre_checkpoint", False, "")

flags.DEFINE_string("pre_checkpoint_path", "", "")

flags.DEFINE_string("save_checkpoint", "", "")

flags.DEFINE_string("graphs", "", "")
# 이 모델에 대해서는 지금 Age_model_6을 진행하고 난뒤 이 모델에 대해서 실험을 진행하자!!
FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

def test_func(img_list, lab_list):
    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    if lab_list == 74:
        label = (lab_list - 2) - 16
    elif lab_list == 75:
        label = (lab_list - 2) - 16
    elif lab_list == 76:
        label = (lab_list - 2) - 16
    elif lab_list == 77:
        label = (lab_list - 2) - 16
    else:
        label = lab_list - 16

    return img, label

def train_func(img_list, lab_list):
    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    if lab_list == 74:
        label = (lab_list - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_list == 75:
        label = (lab_list - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_list == 76:
        label = (lab_list - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_list == 77:
        label = (lab_list - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    else:
        label = lab_list - 16
        label = tf.one_hot(label, FLAGS.num_classes)

    return img, label

@tf.function
def run(model, batch_images, training=True):
    logits = model(batch_images, training=training)
    return logits

def train_loss(model, batch_images, batch_labels):
    with tf.GradientTape() as tape:
        logits = run(model, batch_images, True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(batch_labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main(agrv=None):
    # 내가 코딩한 mobilenet v2하고 tensorflow에 제공된 mobilenetv2하고 비교해보자! 이게 딱 맞아야 내가 fine-tuning 모델을 사용할 수 있다.
    # 기억해!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 그리고 내가 코딩한 모바일넷 레이어가 약간이상하다! 고치자!!!
    model = fix_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch),
                          num_classes=FLAGS.num_classes,
                          weight_decay=FLAGS.weight_decay)
    #model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    model.summary()
    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the checkpoint!!!")

    if FLAGS.train is True:
        count = 0
        train_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        train_img = [FLAGS.img_path + data for data in train_img]
        train_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_img = np.loadtxt(FLAGS.val_txt_path, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img_path + data for data in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        VAL = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
        VAL = VAL.map(test_func)
        VAL = VAL.batch(FLAGS.val_batch_size)
        VAL = VAL.prefetch(tf.data.experimental.AUTOTUNE)

        val_idx = len(val_lab) // FLAGS.val_batch_size
        train_idx = len(train_lab) // FLAGS.batch_size

        for epoch_ in range(FLAGS.epochs):
            A = list(zip(train_img, train_lab))
            shuffle(A)
            train_img, train_lab = zip(*A)
            train_img, train_lab = np.array(train_img), np.array(train_lab)

            TRAIN = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            TRAIN = TRAIN.shuffle(len(train_lab))
            TRAIN = TRAIN.map(train_func)
            TRAIN = TRAIN.batch(FLAGS.batch_size)
            TRAIN = TRAIN.prefetch(tf.data.experimental.AUTOTUNE)
            TRAIN_iter = iter(TRAIN)

            for step in range(train_idx):

                batch_images, batch_labels = next(TRAIN_iter)

                loss = train_loss(model, batch_images, batch_labels)

                if count % 10 == 0:
                    print("Loss = {}".format(loss))

                if (count + 1) % val_idx == 0:
                    AE = 0      # MAE of validation <test> set

                count += 1

if __name__ == "__main__":
    app.run(main)