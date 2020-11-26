# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from Age_model_7 import *

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("val_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_string("val_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_integer("img_size", 224, "Image size (height and width)")

flags.DEFINE_integer("ch", 3, "Channels")

flags.DEFINE_integer("batch_size", 64, "Training batch size")

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

def func(name, label):

    img = tf.io.read_file(name)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)
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

    return img, label

def func_val(name, label):

    img = tf.io.read_file(name)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    if label == 74:
        label = 72
        label = label - 16
    elif label == 75:
        label = 73
        label = label - 16
    elif label == 76:
        label = 74
        label = label - 16
    elif label == 77:
        label = 75
        label = label - 16
    else:
        label = label - 16

    return img, label

@tf.function
def run(model, images, training=True):
    logits = model(images, training=training)
    return logits

def train_loss(model, images, labels):
    with tf.GradientTape() as tape:
        logits = run(model, images, True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def cal_MAE(logits, labels):

    logits = tf.nn.softmax(logits)
    predict_age = tf.cast(tf.argmax(logits, 1), tf.int32)
    predict_age = tf.reduce_sum(tf.abs(labels - predict_age))
    return predict_age

def main(argv=None):
    model = Recurrent_CNN(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch),
                          num_classes=FLAGS.num_classes,
                          weight_decay=FLAGS.weight_decay)
    model.summary()

    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restroed the checkpoint!! [* {} *]".format(ckpt_manager.latest_checkpoint))

    if FLAGS.train is True:
        count = 0
        # input data
        train_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        train_img = [FLAGS.img_path + data for data in train_img]
        train_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_img = np.loadtxt(FLAGS.val_txt_path, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img_path + data for data in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_gener = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
        val_gener = val_gener.map(func_val)
        val_gener = val_gener.batch(FLAGS.val_batch_size)
        val_gener = val_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img, train_lab))
            shuffle(A)
            train_img, train_lab = zip(*A)
            train_img, train_lab = np.array(train_img), np.array(train_lab)

            tr_gener = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            tr_gener = tr_gener.shuffle(len(train_lab))
            tr_gener = tr_gener.map(func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_iter = iter(tr_gener)
            train_idx = len(train_lab) // FLAGS.batch_size
            
            val_idx = len(val_lab) // FLAGS.val_batch_size

            for step in range(train_idx):

                batch_images, batch_labels = next(train_iter)

                total_loss = train_loss(model, batch_images, batch_labels)

                if count % 10 == 0:
                    AE = 0
                    val_iter = iter(val_gener)
                    for step_ in range(val_idx):
                        images, labels = next(val_iter)
                        predict = run(model, images, False)
                        predict = cal_MAE(predict, labels)
                        AE += predict
                    MAE = AE / len(val_lab)
                    print("Epochs : {} [{}/{}], loss = {}, MAE = {}({} steps)".format(epoch, step + 1, train_idx, total_loss, MAE, count))


                if (count + 1) % val_idx ==0:
                    AE = 0
                    val_iter = iter(val_gener)
                    for step_ in range(val_idx):
                        images, labels = next(val_iter)
                        predict = run(model, images, False)
                        predict = cal_MAE(predict, labels)
                        AE += predict
                    MAE = AE / len(val_lab)
                    print("=======================")
                    print("MAE = {}".format(MAE))
                    print("=======================")
                    model_dir = FLAGS.save_checkpoint
                    folder_name = int(count/(len(val_lab) // FLAGS.val_batch_size))
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print("Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(model=model,
                                                    optimizer=optimizer)
                    checkpoint_dir = folder_neme_str + "/" + "new_age_estimation_{}_steps.ckpt".format(count)
                    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

                    manager.save()
                    #with val_summary_writer.as_default():
                    #    tf.summary.scalar(u'MAE', AE / len(val_img), step=count)


                count += 1

if __name__ == "__main__":
    app.run(main)