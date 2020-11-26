# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from Age_model_10 import *

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Train images path")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Train texts path")

flags.DEFINE_string("val_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation images path")

flags.DEFINE_string("val_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation texts path")

flags.DEFINE_integer("img_size", 224, "Image width and height")

flags.DEFINE_integer("ch", 3, "Image channels")

flags.DEFINE_integer("batch_size", 128, "Train batch size")

flags.DEFINE_integer("num_classes", 60, "Number of classes (Train set)")

flags.DEFINE_integer("val_batch_size", 128, "Validation batch size")

flags.DEFINE_integer("epochs", 300, "Training total epochs")

flags.DEFINE_float("lr", 0.01, "Train learning rate")

flags.DEFINE_float("weight_decay", 1e-3, "Train weight decay")

flags.DEFINE_bool("train", True, "Train or test")

flags.DEFINE_bool("pre_checkpoint", False, "Restore <test> or not")

flags.DEFINE_string("pre_checkpoint_path", "", "Pre checkpoint path")

flags.DEFINE_string("save_checkpoint_path", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Save checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Save graphs path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

#optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
optimizer = tf.keras.optimizers.SGD(FLAGS.lr)

def train_func(data, label):

    img = tf.io.read_file(data)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    if label == 74:
        label = (label - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 75:
        label = (label - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 76:
        label = (label - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif label == 77:
        label = (label - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    else:
        label = label - 16
        label = tf.one_hot(label, FLAGS.num_classes)

    return img, label

def val_func(data, label):
    img = tf.io.read_file(data)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    if label == 74:
        label = (label - 2) - 16
    elif label == 75:
        label = (label - 2) - 16
    elif label == 76:
        label = (label - 2) - 16
    elif label == 77:
        label = (label - 2) - 16
    else:
        label = label - 16
    
    return img, label

def compact_input_img(img):     # 이게 만일 tensorflow transform이 된다면 필요가 없어진다!!!

    img_r = img[:, :, :, 0]
    img_g = img[:, :, :, 1]
    img_b = img[:, :, :, 2]

    img_r = tf.reshape(img_r, [FLAGS.batch_size, FLAGS.img_size * FLAGS.img_size])
    img_g = tf.reshape(img_g, [FLAGS.batch_size, FLAGS.img_size * FLAGS.img_size])
    img_b = tf.reshape(img_b, [FLAGS.batch_size, FLAGS.img_size * FLAGS.img_size])

    return X

@tf.function
def run(model, images, training=True):
    logits = model(images, training=training)
    return logits

@tf.function
def cal_MAE(logits, labels):
    
    logits = tf.nn.softmax(logits)
    predict_age = tf.cast(tf.argmax(logits, 1), tf.int32)
    AE = tf.reduce_sum(tf.abs(labels - predict_age))

    return AE
    
def train_loss(images, labels, model):
    with tf.GradientTape() as tape:
        logits = run(model, images, True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    gradents = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradents, model.trainable_variables))
    return loss

def main(argv=None):

    model = model_fit(input_shape=(FLAGS.img_size,FLAGS.img_size,FLAGS.ch), num_classes=FLAGS.num_classes, weight_decay=FLAGS.weight_decay)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored succeess!!")

    if FLAGS.train:

        data_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        data_img = [FLAGS.img_path + img for img in data_img]
        data_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_img = np.loadtxt(FLAGS.val_txt_path, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img_path + img for img in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_gener = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
        val_gener = val_gener.map(val_func)
        val_gener = val_gener.batch(FLAGS.val_batch_size)
        val_gener = val_gener.prefetch(tf.data.experimental.AUTOTUNE)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        count = 0
        for ep in range(FLAGS.epochs):

            A = list(zip(data_img, data_lab))
            shuffle(A)
            data_img, data_lab = np.array(data_img), np.array(data_lab)

            gener = tf.data.Dataset.from_tensor_slices((data_img, data_lab))
            gener = gener.shuffle(len(data_lab))
            gener = gener.map(train_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_it = iter(gener)
            train_idx = len(data_lab) // FLAGS.batch_size
            val_idx = len(val_lab) // FLAGS.val_batch_size

            for step in range(train_idx):
                batch_images, batch_labels = next(train_it)

                loss = train_loss(batch_images, batch_labels, model)

                with val_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', loss, step=count)

                if count % 10 == 0:
                    #AE_ = 0
                    #val_iter = iter(val_gener)
                    #for i in range(val_idx):
                    #    val_imgs, val_ages = next(val_iter)
                    #    logits = run(model, val_imgs, False)
                    #    AE_ += cal_MAE(logits, val_ages)
                    #MAE = AE_ / len(val_lab)
                    MAE = 0
                    print("Epoch: {} [{}/{}] Loss = {}, MAE = {} (per 10 steps)".format(ep, step + 1, train_idx, loss, MAE))
                    print("Epoch: {}, step: |{}| {}, loss = {}".format(ep, bar, step + 1, loss))

                if (count + 1) % val_idx == 0:
                    AE = 0
                    val_iter = iter(val_gener)
                    for i in range(val_idx):
                        val_imgs, val_ages = next(val_iter)
                        logits = run(model, val_imgs, False)
                        AE += cal_MAE(logits, val_ages)
                    MAE = AE / len(val_lab)
                    print("==========")
                    print("MAE = {}".format(MAE))
                    print("==========")

                    model_dir = FLAGS.save_checkpoint_path
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

                    with val_summary_writer.as_default():
                        tf.summary.scalar(u'MAE', AE / len(val_img), step=count)

                count += 1


if __name__ == "__main__":
    app.run(main)