# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from Age_model_9 import *

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

flags.DEFINE_integer("batch_size", 45, "Train batch size")

flags.DEFINE_integer("num_classes", 60, "Number of classes (Train set)")

flags.DEFINE_integer("val_batch_size", 128, "Validation batch size")

flags.DEFINE_integer("epochs", 300, "Training total epochs")

flags.DEFINE_float("lr", 1e-4, "Train learning rate")

flags.DEFINE_float("weight_decay", 1e-4, "Train weight decay")

flags.DEFINE_bool("train", True, "Train or test")

flags.DEFINE_bool("pre_checkpoint", False, "Restore <test> or not")

flags.DEFINE_string("pre_checkpoint_path", "", "Pre checkpoint path")

flags.DEFINE_string("save_checkpoint_path", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Save checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Save graphs path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

def val_func(data, label):
    img = tf.io.read_file(data)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.flip_left_right(img)
        
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

def func(data, label):
    img = tf.io.read_file(data)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.flip_left_right(img)
        
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

@tf.function
def cal_MAE(logits, labels):
    
    logits = tf.nn.softmax(logits)
    predict_age = tf.cast(tf.argmax(logits, 1), tf.int32)
    AE = tf.reduce_sum(tf.abs(labels - predict_age))

    return AE

def main(argv=None):
    model = My_compact_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch),
                             num_classes=FLAGS.num_classes,
                             weight_decay=FLAGS.weight_decay)
    model.summary()

    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 3)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Success restoring the checkpoint")

    if FLAGS.train is True:
        count = 0
        train_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        train_img = [FLAGS.img_path + img for img in train_img]
        train_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_img = np.loadtxt(FLAGS.val_txt_path, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img_path + img for img in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_gener = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
        val_gener = val_gener.map(val_func)
        val_gener = val_gener.batch(FLAGS.val_batch_size)
        val_gener = val_gener.prefetch(tf.data.experimental.AUTOTUNE)

        train_idx = len(train_lab) // FLAGS.batch_size
        val_idx = len(val_lab) // FLAGS.val_batch_size

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for ep in range(FLAGS.epochs):
            
            A = list(zip(train_img, train_lab))
            shuffle(A)
            train_img, train_lab = np.array(train_img), np.array(train_lab)

            train_gener = tf.data.Dataset.from_tensor_slices((train_img, train_lab)) 
            train_gener = train_gener.shuffle(len(train_lab))
            train_gener = train_gener.map(func)
            train_gener = train_gener.batch(FLAGS.batch_size)
            train_gener = train_gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_iter = iter(train_gener)
            for step in range(train_idx):
                batch_images, batch_labels = next(train_iter)

                loss = train_loss(model, batch_images, batch_labels)

                with val_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', loss, step=count)

                if count % 10 == 0:
                    AE_ = 0
                    val_iter = iter(val_gener)
                    for i in range(val_idx):
                        val_imgs, val_ages = next(val_iter)
                        logits = run(model, val_imgs, False)
                        AE_ += cal_MAE(logits, val_ages)
                    MAE = AE_ / len(val_lab)
                    print("Epoch: {} [{}/{}] Loss = {}, MAE = {} (per 10 steps)".format(ep, step + 1, train_idx, loss, MAE))

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
