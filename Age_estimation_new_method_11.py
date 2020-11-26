# -*- coding:utf-8 -*-
from absl import flags, app
from random import random
from Age_model_11 import *
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

flags.DEFINE_integer("val_batch_size", 32, "Validation batch size")

flags.DEFINE_integer("epochs", 300, "Training total epochs")

flags.DEFINE_float("lr", 0.01, "Train learning rate")

flags.DEFINE_float("weight_decay", 0.0005, "Train weight decay")

flags.DEFINE_bool("train", True, "Train or test")

flags.DEFINE_bool("pre_checkpoint", False, "Restore <test> or not")

flags.DEFINE_string("pre_checkpoint_path", "", "Pre checkpoint path")

flags.DEFINE_string("save_checkpoint_path", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Save checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Save graphs path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.SGD(FLAGS.lr)

def train_func(file_list, lab_list):

    img = tf.io.read_file(file_list)
    img = tf.image.decode_jpeg(img, FLAGS.ch)
    img = tf.image.resize(img, [FLAGS.img_size,FLAGS.img_size])
    
    if random() > 0.5:
        img = tf.image.per_image_standardization(img)

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

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

@tf.function
def cal_MAE(logits, labels):
    
    logits_ = tf.argmax(logits, 1, output_type=tf.int32)
    AE_ = tf.reduce_sum(tf.abs(labels - logits_))

    logits = tf.nn.sigmoid(logits)
    logits = tf.argmax(logits, 1, output_type=tf.int32)
    AE = tf.reduce_sum(tf.abs(labels - logits))

    return AE,AE_

def train_step(model, images, labels, age):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        #logits = tf.math.sigmoid(logits)
        #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)()

        loss = 0
        for i in range(FLAGS.batch_size):

            if age[i] > FLAGS.num_classes - (FLAGS.num_classes - 1) and age[i] < FLAGS.num_classes - 2:
                loss1 = tf.reduce_mean( (tf.maximum(logits[i][age[i]], 0) - logits[i][age[i]] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i]])))) )
                loss2 = tf.reduce_mean( (tf.maximum(logits[i][age[i] - 1], 0) - logits[i][age[i] - 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] - 1])))) )
                loss3 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 1], 0) - logits[i][age[i] + 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 1])))) )
                loss4 = tf.reduce_mean( (tf.maximum(logits[i][0: age[i] - 1], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][0: age[i] - 1]))))*5.0 )
                loss5 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 2:], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 2:]))))*5.0 )
                loss += loss1 + loss2 + loss3 + loss4 + loss5
            elif age[i] == FLAGS.num_classes - FLAGS.num_classes:
                loss1 = tf.reduce_mean( (tf.maximum(logits[i][age[i]], 0) - logits[i][age[i]] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i]])))) )
                loss2 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 1], 0) - logits[i][age[i] + 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 1])))) )
                loss3 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 2:], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 2:]))))*5.0 )
                loss += loss1 + loss2 + loss3
            elif age[i] == FLAGS.num_classes - (FLAGS.num_classes - 1):
                loss1 = tf.reduce_mean( (tf.maximum(logits[i][age[i]], 0) - logits[i][age[i]] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i]])))) )
                loss2 = tf.reduce_mean( (tf.maximum(logits[i][age[i] - 1], 0) - logits[i][age[i] - 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] - 1])))) )
                loss3 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 1], 0) - logits[i][age[i] + 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 1])))) )
                loss4 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 2:], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 2:]))))*5.0 )
                loss += loss1 + loss2 + loss3 + loss4
            elif age[i] == FLAGS.num_classes - 2:
                loss1 = tf.reduce_mean( (tf.maximum(logits[i][age[i]], 0) - logits[i][age[i]] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i]])))) )
                loss2 = tf.reduce_mean( (tf.maximum(logits[i][age[i] - 1], 0) - logits[i][age[i] - 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] - 1])))) )
                loss3 = tf.reduce_mean( (tf.maximum(logits[i][age[i] + 1], 0) - logits[i][age[i] + 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] + 1])))) )
                loss4 = tf.reduce_mean( (tf.maximum(logits[i][0: age[i] - 1], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][0: age[i] - 1]))))*5.0 )
                loss += loss1 + loss2 + loss3 + loss4
            elif age[i] == FLAGS.num_classes - 1:
                loss1 = tf.reduce_mean( (tf.maximum(logits[i][age[i]], 0) - logits[i][age[i]] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i]])))) )
                loss2 = tf.reduce_mean( (tf.maximum(logits[i][age[i] - 1], 0) - logits[i][age[i] - 1] + tf.math.log(1 + tf.exp(-tf.abs(logits[i][age[i] - 1])))) )
                loss3 = tf.reduce_mean( (tf.maximum(logits[i][0:FLAGS.num_classes - 2], 0) + tf.math.log(1 + tf.exp(-tf.abs(logits[i][0:FLAGS.num_classes - 2]))))*5.0 )
                loss += loss1 + loss2 + loss3

        loss /= FLAGS.batch_size

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def make_label(labels):
    # [0,1,0,0,0] --> [1,1,0,0,0] --> [0,1,0,-1,-1,-1]
    levels = []
    for i in range(FLAGS.batch_size):
        if labels[i].numpy() > 0 and labels[i].numpy() < 59:
            l = [-1] * (labels[i].numpy() - 1) + [0] + [1] + [0] + [-1]*(FLAGS.num_classes - (labels[i].numpy()+2))
        elif labels[i].numpy() == 0:
            l = [1] + [0] + [-1]*(FLAGS.num_classes - 2)
        else:
            l = [-1]*(FLAGS.num_classes - 2) + [0] + [1]

        l = tf.cast(l, tf.float32)
        levels.append(l)
    label = tf.convert_to_tensor(levels, tf.float32)

    return label

def main(argv=None):

    #model = Resnext_50
    model = model_fit(input_shape=(FLAGS.img_size,FLAGS.img_size,FLAGS.ch), num_classes=FLAGS.num_classes, weight_decay=FLAGS.weight_decay)
    model.summary()

    # Restore or not
    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint file!!!!!!!")

    if FLAGS.train is True:

        # Input
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
        count = 0

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        for ep in range(FLAGS.epochs):
            
            train_gener = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            train_gener = train_gener.shuffle(len(train_lab))
            train_gener = train_gener.map(train_func)
            train_gener = train_gener.batch(FLAGS.batch_size)
            train_gener = train_gener.prefetch(tf.data.experimental.AUTOTUNE)
        
            train_iter = iter(train_gener)
            for step in range(train_idx):
                batch_images, batch_labels = next(train_iter)
                batch_labels_ = make_label(batch_labels)

                loss = train_step(model, batch_images, batch_labels_, batch_labels)
                L = int(round(50 * step / float(train_idx - 1)))
                bar = ">" * L + "-" * (50 - L)

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
                    #MAE = 0
                    #print("Epoch: {} [{}/{}] Loss = {} (per 10 steps)".format(ep, step + 1, train_idx, loss))
                    print("Epoch: {}, step: |{}| {}, loss = {}".format(ep, bar, step + 1, loss))

                if (count + 1) % val_idx == 0:
                    AE = 0
                    AE_ = 0
                    val_iter = iter(val_gener)
                    for i in range(val_idx):
                        val_imgs, val_ages = next(val_iter)
                        logits = run_model(model, val_imgs, False)
                        output, output_ = cal_MAE(logits, val_ages)
                        AE += output
                        AE_ += output_
                    MAE = AE / len(val_lab)
                    MAE_ = AE_ / len(val_lab)
                    print("==========")
                    print("MAE = {}".format(MAE))
                    print("MAE (without ac) = {}".format(MAE_))
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