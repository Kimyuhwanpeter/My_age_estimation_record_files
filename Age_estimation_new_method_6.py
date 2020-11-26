# -*- coding:utf-8 -*-
from absl import flags, app
from Age_model_6 import *
from random import shuffle, random
from collections import Counter

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("val_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_string("val_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_string("test_img_path", "", "Test image path")

flags.DEFINE_string("test_txt_path", "", "Test text path")

flags.DEFINE_integer("img_size", 224, "Image width and height")

flags.DEFINE_integer("batch_size", 40, "Train batch size")

flags.DEFINE_integer("val_batch_size", 128, "Validation batch size")

flags.DEFINE_integer("test_batch_size", 64, "Test batch size")

flags.DEFINE_integer("epochs", 200, "Total training epochs")

flags.DEFINE_integer("num_classes", 60, "Number of classes")

flags.DEFINE_float("lr", 2e-5, "Learning rate")

flags.DEFINE_float("weight_decay", 2e-5, "Regularization term")

flags.DEFINE_bool("train", True, "True - train, False - test")

flags.DEFINE_bool("pre_checkpoint", False, "True - Start latest training and restore, False - keep training")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Saving the checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Graphs path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

def _test_func(image, label):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img)
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

def _func(image, label):

    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        image = tf.image.flip_left_right(image)

    image = tf.image.per_image_standardization(image)

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

    return image, label

@tf.function
def run(model, images, training=True):
    logits = model(images, training=training)
    return logits

@tf.function
def MAE(logits, labels):
    
    logits = tf.nn.softmax(logits)
    logits = tf.cast(tf.argmax(logits, 1), tf.int32)
    AE = tf.reduce_sum(tf.abs(logits - labels))

    return AE

def cal_loss(logits, labels):
    total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    # logits에 대해 뭔가 독립적으로 만들어서 loss를 생성하는게 좀 더 효율적일 것 같다.
    #loss = 0.

    #for i in range(FLAGS.num_classes):
    #    logit = logits[:, i]
    #    label = labels[:, i]
    #    loss += tf.keras.losses.BinaryCrossentropy(from_logits=True)(label, logit)

    #total_loss = loss / FLAGS.batch_size

    return total_loss

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = run(model, images, True)
        total_loss = cal_loss(logits, labels)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

def main(argv=None):
    model = compact_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                          weight_decay=FLAGS.weight_decay,
                          num_classes=FLAGS.num_classes)

    model.summary()

    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 3)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Resotred the latest checkpoint path")

    if FLAGS.train is True:

        train_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        train_img = [FLAGS.img_path + data for data in train_img]
        train_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        N_train_label = Counter(train_lab)
        
        prob_train_age = []
        for k in range(FLAGS.num_classes):
            if k >= 56:
                prob_train_age.append( (N_train_label[k+16+2]) / len(train_lab) )
            else:
                prob_train_age.append( (N_train_label[k+16]) / len(train_lab) )
        #print(tf.nn.softmax(prob_train_age))
        #print(tf.math.sigmoid(prob_train_age))
        val_img = np.loadtxt(FLAGS.val_txt_path, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img_path + data for data in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)
        count = 0

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img, train_lab))
            shuffle(A)
            train_img, train_lab = zip(*A)
            train_img, train_lab = np.array(train_img), np.array(train_lab)

            tr_generation = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            tr_generation = tr_generation.shuffle(len(train_lab))
            tr_generation = tr_generation.map(_func)
            tr_generation = tr_generation.batch(FLAGS.batch_size)
            tr_generation = tr_generation.prefetch(tf.data.experimental.AUTOTUNE)

            val_generation = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
            val_generation = val_generation.map(_test_func)
            val_generation = val_generation.batch(FLAGS.val_batch_size)
            val_generation = val_generation.prefetch(tf.data.experimental.AUTOTUNE)

            train_batch_idx = len(train_lab) // FLAGS.batch_size
            train_iter = iter(tr_generation)

            val_batch_idx = len(val_lab) // FLAGS.val_batch_size

            for step in range(train_batch_idx):
                batch_images, batch_labels = next(train_iter)
                
                total_loss = train_step(model, batch_images, batch_labels)

                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', total_loss, step=count)

                if count % 10 == 0:
                    val_iter = iter(val_generation)
                    AE = 0
                    for i in range(val_batch_idx):
                        val_images, val_labels = next(val_iter)
                        predict = run(model, val_images, False)
                        AE += MAE(predict, val_labels)
                    with val_summary_writer.as_default():
                        tf.summary.scalar(u'MAE(per 10 step)', AE / len(val_img), step=count)
                    print("Epoch: {}[{}/{}] Total loss = {}, MAE (per 10 steps) = {} (total step: {})".format(epoch, step + 1, train_batch_idx, total_loss, AE / len(val_img), count + 1))

                if (count + 1) % val_batch_idx == 0:
                    val_iter = iter(val_generation)
                    AE = 0
                    for i in range(val_batch_idx):
                        val_images, val_labels = next(val_iter)
                        predict = run(model, val_images, False)
                        AE += MAE(predict, val_labels)
                    print("==================================")
                    print("MAE = {}".format(AE / len(val_lab)))
                    print("==================================")
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
                    with val_summary_writer.as_default():
                        tf.summary.scalar(u'MAE', AE / len(val_img), step=count)

                count += 1

if __name__ == "__main__":
    app.run(main)