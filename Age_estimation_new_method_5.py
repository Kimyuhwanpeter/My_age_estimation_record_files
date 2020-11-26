# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from Age_model_5 import *

import tensorflow as tf
import numpy as np
import datetime
import sys
import os

flags.DEFINE_integer("img_size", 224, "Image height and width")

flags.DEFINE_integer("ch", 3, "Image channel")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Train text path")

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Train image path")

flags.DEFINE_string("val_txt", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_string("val_img", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_integer("batch_size", 64, "Train batch size")

flags.DEFINE_integer("val_batch", 128, "Validation batch size")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_float("weight_decay", 0.0001, "Weight decay (L2)")

flags.DEFINE_float("lr", 0.001, "Learning rate")

flags.DEFINE_integer("num_classes", 60, "Number of classes")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Saveing checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Loss graphs (training and validation)")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

def _test_func(image, label):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

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
def run_model(model, images, training=False):
    logits = model(images, training=training)
    return logits

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:

        logits = model(images, training=True)

        total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

def main(argv=None):
    Model = model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), filters=32, weight_decay=FLAGS.weight_decay, num_classes=FLAGS.num_classes)
    Model.summary()

    # pre checkpoint
    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(Model=Model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Success restoring the checkpotin * {} *".format(ckpt_manager.latest_checkpoint))

    # input
    data_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
    data_img = [FLAGS.img_path + data for data in data_img]
    data_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

    if FLAGS.train is True:
        count = 0
        
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

        for epoch in range(FLAGS.epochs):
            A = list(zip(data_img, data_lab))
            shuffle(A)
            data_img, data_lab = zip(*A)
            data_img, data_lab = np.array(data_img), np.array(data_lab)

            tr_generation = tf.data.Dataset.from_tensor_slices((data_img, data_lab))
            tr_generation = tr_generation.shuffle(len(data_lab))
            tr_generation = tr_generation.map(_func)
            tr_generation = tr_generation.batch(FLAGS.batch_size)
            tr_generation = tr_generation.prefetch(tf.data.experimental.AUTOTUNE)

            val_generation = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
            val_generation = val_generation.map(_test_func)
            val_generation = val_generation.batch(FLAGS.val_batch)
            val_generation = val_generation.prefetch(tf.data.experimental.AUTOTUNE)

            batch_idx = len(data_lab) // FLAGS.batch_size
            val_idx = len(val_lab) // FLAGS.val_batch
            tr_iter = iter(tr_generation)
            for step in range(batch_idx):
                batch_images, batch_labels = next(tr_iter)
                
                total_loss = train_step(Model, batch_images, batch_labels)

                if (count + 1) % (val_idx + 1) == 0:
                    AE = 0
                    val_iter = iter(val_generation)
                    for i in range(val_idx):
                        val_images, val_labels = next(val_iter)
                        predict_age = run_model(model=Model, images=val_images, training=False)
                        predict_age = tf.argmax(predict_age, 1)
                        AE += tf.reduce_sum(tf.math.abs(val_labels.numpy() - predict_age.numpy()))
                    print("===============================")
                    print("Epoch: {} [{}/{}] MAE = {}".format(epoch + 1, step + 1, batch_idx, AE / len(val_img)))
                    print("===============================")
                    model_dir = FLAGS.save_checkpoint
                    folder_name = int(count/(len(val_lab) // FLAGS.val_batch))
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print("Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(Model=Model,
                                                    optimizer=optimizer)
                    checkpoint_dir = folder_neme_str + "/" + "new_age_estimation_{}_steps.ckpt".format(count)
                    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

                    manager.save()
                    with val_summary_writer.as_default():
                        tf.summary.scalar(u'MAE', AE / len(val_img), step=count)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch + 1, step + 1, batch_idx, total_loss))

                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', total_loss, step=count)

                count += 1




if __name__ == "__main__":
    app.run(main)