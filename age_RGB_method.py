# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random
from keras_radam.training import RAdamOptimizer
from age_RGB_model import *

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("val_img", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_string("val_txt", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_integer("img_size", 224, "Image size(height and width)")

flags.DEFINE_integer("ch", 3, "Channels")

flags.DEFINE_integer("num_classes", 60, "Number of classes")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("val_batch", 8, "Validation batch size")

flags.DEFINE_integer("epochs", 100, "Total epochs")

flags.DEFINE_float("lr", 0.0001, "Learning rate")

flags.DEFINE_float("weight_decay", 0.0001, "L2 weight decay")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Tensorflow graphs path")

flags.DEFINE_bool("train", True, "Train or Test")

flags.DEFINE_bool("pre_checkpoint", False, "Continue training <pre training> or testing")

flags.DEFINE_string("pre_checkpoint_path", "", "load the checkpoint path")

flags.DEFINE_string("save_checkpoint", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Save the checkpoint path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

#optimizer = RAdamOptimizer(FLAGS.lr)
optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

# MAE = 3.84 for 1st fold

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

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = run(model, images, True)

        total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

@tf.function()
def run(model, images, training=True):
    logits = model(images, training=training)
    return logits

def main(argv=None):

    model = RGB_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), weight_decay=FLAGS.weight_decay, num_classes=FLAGS.num_classes)
    
    model.summary()

    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 3)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!!!!-->[ {} ]".format(ckpt_manager.latest_checkpoint))

    if FLAGS.train is True:
        data_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        data_img = [FLAGS.img_path + img for img in data_img]
        data_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)\

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
        
        # writing the model flow (graphs)
        model_graphs = FLAGS.graphs + 'model'
        writer = tf.summary.create_file_writer(model_graphs)
        x = tf.random.uniform((1, FLAGS.img_size, FLAGS.img_size,3))

        tf.summary.trace_on()
        y = run(model, x, True)
        with writer.as_default():
            tf.summary.trace_export(name="MODEL",step=0)

        count = 0;
        for ep in range(FLAGS.epochs):
            A = list(zip(data_img, data_lab))
            shuffle(A)
            data_img, data_lab = zip(*A)
            data_img, data_lab = np.array(data_img), np.array(data_lab)

            data_gen = tf.data.Dataset.from_tensor_slices((data_img, data_lab))
            data_gen = data_gen.shuffle(len(data_lab))
            data_gen = data_gen.map(_func)
            data_gen = data_gen.batch(FLAGS.batch_size)
            data_gen = data_gen.prefetch(tf.data.experimental.AUTOTUNE)

            val_gen = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
            val_gen = val_gen.map(_test_func)
            val_gen = val_gen.batch(FLAGS.val_batch)
            val_gen = val_gen.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = len(data_lab) // FLAGS.batch_size
            it_train = iter(data_gen)

            val_idx = len(val_lab) // FLAGS.val_batch

            for step in range(train_idx):
                batch_images, batch_labels = next(it_train)

                total_loss = train_step(model, batch_images, batch_labels)
                    
                if (count + 1) % (val_idx + 1) == 0:
                    AE = 0
                    val_iter = iter(val_gen)
                    for i in range(val_idx):
                        val_images, val_labels = next(val_iter)
                        predict_age = run(model, val_images, training=False)
                        predict_age = tf.argmax(predict_age, 1)
                        AE += tf.reduce_sum(tf.math.abs(val_labels.numpy() - predict_age.numpy()))
                    print("===============================")
                    print("Epoch: {} [{}/{}] MAE = {}".format(ep + 1, step + 1, train_idx, AE / len(val_img)))
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
                    print("Epoch: {} [{}/{}] total_loss = {}".format(ep + 1, step + 1, train_idx, total_loss))

                count +=1


if __name__ == "__main__":
    app.run(main)