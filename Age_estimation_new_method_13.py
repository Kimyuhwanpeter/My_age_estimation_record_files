# -*- coding:utf-8 -*-
from random import random, shuffle
from absl import flags
from sklearn.decomposition import PCA

import tensorflow as tf
import numpy as np
import sys
import os

flags.DEFINE_integer("img_size", 299, "Height and Width")

flags.DEFINE_integer("img_ch", 3, "Channels")

flags.DEFINE_string('txt_path', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt', 'Training text path')

flags.DEFINE_string('img_path', 'D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/', 'Training image path')

flags.DEFINE_integer("batch_size", 16, "Batch size")

flags.DEFINE_integer("classes", 60, "Number of classes")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_bool("pre_checkpoint", False, "Saved checkpoint path")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("te_txt_path", "", "Test text path")

flags.DEFINE_string("te_img_path", "", "Test image path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def func(im, lab):

    img = tf.io.read_file(im)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.per_image_standardization(img)

    if lab == 74:
        lab = 72
        lab = lab - 16
    elif lab == 75:
        lab = 73
        lab = lab - 16
    elif lab == 76:
        lab = 74
        lab = lab - 16
    elif lab == 77:
        lab = 75
        lab = lab - 16
    else:
        lab = lab - 16

    return img, lab

def train_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.per_image_standardization(img)

    img = tf.image.per_image_standardization(img)

    if lab_list == 74:
        lab_list = 72
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.classes)
    elif lab_list == 75:
        lab_list = 73
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.classes)
    elif lab_list == 76:
        lab_list = 74
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.classes)
    elif lab_list == 77:
        lab_list = 75
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.classes)
    else:
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.classes)

    return img, lab

def test_func(img_list, lab_list):

    img = tf.io.read_file(img_list)

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, input_pca, images, labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True) # batch x 100
        in_loss = 0.
        de_loss = 0.
        loss = 0.
        for i in range(FLAGS.batch_size):
            logits_ = logits[i]
            labels_ = labels[i]
            #print(labels_.numpy())

            distance = tf.reduce_mean(tf.abs(logits_ - input_pca), 1)
            #less_distance = tf.reduce_min(distance) # 최소 거리값은 구했다!
            distance_arg = tf.argmin(distance)
            label_arg = (tf.argmax(labels_)).numpy()
            #print(label_arg)

            decrease_loss = 0.
            for j in range(FLAGS.classes):
                if distance[label_arg] != distance[j]:
                    decrease_loss += (-distance[j]-1)/(1 - tf.math.exp(0.2*distance[j]) + 0.000001)  # 유사도가 낮은것 !!
                else:
                    increas_loss = tf.math.exp(distance[label_arg] - 1.5)       # 유사도가 높은것 !!

            loss += (decrease_loss + increas_loss) / FLAGS.classes  # 이게 지금 loss감소가 아이에 이뤄지지 않는다 --> 거의 그대로이다

        loss /= FLAGS.batch_size
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def cal_distance(model, final_pca, img, lab):

    logits_ = run_model(model, images, True)
    distance = tf.reduce_mean(tf.abs(logits_ - final_pca), 1)
    predict = tf.argmin(distance)

    AE = tf.abs(predict - lab[0])

    return AE

def main():

    model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch), pooling="avg")
    h = model.output
    h = tf.keras.layers.Dense(100)(h)
    model = tf.keras.Model(inputs=model.input, outputs=h)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("============")
            print("Restored!!!!")
            print("============")

    tr_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
    tr_img = [FLAGS.img_path + img for img in tr_img]
    tr_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

    gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
    gener = gener.map(func)
    gener = gener.batch(1)
    gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

    it = iter(gener)
    age_buf = np.zeros(shape=[FLAGS.classes, FLAGS.img_size, FLAGS.img_size], dtype=np.float32)
    age_count = np.zeros(shape=[FLAGS.classes], dtype=np.int32)
    for i in range(len(tr_img)):
        imgs, labs = next(it)
        age_buf[labs.numpy()] += imgs[0, ..., 0]
        age_count[labs.numpy()] += 1
        if i % 100 == 0:
            print(i)

    final_pca = np.zeros(shape=[FLAGS.classes, 100], dtype=np.float32)
    for j in range(FLAGS.classes):

        age_buf[j] = age_buf[j] / age_count[j]
        pca = PCA(n_components=100)
        img_pca = pca.fit(age_buf[j])
        eigen_value_img_pca = img_pca.explained_variance_
        std = tf.math.reduce_std(eigen_value_img_pca)
        mean = tf.reduce_mean(eigen_value_img_pca)

        for i in range(100):
            final_pca[j][i] = tf.exp( -(eigen_value_img_pca[i] - mean)**2 / (2.*std*std) )

    if FLAGS.train:
        count = 0

        for epoch in range(FLAGS.epochs):

            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                # loss를 구할 때, 위에서 구한 final_pca와 model 출력을 각각 거리값을 구한 후, 가장 작게 나온 거리 값은 증가함수에 적용, 나머지는 감소함수
                # 에 적용한다! 기억해!!! 이렇게 loss를 구성해야함!! ( 내일해!!!!!!!)

                loss = cal_loss(model, final_pca, batch_images, batch_labels)

                if count % 10 == 0:
                    print("Epoch(s): {} [{}/{}] total_loss = {}".format(epoch, step + 1, tr_idx, loss))

                #if count % 1000 == 0:
                #    num_ = int(count) // 1000
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)

                #    ckpt = tf.train.Checkpoint(model=model, optim=optim)

                #    ckpt_dir = model_dir + "/" + "New_age_estimation_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)


                count += 1
    else:
        te_img = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.float32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        te_iter = iter(te_gener)
        for i in range(len(te_img)):
            img, lab = next(te_iter)

            acc_count = cal_distance(model, final_pca, img, lab)

if __name__ == "__main__":
    main()