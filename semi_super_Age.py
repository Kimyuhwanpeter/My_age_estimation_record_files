# -*- coding: utf-8 -*-
from absl import flags, app
from random import shuffle, random

import tensorflow as tf
import numpy as np
import datetime
import os
import sys

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Train text path")

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Train image path")

flags.DEFINE_string("val_txt", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Validation text path")

flags.DEFINE_string("val_img", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Validation image path")

flags.DEFINE_integer("img_size", 224, "Image height and width")

flags.DEFINE_integer("ch", 3, "Image channels")

flags.DEFINE_integer("batch_size", 8, "Training batch size")

flags.DEFINE_integer("val_batch", 128, "Validation batch size")

flags.DEFINE_integer("epochs", 100, "Total epochs in training")

flags.DEFINE_float("weight_decay", 0.0001, "L2 weight decay in every layer")

flags.DEFINE_float("lr", 0.01, "Learning rate")

flags.DEFINE_integer("num_classes", 7, "Number of classes")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Pre checkpoint path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "D:/tensorflor2.0(New_age_estimation)/checkpoint", "Saveing checkpoint path")

flags.DEFINE_string("graphs", "D:/tensorflor2.0(New_age_estimation)/graphs/", "Loss graphs (training and validation)")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optimizer = tf.keras.optimizers.Adadelta(FLAGS.lr)

# 1. 학습데이터에 관해 대표값 이미지를 구함 그러면 총 클래스 개수만큼의 이미지가 생김
# 2. 그리고 0부터 6까지의 라벨을 두고(예를들면 0은 16세에서 19세), 우선 크로스앤트로피로 로스계산
# 3. 그리고 라벨이 0일때의 logit값에 해당하는 대표값 이미지들 (16세부터 19세까지) 을 모델에 넣어서 그 로직을 원래 logits과의 관계를이용해
# loss를 계산

# 테스트할때도 학습데이터의 대표 이미지들을 구하고 (총 60개의 이미지 16부터 75까지) 
# 테스트이미지가 입력으로들어가서 출력이 0에서 6사이의 값이 나오면 그 값에 해당하는 대표값 범위를 비교
# 예를들면 테스트 입력이미지가 1이나오면 대표값 이미지는 20부터 29까지의 총 9개의 이미지가 나온다. 그래서 입력 피쳐와 9개의 출력 피쳐와 거리를
# 구해서 가장 작은값을 나이로 선정!

def replace_label(label):

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

    return label

def _mean_func(image):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    return img

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

    if label >= 16 and label < 20:
        label = 0
    elif label >= 20 and label < 30:
        label = 1
    elif label >= 30 and label < 40:
        label = 2
    elif label >= 40 and label < 50:
        label = 3
    elif label >= 50 and label < 60:
        label = 4
    elif label >= 60 and label < 70:
        label = 5
    elif label >= 70 and label < 80:
        label = 6

    return image, label

@tf.function
def run_model(model, images, training=False):
    logits, feature = model(images, training=training)
    return logits, feature

def cal_loss(model, images, labels, mean_img, prob_classes_img):
# 1. 학습데이터에 관해 대표값 이미지를 구함 그러면 총 클래스 개수만큼의 이미지가 생김
# 2. 그리고 0부터 6까지의 라벨을 두고(예를들면 0은 16세에서 19세), 우선 크로스앤트로피로 로스계산
# 3. 그리고 라벨이 0일때의 logit값에 해당하는 대표값 이미지들 (16세부터 19세까지) 을 모델에 넣어서 그 로직을 원래 logits과의 관계를이용해
# loss를 계산 --> 기억해 잊지마!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 코랩에 우선 케스케이드 방식으로 학습하고 결과 도출--> 나쁜 결과면 --> features만을 이용해서 loss 구성 즉, first loss를 뻄
# 아니면 케스케이드 방식에서 대표값에 대해서는 freezing 시켜놓고 학습!!--> 나쁜 결과면--> 위와 동일하게!
#(실험 loss 경향이 보기 안좋음, 학습이 안되고있는것 같았다.)

# 2020 5월 11일 오전 기준으로 loss를 다시 고치고 있다...
    logits, features = run_model(model, images, True)
    one_hot = tf.one_hot(labels, FLAGS.num_classes)
    first_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(one_hot, logits)

    loss = 0.
    for j in range(FLAGS.batch_size):
        feature = features[j]
        label = labels[j]
        if label.numpy() == 0:
            _,represent_value = run_model(model, mean_img[0:4], True)     # [4, y]
            for i in range(4):      #  여기서 i를 각 클래스 개수의 확률로 대체하면 어떻게 될까? --> 만일 다른 나이인데 같은확률이면 학습이 이상하게 된다
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i]) )
                # 아직 대체될것을 찾지못했다--> 내일할것 배치내에서의 각각의 나이를 고려한 평균과 분산을 고려한 분포를 만들고 이를 loss에 반영????
        elif label.numpy() == 1:
            _,represent_value = run_model(model, mean_img[4:14], True)     # [10, y]
            for i in range(10):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+4]) )
        elif label.numpy() == 2:
            _,represent_value = run_model(model, mean_img[14:24], True)     # [10, y]
            for i in range(10):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+14]) )
        elif label.numpy() == 3:
            _,represent_value = run_model(model, mean_img[24:34], True)     # [10, y]
            for i in range(10):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+24]) )
        elif label.numpy() == 4:
            _,represent_value = run_model(model, mean_img[34:44], True)     # [10, y]
            for i in range(10):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+34]) )
        elif label.numpy() == 5:
            _,represent_value = run_model(model, mean_img[44:54], True)     # [10, y]
            for i in range(10):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+44]) )
        elif label.numpy() == 6:
            _,represent_value = run_model(model, mean_img[54:60], True)     # [6, y]
            for i in range(6):
                loss += tf.reduce_mean( tf.abs(tf.abs(feature[0:100] - represent_value[i, 0:100]) - prob_classes_img[i+54]) )

    total_loss = (loss / FLAGS.batch_size) + first_loss

    return total_loss

def train_step(model, images, labels, mean_img, prob_classes_img):
    with tf.GradientTape() as tape:
        total_loss = cal_loss(model, images, labels, mean_img, prob_classes_img)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

def test_step(model, images, mean_img):
# 테스트할때도 학습데이터의 대표 이미지들을 구하고 (총 60개의 이미지 16부터 75까지) 
# 테스트이미지가 입력으로들어가서 출력이 0에서 6사이의 값이 나오면 그 값에 해당하는 대표값 범위를 비교
# 예를들면 테스트 입력이미지가 1이나오면 대표값 이미지는 20부터 29까지의 총 9개의 이미지가 나온다. 그래서 입력 피쳐와 9개의 출력 피쳐와 거리를
# 구해서 가장 작은값을 나이로 선정! --> 완료

    predict_range, predcit_feature = run_model(model, images, False)
    predict_range = tf.nn.softmax(predict_range)
    predict_range = tf.argmax(predict_range[0])

    if predict_range == 0:
        _,represent_value = run_model(model, mean_img[0:4], False)
        distance = []
        for i in range(4):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 16 + predict_age

    elif predict_range == 1:
        _,represent_value = run_model(model, mean_img[4:14], False)
        distance = []
        for i in range(10):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 20 + predict_age

    elif predict_range == 2:
        _,represent_value = run_model(model, mean_img[14:24], False)
        distance = []
        for i in range(10):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 30 + predict_age

    elif predict_range == 3:
        _,represent_value = run_model(model, mean_img[24:34], False)
        distance = []
        for i in range(10):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 40 + predict_age

    elif predict_range == 4:
        _,represent_value = run_model(model, mean_img[34:44], False)
        distance = []
        for i in range(10):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 50 + predict_age

    elif predict_range == 5:
        _,represent_value = run_model(model, mean_img[44:54], False)
        distance = []
        for i in range(10):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 60 + predict_age

    elif predict_range == 6:
        _,represent_value = run_model(model, mean_img[54:60], False)
        distance = []
        for i in range(6):
            distance.append(tf.reduce_sum(tf.abs(predcit_feature[0, 0:100] - represent_value[i, 0:100])))
        predict_age = tf.argmin(distance)
        predict_age = 70 + predict_age

    return predict_age

def main(argv=None):

    model = tf.keras.applications.MobileNetV2(
    input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), alpha=1.0, 
    include_top=False, weights='imagenet',input_tensor=None, pooling='avg')
    regularizer = tf.keras.regularizers.l2(FLAGS.weight_decay)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)
    h = model.output
    h2 = tf.keras.layers.Dense(FLAGS.num_classes)(h)
    model = tf.keras.Model(inputs=model.input, outputs=[h2, h])

    model.summary()
    
    if FLAGS.pre_checkpoint is True:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the checkpoint!! * {} *".format(ckpt_manager.latest_checkpoint))

    if FLAGS.train is True:
        count = 0;

        # input
        data_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        data_img = [FLAGS.img_path + data for data in data_img]
        data_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        val_img = np.loadtxt(FLAGS.val_txt, dtype="<U100", skiprows=0, usecols=0)
        val_img = [FLAGS.val_img + data for data in val_img]
        val_lab = np.loadtxt(FLAGS.val_txt, dtype=np.int32, skiprows=0, usecols=1)
        val_lab = [data - 2 if data > 71 else data for data in val_lab]
        val_lab = np.array(val_lab, np.int32) - 16

        age = np.arange(0, 60).astype(np.int32)
        represent_img = []
        prob_classes_img = []
        print("같은 클래스들을 서로 묶는중....")
        for i in range(len(age)):
            sum = 0
            c = 0
            age_buf = []
            for j in range(len(data_img)):
                image = data_img[j]
                label = data_lab[j]
                lab = replace_label(label)
                if lab == i:
                    age_buf.append(image)
                    c += 1
            print("=======================")
            print("Save {} age image...".format(i))
            print("=======================")
            represent_img.append(age_buf)
            #prob_classes_img.append( ((c / len(data_lab) ) * 100.))
            prob_classes_img.append( (1. - (c / len(data_lab) ) * 100.))
        represent_img = np.array(represent_img)
        prob_classes_img = np.array(prob_classes_img)
        print(np.sum(prob_classes_img))

        train_mean_buf = []
        print("==================================================")
        print("같은 클래스끼리 묶은 버퍼를 이용해 평균을 구하는 중...")
        print("==================================================")
        for i in range(len(represent_img)):
            mean_img = tf.data.Dataset.from_tensor_slices(represent_img[i])
            mean_img = mean_img.map(_mean_func)
            mean_img = mean_img.batch(1)
            mean_img = mean_img.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(mean_img)
            sum = 0.
            for j in range(len(represent_img[i])):
                image = next(it)
                sum += image
            
            print("Calculated the mean of images...{}".format(i + 1))
            image = sum / len(represent_img[i])
            #image = tf.reduce_mean(tf.abs(image))
            #print(image.numpy())
            #train_mean_buf.append(image)
            train_mean_buf.append(image[0].numpy())

        train_mean_buf = np.array(train_mean_buf).astype(np.float32)
        print("학습데이터 대표값(평균) 전체 shape: {}".format(train_mean_buf.shape))


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
                
                total_loss = train_step(model, batch_images, batch_labels, train_mean_buf, prob_classes_img)

                if (count + 1) % (val_idx + 1) == 0:
                    AE = 0
                    val_iter = iter(val_generation)
                    for i in range(val_idx):
                        val_images, val_labels = next(val_iter)
                        predict_age = test_step(model, val_images, train_mean_buf)
                        AE += tf.reduce_sum(tf.math.abs(val_labels.numpy() - predict_age))
                    print("===============================")
                    print("Epoch: {} [{}/{}] MAE = {}".format(epoch + 1, step + 1, batch_idx, AE / len(val_img)))
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
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch + 1, step + 1, batch_idx, total_loss))

                with train_summary_writer.as_default():
                    tf.summary.scalar(u'total loss', total_loss, step=count)

                count += 1

        



if __name__ == "__main__":
    app.run(main)