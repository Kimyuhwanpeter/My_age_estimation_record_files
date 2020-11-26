# -*- coding: utf-8 -*-
from absl import flags, app
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

import numpy as np
import sys

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/age_labels.txt", "Text path")

flags.DEFINE_string("train_txt", "D:/[1]DB/[1]second_paper_DB/original_MORPH/80_20/train", "write train txt")

flags.DEFINE_string("test_txt", "D:/[1]DB/[1]second_paper_DB/original_MORPH/80_20/test", "write test txt")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main(agrv=None):

    data_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
    data_lab = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=1)

    data = list(zip(data_img, data_lab))

    #yj = PowerTransformer(method = 'yeo-johnson')
    #yj_data = yj.fit_transform(data)

    # 80 % - train dataset, 20 % - test dataset
    # 5-fold 로 나눠야한다!

    write_train1 = open(FLAGS.train_txt + "_1.txt", "w")
    write_train2 = open(FLAGS.train_txt + "_2.txt", "w")
    write_train3 = open(FLAGS.train_txt + "_3.txt", "w")
    write_train4 = open(FLAGS.train_txt + "_4.txt", "w")
    write_train5 = open(FLAGS.train_txt + "_5.txt", "w")

    write_test1 = open(FLAGS.test_txt + "_1.txt", "w")
    write_test2 = open(FLAGS.test_txt + "_2.txt", "w")
    write_test3 = open(FLAGS.test_txt + "_3.txt", "w")
    write_test4 = open(FLAGS.test_txt + "_4.txt", "w")
    write_test5 = open(FLAGS.test_txt + "_5.txt", "w")

    for age in range(62):
        age = age + 16
        buf_img = []
        buf_lab = []
        count = 0
        for i in range(len(data_img)):

            if age == int(data_lab[i]):
                buf_img.append(data_img[i])
                buf_lab.append(data_lab[i])
                count += 1

        te = int(count * 0.2)
        for k in range(count):
        
            if k < te:
                write_test1.write(buf_img[k])
                write_test1.write(" ")
                write_test1.write(buf_lab[k])
                write_test1.write("\n")
                write_test1.flush()
            else:
                write_train1.write(buf_img[k])
                write_train1.write(" ")
                write_train1.write(buf_lab[k])
                write_train1.write("\n")
                write_train1.flush()

        for k in range(count):
        
            if k >= te and k < 2*te:
                write_test2.write(buf_img[k])
                write_test2.write(" ")
                write_test2.write(buf_lab[k])
                write_test2.write("\n")
                write_test2.flush()
            else:
                write_train2.write(buf_img[k])
                write_train2.write(" ")
                write_train2.write(buf_lab[k])
                write_train2.write("\n")
                write_train2.flush()

        for k in range(count):
        
            if k >= 2*te and k < 3*te:
                write_test3.write(buf_img[k])
                write_test3.write(" ")
                write_test3.write(buf_lab[k])
                write_test3.write("\n")
                write_test3.flush()
            else:
                write_train3.write(buf_img[k])
                write_train3.write(" ")
                write_train3.write(buf_lab[k])
                write_train3.write("\n")
                write_train3.flush()

        for k in range(count):
        
            if k >= 3*te and k < 4*te:
                write_test4.write(buf_img[k])
                write_test4.write(" ")
                write_test4.write(buf_lab[k])
                write_test4.write("\n")
                write_test4.flush()
            else:
                write_train4.write(buf_img[k])
                write_train4.write(" ")
                write_train4.write(buf_lab[k])
                write_train4.write("\n")
                write_train4.flush()

        for k in range(count):
        
            if k >= 4*te and k < 5*te:
                write_test5.write(buf_img[k])
                write_test5.write(" ")
                write_test5.write(buf_lab[k])
                write_test5.write("\n")
                write_test5.flush()
            else:
                write_train5.write(buf_img[k])
                write_train5.write(" ")
                write_train5.write(buf_lab[k])
                write_train5.write("\n")
                write_train5.flush()

        print("Complete the {} years...".format(age))


    #train, test = train_test_split(data, shuffle=False, test_size= 0.2, random_state= 123)
    #print(len(train))
    #print(len(test))
    
    #train_img, train_lab = zip(*train)
    #test_img, test_lab = zip(*test)

    #for i in range(len(train)):
    #    write_train.write((train_img[i]))
    #    write_train.write(" ")
    #    write_train.write((train_lab[i]))
    #    write_train.write("\n")
    #    write_train.flush()

    #for j in range(len(test)):
    #    write_test.write((test_img[j]))
    #    write_test.write(" ")
    #    write_test.write((test_lab[j]))
    #    write_test.write("\n")
    #    write_test.flush()

    

if __name__ == "__main__":
    app.run(main)