# -*-coding:utf-8-*-
import random

import tensorflow as tf
import numpy as np
import os


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 制作TFRecord格式
def write_to_tfrecords(filename, data_dir):
    # 输出TFRecord文件的地址

    writer = tf.python_io.TFRecordWriter(filename)

    files = os.listdir(data_dir)
    random.shuffle(files)
    count = 0

    for file in files:
        files2 = os.listdir(data_dir + '/' + file)
        if files2[0][2] == 'I':
            I = np.loadtxt(data_dir + '/' + file + '/' + files2[0])
            Q = np.loadtxt(data_dir + '/' + file + '/' + files2[1])
        else:
            I = np.loadtxt(data_dir + '/' + file + '/' + files2[1])
            Q = np.loadtxt(data_dir + '/' + file + '/' + files2[0])

        I = I.reshape(-1, 1)
        Q = Q.reshape(-1, 1)

        num_flag = 0
        if len(I) > 8800:
            num_flag = 1

        data = np.zeros(shape=(8, 2200, 2), dtype=np.float64)

        try:
            I = I.reshape(8, -1)
            Q = Q.reshape(8, -1)
        except ValueError:
            continue


        if num_flag == 0:  # len==1100
            for i in range(0, 8):
                for j in range(0, 1100):
                    data[i][j][0] = I[i][j]
                    data[i][j][1] = Q[i][j]
                data[i][1100][0] = 2048
                data[i][1100][1] = 2048
        else:#len==2200
            for i in range(0, 8):
                for j in range(0, 2200):
                    data[i][j][0] = I[i][j]
                    data[i][j][1] = Q[i][j]

        data_raw = data.tostring()
        type = file[0]
        index = int(-1)
        if type == 'A':
            index = int(0)
        if type == 'B':
            index = int(1)
        if type == 'C':
            index = int(2)
        if type == 'F':
            index = int(3)
        if type == 'G':
            index = int(4)
        if type == 'H':
            index = int(5)
        if type == 'I':
            index = int(6)
        if type == 'J':
            index = int(7)

        if index == -1:
            print('label not found exception')
            continue

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(index),
            'data_raw': _bytes_feature(data_raw)
        }))
        print('num:' + str(count) + '_' + str(index))
        writer.write(example.SerializeToString())
        count += 1

    writer.close()
    print('success：' + str(count))


if __name__ == '__main__':
    tfrecords_filename = "../Data/train.tfrecords"
    data_dir = '/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/连续手势集/Train'
    write_to_tfrecords(tfrecords_filename, data_dir)
