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
def test_write_to_tfrecords(filename, data_dir, data_count):
    # 输出TFRecord文件的地址

    writer = tf.python_io.TFRecordWriter(filename)

    files = os.listdir(data_dir)
    num = 0
    random.shuffle(files)
    acount = 0
    bcount = 0
    ccount = 0
    count = 0
    all_count=data_count * 3

    for file in files:


        files2 = os.listdir(data_dir + '/' + file)
        if files2[0][2] == 'I':
            I = np.loadtxt(data_dir + '/' + file + '/' + files2[0])
            Q = np.loadtxt(data_dir + '/' + file + '/' + files2[1])

        else:
            I = np.loadtxt(data_dir + '/' + file + '/' + files2[1])
            Q = np.loadtxt(data_dir + '/' + file + '/' + files2[0])

        data = np.ndarray((8, 550, 2), dtype=np.float64)
        if len(I) != 4400:
            print 'error'
            continue
        if len(Q) != 4400:
            print 'error'
            continue
        I = I.reshape(8, 550)
        Q = Q.reshape(8, 550)

        for i in range(0, 8):
            for j in range(0, 550):
                data[i][j][0] = I[i][j]
                data[i][j][1] = Q[i][j]
        data_raw = data.tostring()
        if files2[0][0] == 'A':
            index = int(0)
            acount += 1
            if acount > data_count:
                continue
        if files2[0][0] == 'B':
            index = int(1)
            bcount += 1
            if bcount > data_count:
                continue
        if files2[0][0] == 'C':
            index = int(2)
            ccount += 1
            if ccount > data_count:
                continue

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(index),
            'data_raw': _bytes_feature(data_raw)
        }))
        print('num:' + str(count)+'_'+str(index))
        writer.write(example.SerializeToString())
        count += 1
        if count == all_count:
            break

    writer.close()
    print('success：' + str(count))


if __name__ == '__main__':
    tfrecords_filename = "abc_mic_val_5.tfrecords"
    data_dir = '/home/dmrf/文档/Gesture/New_Data/Gesture_Data/abc_mic_test_5'
    data_count = 480#2080#480
    test_write_to_tfrecords(tfrecords_filename, data_dir, data_count)
