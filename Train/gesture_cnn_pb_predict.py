#-*-coding:utf-8-*-



from __future__ import absolute_import, unicode_literals
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from Utils.ReadAndDecode import read_and_decode

val_path = '/home/dmrf/tensorflow_gesture_data/Gesture_data/mic_train_5ms.tfrecords'
x_val, y_val = read_and_decode(val_path)
test_batch=64
min_after_dequeue_test = test_batch * 2

num_threads = 3
test_capacity = min_after_dequeue_test + num_threads * test_batch
# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

labels_type=3
test_count = labels_type * 100
Test_iterations = test_count / test_batch


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = '../model/gesture_cnn.pb'

    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        for step in range(Test_iterations + 1):
            test_x, test_y = sess.run([test_x_batch, test_y_batch])
            b = sess.run(output, feed_dict={input: test_x, output: test_y})
            print('Test Accuracy', step,
                  b)



