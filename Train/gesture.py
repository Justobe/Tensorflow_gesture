# -*- coding: utf-8 -*-

import tensorflow as tf

from Utils.ReadAndDecode import read_and_decode

from Net.CNN_Init import weight_variable, bias_variable, conv2d, max_pool_2x2

train_path = '/home/tensorflow_gesture_data/abc_mic_train_5.tfrecords'
val_path = '/home/tensorflow_gesture_data/abc_mic_val_5.tfrecords'
x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

w = 550
h = 8
c = 2
labels_count = 3
Learning_rate = 0.01

# 占位符

# [batch, in_height, in_width, in_channels]
x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, ], name='y_')

# [filter_height, filter_width, in_channels, out_channels]
w_conv1 = weight_variable([1, 7, 2, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x, w_conv1, [1, 1, 3,
                                         1]) + b_conv1)  # stride/kernel:The stride of the sliding window for each  dimension of `input`.

h_pool1 = max_pool_2x2(h_conv1, [1, 1, 2, 1],
                       [1, 1, 2, 1])  # stride/kernel:The size of the window for each dimension of the input tensor.

w_conv2 = weight_variable([1, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, s=[1, 2, 1, 1]) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2, k=[1, 2, 1, 1], s=[1, 2, 1, 1])

w_conv3 = weight_variable([1, 4, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 2, 1, 1]) + b_conv3)
h_pool3 = max_pool_2x2(h_conv2, [1, 2, 1, 1], [1, 2, 1, 1])

w_fc1 = weight_variable([8 * 46 * 64, 256])
b_fc1 = bias_variable([256])
h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 46 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

w_fc2 = weight_variable([256, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

# Loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction
correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 组合batch
train_batch = 64
test_batch = 32

min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

Training_iterations = 10000
Validation_size = 100

test_count = labels_count * 100
Test_iterations = test_count / test_batch

display_step = 100

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(Training_iterations + 1):
        batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=train_batch,num_threads=num_threads,
                                             capacity=train_capacity, min_after_dequeue=min_after_dequeue_train)
        batch_train = sess.run(batch_train)

        sess.run(train, feed_dict={x: batch_train[0], y_: batch_train[1]})
        # Train accuracy
        if step % Validation_size == 0:
            print('Training Accuracy', step,
                  sess.run(accuracy, feed_dict={x: batch_train[0], y_: batch_train[1]}))

    for step in range(Test_iterations + 1):
        batch_x_test, batch_y_test = tf.train.shuffle_batch([x_train, y_train], batch_size=test_batch,
                                                            num_threads=num_threads,
                                                            capacity=test_capacity,
                                                            min_after_dequeue=min_after_dequeue_test)

        print('Test Accuracy', step,
              sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test}))
