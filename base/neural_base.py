import inspect
import os
import logging
import time
from abc import ABCMeta

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError


class NeuralModelBase(metaclass=ABCMeta):
    """Neural model base"""
    INPUT_SIZE = [-1, 64, 64, 3]
    ANSWERS_SIZE = 2
    dataset_class = None

    def __init__(self):
        super(NeuralModelBase, self).__init__()

        self.BASE_DIR = os.path.dirname(inspect.getfile(self.__class__))

        self.model()

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.sess, os.path.join(self.BASE_DIR, "data/saved/model.ckpt"))
        except NotFoundError:
            logging.warning("No file to load model. Place model to 'data/saved/model.ckpt'")
            tf.global_variables_initializer().run()

    def model(self):
        # initialise model
        self.x = tf.placeholder(tf.float32, shape=[None, self.INPUT_SIZE[1] * self.INPUT_SIZE[2], self.INPUT_SIZE[3]])
        x_image = tf.reshape(self.x, self.INPUT_SIZE)

        # First layer:
        W_conv1 = self.weight_variable([5, 5, 3, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Second layer:
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Densely connected layer:
        W_fc1 = self.weight_variable([(self.INPUT_SIZE[1] // 4) * (self.INPUT_SIZE[2] // 4) * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, (self.INPUT_SIZE[1] // 4) * (self.INPUT_SIZE[2] // 4) * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout:
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout layer:
        W_fc2 = self.weight_variable([1024, self.ANSWERS_SIZE])
        b_fc2 = self.bias_variable([self.ANSWERS_SIZE])

        self.y_conv = tf.matmul(self.h_fc1_drop, W_fc2) + b_fc2

        # correct values
        self.y_ = tf.placeholder(tf.float32, [None, self.ANSWERS_SIZE])

        return self.y_conv

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def train(self, steps=1000, dataset_class=None):
        """
        Train model
        :param dataset_class: subclass of DataSetBase class to get data from 
        :param steps: amount of training steps 
        """
        dataset_class = dataset_class or self.dataset_class
        dataset_class = dataset_class()

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())

        for i in range(steps):
            logging.info(f"Started {i+1} training")

            t = time.time()
            batch = dataset_class.get_batch(amount=100)
            logging.debug(f"data got in: {time.time() - t:.2f}s")

            t = time.time()
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            del batch
            logging.debug(f"train step took: {time.time() - t:.2f}s")

            if (i + 1) % 100 == 0:
                test_batch = dataset_class.get_batch(test=True)
                acc = accuracy.eval(feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
                del test_batch
                logging.info(f"----Step {i+1}, training accuracy {acc}----")
                self.saver.save(self.sess, "data/saved/model.ckpt")

        test_batch = dataset_class.get_batch(test=True)
        test_acc = accuracy.eval(feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
        logging.info(f"Test accuracy {test_acc}")

        self.saver.save(self.sess, "data/saved/model.ckpt")

    def process_data(self, data):
        """
        prepares data for neural network
        :param data: data to process
        :return: processed data 
        """
        return data

    def process_result(self, result):
        """
        processes neural network result so we can return more convenient info
        :param result: result to process
        :return: processed data
        """
        return result

    def run(self, data):
        """
        Runs neural network for given data and returns processed answer
        :param data: data for neural network
        :return: processed neural network result
        """
        data = self.process_data(data)

        softmax = tf.nn.softmax(logits=self.y_conv)

        result = self.sess.run(softmax, feed_dict={self.x: data, self.keep_prob: 1.0})

        return self.process_result(result)


