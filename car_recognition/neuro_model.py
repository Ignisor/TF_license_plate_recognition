from urllib.request import urlopen
from io import BytesIO
import os
import logging

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from numpy import array
from PIL import Image


BASE_DIR = os.path.dirname(__file__)


class CarRecogniser(object):
    """Neuro model used to recognise is it car on image. It works with 64x64 RGB images."""

    def __init__(self):
        # initialise model
        self.x = tf.placeholder(tf.float32, shape=[None, 64 * 64, 3])
        x_image = tf.reshape(self.x, [-1, 64, 64, 3])

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
        W_fc1 = self.weight_variable([16 * 16 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout:
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout layer:
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])

        self.y_conv = tf.matmul(self.h_fc1_drop, W_fc2) + b_fc2

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.sess, os.path.join(BASE_DIR, "data/saved/model.ckpt"))
        except NotFoundError:
            logging.warning("No file to load model. Place model to 'data/saved/model.ckpt'")
            tf.global_variables_initializer().run()

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

    def is_car(self, image):
        """
        Checks is it image with car on it
        :param image: PIL Image object
        :return: bool
        """
        if type(image) != Image.Image:
            raise TypeError('image must be PIL Image object')
        if image.size != (64, 64):
            raise ValueError('image size must be 64x64 pixels')

        vector = []
        for pixel in image.getdata():
            vector.append(tuple((color/255 for color in pixel)))

        vector = array(vector).reshape(1, 64 * 64, 3)

        softmax = tf.nn.softmax(logits=self.y_conv)

        result = self.sess.run(softmax, feed_dict={self.x: vector, self.keep_prob: 1.0})[0]

        return result[0] > result[1]

    def is_car_from_url(self, image_url):
        """Check is car on image loaded from url. Used for testing purposes."""
        img_file = BytesIO(urlopen(image_url).read())

        image = Image.open(img_file)
        image = image.resize((64, 64))
        image = image.convert('RGB')

        if self.is_car(image):
            print('It\'s NOT a car')
        else:
            print('It\'s a car')
