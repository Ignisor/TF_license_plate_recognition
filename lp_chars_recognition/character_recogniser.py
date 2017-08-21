import tensorflow as tf

from base import NeuralModelBase
from .dataset import CharsDataset

class CharRecogniser(NeuralModelBase):
    INPUT_SIZE = [-1, 24, 32, 1]
    ANSWERS_SIZE = 36
    CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    dataset_class = CharsDataset

    def model(self):
        # initialise model
        self.x = tf.placeholder(tf.float32, shape=[None, self.INPUT_SIZE[1] * self.INPUT_SIZE[2], self.INPUT_SIZE[3]])
        x_image = tf.reshape(self.x, self.INPUT_SIZE)

        # First layer:
        W_conv1 = self.weight_variable([5, 5, 1, 32])
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

        self.softmax = tf.nn.softmax(logits=self.y_conv)

        return self.y_conv