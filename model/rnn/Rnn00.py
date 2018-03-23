import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import numpy as np
from random import shuffle
from model.tool.Decorators import define_scope
from model.tool.TensorVisual import var_summaries

class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 20, 1), name='x-input')
        self.y = tf.placeholder(tf.float32, shape=(None, 21), name='y-input')
        self.prediction
        self.optimize
        self.error
        self.summary = tf.summary.merge_all()

    @define_scope
    def prediction(self):
        num_hidden = 24

        with tf.name_scope('rnn'):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
            val, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.x, dtype=tf.float32)
            val = tf.transpose(val, [1,0,2])
            with tf.name_scope('lstm-last'):
                last = tf.gather(val, int(val.get_shape()[0]) - 1)
                var_summaries(val)
                var_summaries(last)

        with tf.name_scope('fc'):
            with tf.name_scope('w'):
                weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.y.get_shape()[1])]), name='weight')
                var_summaries(weight)
    
            with tf.name_scope('b'):
                bias = tf.Variable(tf.constant(.1, shape=[self.y.get_shape()[1]]), name='bias')
                var_summaries(bias)
        
        return tf.nn.softmax(tf.matmul(last, weight) + bias)
    
    
    @define_scope
    def optimize(self):
        with tf.name_scope('cross_entropy'):    
            cross_entropy = -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)))
            tf.summary.scalar('cross_entropy', cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
        return optimizer.minimize(cross_entropy)


    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        tf.summary.scalar('error', error)
        return error


    @staticmethod
    def data_generator(bits=14):
        data_input = ['{0:020b}'.format(i) for i in range(2**16)]
        shuffle(data_input)
        data_input = [map(int,i) for i in data_input]
        ti = []
        to = []
        for i in data_input:
            temp_o = np.zeros((21))
            temp_list = []
            for j in i:
                temp_list.append([j])
            temp_list = np.array(temp_list)
            temp_o[np.sum(temp_list)] = 1
            ti.append(temp_list)
            to.append(temp_o)
        return (ti, to)