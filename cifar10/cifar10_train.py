# -*- coding: utf-8 -*-
"""train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import argparse
import tensorflow as tf

from cifar10 import cifar10
from cifar10 import util
from tensorflow.contrib.training.python.training import hparam

FLAGS = util.FLAGS

def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
        
        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels)
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()
            
            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)
            
            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    tf.logging.info(format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))
        
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True)
    args = parser.parse_args()
    hparams = hparam.HParams(**args.__dict__)
    tf.logging.info("MODEL_DIR: %s", hparams.job_dir)
    FLAGS.train_dir = hparams.job_dir

    util.may_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.logging.info('model dir: %s', FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()