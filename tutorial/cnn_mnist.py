"""Example of CNN with MNIST dataset.
from tensorflow tutorial : https://www.tensorflow.org/tutorials/layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tutorial import util

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer (28x28x1)
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Conv Layer 1 out:(28x28x32)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer 1 out:(14x14x32)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Conv Layer 2 & Pooling 2
    # out: (14x14x64)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # out: (7x7x64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        tensors_to_log = {"loss": loss}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=20)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook])

    # for mode: EVAL
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main():
    """Program entrance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True)

    args = parser.parse_args()
    hparams = hparam.HParams(**args.__dict__)
    # run_config = tf.estimator.RunConfig()
    # run_config = run_config.replace(model_dir=hparams.job_dir)
    tf.logging.info("MODEL_DIR: %s", hparams.job_dir)
    util.recreate_folder(hparams.job_dir)

    tf.logging.info("loading mnist dataset")
    print("loading mnist dataset")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    tf.logging.info("mnist dataset loaded")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    tf.logging.info("train / eval data prepared")
    # Create the estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=hparams.job_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    tf.logging.info("model graph setup")
    mnist_classifier.train(input_fn=train_input_fn, steps=50)

    # Evaluate
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.logging.info("starting")
    tf.app.run()
