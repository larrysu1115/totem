import tensorflow as tf
import numpy as np
from model.rnn.Rnn00 import Model

def main():

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    print('generating training data')
    x, y = Model.data_generator()
    print('generated training data %d' % len(y))
    NUM_TRAIN = 20000
    LOGDIR = '/tmp/rnn1'


    if tf.gfile.Exists(LOGDIR):
        tf.gfile.DeleteRecursively(LOGDIR)
    tf.gfile.MakeDirs(LOGDIR)
    print('clear tensorBoard log dir: ', LOGDIR)

    x_dev = x[NUM_TRAIN:]
    y_dev = y[NUM_TRAIN:]
    x_train = x[:NUM_TRAIN]
    y_train = y[:NUM_TRAIN]

    m = Model()
    print('model initialization')
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)

    sess.run(tf.global_variables_initializer())

    batch_size = 512
    no_of_batches = int(len(x_train)/batch_size)
    epoch = 50

    for i in range(epoch):
        ptr = 0
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        for _ in range(no_of_batches):
            inp, out = x_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
            ptr+=batch_size
            summary, _, error_train = sess.run([m.summary, m.optimize, m.error],{m.x: inp, m.y: out})

        error_dev = sess.run(m.error,{m.x: x_dev, m.y: y_dev})
        train_writer.add_summary(summary, i)
        print("Epoch %d : train error=%.6f   -  dev error=%.6f" % (i, error_train, error_dev))

    # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    pred_val = sess.run(m.prediction,{m.x: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
    print(pred_val)
    for i in range(0,len(pred_val[0])):
        print('count %2d  prob: %.6f' % (i, pred_val[0][i]))

    sess.close()
    train_writer.close()



if __name__ == '__main__':
    main()