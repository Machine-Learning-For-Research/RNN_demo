# coding=utf-8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def biases_variables(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


class RNN:
    def __init__(self, leaning_rate=1e-2, max_iterators=200, batch_size=200):
        self.learning_rate = leaning_rate
        self.max_iterators = max_iterators
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])

    def train_step(self, logits, labels):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return optimizer.minimize(loss)

    def train(self):
        prediction = self.inference()
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        train_op = self.train_step(prediction, self.y)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        test_images, test_labels = mnist.test.images[:self.batch_size], mnist.test.labels[:self.batch_size]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, self.max_iterators + 1):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                sess.run(train_op, feed_dict={self.x: batch_x, self.y: batch_y})
                if step % 10 == 0 or step == self.max_iterators:
                    acc = sess.run(accuracy, feed_dict={self.x: test_images, self.y: test_labels})
                    print('Step %s, Accuracy %s' % (step, acc))

    def inference(self):

        n_steps = 28
        n_inputs = 28
        n_hidden = 50

        x = self.x

        x = tf.reshape(x, [-1, n_inputs])

        W = weight_variables([n_steps, n_hidden])
        b = biases_variables([n_hidden])
        x = tf.matmul(x, W) + b

        x = tf.reshape(x, [-1, n_steps, n_hidden])

        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.unstack(outputs)
        x = outputs[-1]

        W = weight_variables([n_hidden, 10])
        b = biases_variables([10])
        x = tf.matmul(x, W) + b

        return x


if __name__ == '__main__':
    RNN().train()
