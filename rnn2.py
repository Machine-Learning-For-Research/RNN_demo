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

        # 取出输入数据流(batch_size, 28 x 28 x 1)
        x = self.x
        # 将数据reshape为(batch_size x 28, 28)格式
        x = tf.reshape(x, [-1, n_inputs])

        # 通过隐藏层, shape变为(batch_size x 28, 50)格式
        W = weight_variables([n_inputs, n_hidden])
        b = biases_variables([n_hidden])
        x = tf.matmul(x, W) + b

        # 将数据reshape为(batch_size, 28, 50)格式
        x = tf.reshape(x, [-1, n_steps, n_hidden])

        # rnn的入参为n_hidden
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        initializer = cell.zero_state(self.batch_size, tf.float32)
        # dynamic_rnn之后, output的shape是(batch_size, 28, 50)
        outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=initializer)

        # 将outputs维度转置为(28, batch_size, 50)
        outputs = tf.transpose(outputs, [1, 0, 2])
        # 将outputs数据展开为 28 x (batch_size, 50)
        outputs = tf.unstack(outputs)
        # 因为预测值是labels, 所以只取最后一项(batch_size, 50)作为输出
        x = outputs[-1]

        # 最后将RNN的输出进行(50 x 10)的fc, 变为(batch_size, 10), 即最终结果
        W = weight_variables([n_hidden, 10])
        b = biases_variables([10])
        x = tf.matmul(x, W) + b

        return x


if __name__ == '__main__':
    RNN().train()
