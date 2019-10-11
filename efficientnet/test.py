
import tensorflow as tf
import numpy as np
import random

a = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')

start = tf.compat.v1.get_variable(shape=(), initializer=tf.constant_initializer(-1), name='range_start')
end = tf.compat.v1.get_variable(shape=(), initializer=tf.constant_initializer(1), name='range_end')


aq = tf.quantization.fake_quant_with_min_max_vars(
    a,
    start,
    end,
    num_bits=8
)

diff = tf.compat.v1.losses.mean_squared_error(a, aq)

optim = tf.train.AdamOptimizer(learning_rate=0.1).minimize(diff, var_list=[start, end])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(optim, feed_dict={
            "input:0": np.array([random.uniform(-3, 3)])
        })

    print(sess.run([start, end]))

