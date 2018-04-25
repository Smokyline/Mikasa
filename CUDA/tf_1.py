import tensorflow as tf # в дальнейшем эта строка будет опускаться
import numpy as np
import time

# Creates a graph.
tf.device('/gpu:0')

size=2000
x = tf.placeholder(tf.float32, shape=(size, size))
y = tf.matmul(x, x)
z = tf.matmul(y, x)

with tf.Session() as sess:
  rand_array = np.random.rand(size, size)

  start_time = time.time()
  for _ in range(10):
      np.dot(np.dot(rand_array,rand_array), rand_array)
  print("--- %s seconds numpy multiply ---" % (time.time() - start_time))

  start_time = time.time()
  for _ in range(10):
      sess.run(z, feed_dict={x: rand_array})
  print("--- %s seconds tensorflow---" % (time.time() - start_time))

