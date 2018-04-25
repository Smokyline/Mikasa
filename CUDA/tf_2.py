import tensorflow as tf

default_graph = tf.get_default_graph()

c1 = tf.constant(1.0)

second_graph = tf.Graph()
with second_graph.as_default():
    c2 = tf.constant(10.0)

with tf.Session(graph= second_graph) as session:
    print(c2.eval())