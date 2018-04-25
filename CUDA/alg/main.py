import tensorflow as tf
import numpy as np
import os

from CUDA.alg.tools import *
from CUDA.alg.read_data import *
from CUDA.alg.draw import visual_circle

data_x, data_y = get_data_XY(norm=True)

samples = len(data_x)
packetSize = len(data_y)
shape_y = data_y.shape

tf_data_x = tf.placeholder(tf.float32, shape=shape_y)
tf_data_y = tf.placeholder(tf.float32, shape=shape_y)

weight = tf.Variable(initial_value=np.ones((1, 19))/10, dtype=tf.float32, name="a")
bias = tf.Variable(initial_value=[[0.0] for q in range(shape_y[0])], dtype=tf.float32, name="b")
model = tf.add(tf.multiply(tf_data_x, weight), bias)

loss = tf.reduce_mean(tf.square(model-tf_data_y)) # функция потерь, о ней ниже
optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss) # метод оптимизации, о нём тоже ниже


with tf.Session() as session:
    tf.global_variables_initializer().run()
    for i in range(5000):
        #feed_dict={tf_data_x: data_x[i*packetSize:(i+1)*packetSize], tf_data_y: data_y}
        feed_dict={tf_data_x: data_y, tf_data_y: data_y}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict) # запускаем оптимизатор и вычисляем "потери"
        if i%1000==0:
            print("ошибка: %f" % (l,))

    model_y = np.array(tf.add(tf.multiply(data_y, weight.eval()), bias.eval()).eval())
    w, b = weight.eval(), bias.eval()
    print("ошибка: %f" % (l,))
    #print(w, b)

    total_loss_x = np.empty((0, 16))
    for j in range(samples//packetSize):
        model_x = np.multiply(data_x[j*packetSize:(j+1)*packetSize], w) + b
        loss_x = np.mean(np.sqrt(np.power(model_y-model_x, 2)), axis=0)
        total_loss_x = np.append(total_loss_x, loss_x)

    asort = np.argsort(total_loss_x[:samples])
    idx_best = asort[-135:]

eqs = get_eq()
coord = get_coordX()
print(len(idx_best))
acc = acc_check(coord[idx_best], np.append(eqs[0],eqs[1], axis=0), r=0.225)
print('acc=%f' % acc)

visual_circle(X=coord,B=coord[idx_best],eqs=eqs,title='tf')