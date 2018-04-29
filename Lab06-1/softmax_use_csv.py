import tensorflow as tf
import numpy as np

xy = np.loadtxt('dataset.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-3]
y_data = xy[:, -3:]

print(y_data)
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
	if step % 200 == 0:
		print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.arg_max(a, 1)))
