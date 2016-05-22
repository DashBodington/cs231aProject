import tensorflow as tf
import numpy as np


def perceptron(trainData, testData):

	def getBatch(data, batchSize):
		inds = range(len(data[0]))
		inds = np.random.choice(inds, batchSize, replace=False)
		imOut = []
		labelOut = []
		for ind in inds:
			imOut.append(data[0][ind])
			labelOut.append(data[1][ind])
		return imOut, labelOut

	#Machine learning
	imSize = 64
	colors = 3
	x = tf.placeholder(tf.float32, [None, imSize*imSize*colors])
	W = tf.Variable(tf.zeros([imSize*imSize*colors, 1]))
	b = tf.Variable(tf.zeros([1]))
	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None])
	loss = tf.reduce_mean(tf.squared_difference(y,y_))

	train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	correct_prediction = tf.less(tf.abs(y-y_), 0.5)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in range(10000):
		batch_xs, batch_ys = getBatch(trainData,1000)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


	print "Accuracy: ", (sess.run(accuracy, feed_dict={x: testData[0], y_: testData[1]}))