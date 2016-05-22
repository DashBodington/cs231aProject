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

def lenet(trainData, testData):
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
	imSize = trainData[0].shape[0]
	colors = trainData[0].shape[2]

	sess = tf.InteractiveSession()

	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	#Variables
	x = tf.placeholder(tf.float32, shape=[None, imSize*imSize*colors])
	y_ = tf.placeholder(tf.float32, shape=[None])
	x_image = tf.reshape(x, [-1,imSize,imSize,colors])

	W_conv1 = weight_variable([15, 15, 3, 8])
	b_conv1 = bias_variable([8])

	W_conv2 = weight_variable([15, 15, 8, 16])
	b_conv2 = bias_variable([16])

	W_fc1 = weight_variable([16 * 16 * 16, 1024])
	b_fc1 = bias_variable([1024])

	#Network
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 1])
	b_fc2 = bias_variable([1])

	y=tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	loss = tf.reduce_mean(tf.squared_difference(y,y_))

	train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
	sess.run(tf.initialize_all_variables())

	correct_prediction = tf.less(tf.abs(y-y_), 0.5)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Train model
	for i in range(10000):
		batch_xs, batch_ys = getBatch(trainData,100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})


	#Too much data to process training set at once
	testAcc = []
	for i in range(1,len(testData[1])-2,2):
		testAcc.append((sess.run(accuracy, feed_dict={x: testData[0][i:i+1], y_: testData[1][i:i+1], keep_prob: 1.0})))

	#print testAcc
	print "Accuracy: ", np.mean(testAcc)