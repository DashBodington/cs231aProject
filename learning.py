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
	imSize = 128
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

def perceptronOneHot(trainData, testData):
	sess = tf.InteractiveSession()#

	newLabels = []
	#Make dataset one-hot
	for label in trainData[1]:
		if(label == 0):
			newLabels.append(np.array([1, 0]))
		else:
			newLabels.append(np.array([0, 1]))

	trainData = (trainData[0], newLabels)
	newLabels = []


	for label in testData[1]:
		if(label == 0):
			newLabels.append(np.array([1, 0]))
		else:
			newLabels.append(np.array([0, 1]))
			
	testData = (testData[0], newLabels)

	inLength = trainData[0][0].shape[0]

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
	imSize = 128
	colors = 3
	x = tf.placeholder(tf.float32, [None, inLength])
	y_ = tf.placeholder(tf.float32, [None,2])
	W = tf.Variable(tf.truncated_normal([inLength, 2], stddev=0.1))
	b = tf.Variable(tf.truncated_normal([2]))

	sess.run(tf.initialize_all_variables())

	y = tf.nn.softmax(tf.matmul(x, W) + b)
	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	#Cross entropy which only updates on incorrect predictions
	cross_entropy= tf.reduce_mean(tf.mul(-tf.reduce_sum(y_*tf.log(y), reduction_indices=1),(1.5-tf.to_float(correct_prediction)))) # Cross entropy
	
	#Normal cross entropy
	#cross_entropy= tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=1)) # Cross entropy

	#optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(cost) # Gradient Descent

	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) 

	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	train_step = tf.train.MomentumOptimizer(1e-4,0.8).minimize(cross_entropy)
	init = tf.initialize_all_variables()
	#sess = tf.Session()
	#sess.run(init)
	with tf.Session() as sess:
		sess.run(init)

		for i in range(100000):
			batch_xs, batch_ys = getBatch(trainData,200)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			if i % 500 == 0:
				print i, "Train Accuracy: ", (sess.run(accuracy, feed_dict={x: trainData[0], y_: trainData[1]}))
				print "Test Accuracy: ", (sess.run(accuracy, feed_dict={x: testData[0], y_: testData[1]}))

		print "Train Accuracy: ", (sess.run(accuracy, feed_dict={x: trainData[0], y_: trainData[1]}))
		print "Test Accuracy: ", (sess.run(accuracy, feed_dict={x: testData[0], y_: testData[1]}))




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
	imSize = 128
	colors = 3

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

	W_conv1 = weight_variable([15, 15, 3, 16])
	b_conv1 = bias_variable([16])

	W_conv2 = weight_variable([15, 15, 16, 32])
	b_conv2 = bias_variable([32])

	W_fc1 = weight_variable([16 * 16 * 32, 1024])
	b_fc1 = bias_variable([1024])

	#Network
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 1])
	b_fc2 = bias_variable([1])

	y=tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	loss = tf.reduce_mean(tf.squared_difference(y,y_))

	train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
	sess.run(tf.initialize_all_variables())

	correct_prediction = tf.less(tf.abs(y-y_), 0.5)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Train model
	for i in range(10000):
		batch_xs, batch_ys = getBatch(trainData,100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})



	#Too much data to process training set at once
	trainAcc = []
	for i in range(1,len(trainData[1])-2,2):
		trainAcc.append((sess.run(accuracy, feed_dict={x: trainData[0][i:i+1], y_: trainData[1][i:i+1], keep_prob: 1.0})))

	#print testAcc
	print "Train Accuracy: ", np.mean(trainAcc)
	#Too much data to process test set at once
	testAcc = []
	for i in range(1,len(testData[1])-2,2):
		testAcc.append((sess.run(accuracy, feed_dict={x: testData[0][i:i+1], y_: testData[1][i:i+1], keep_prob: 1.0})))

	#print testAcc
	print "Test Accuracy: ", np.mean(testAcc)

def preprocessedLenet(trainData, testData):
	#Mean subtract and normalize deviation
	for im in trainData[0]:
		im = (im-np.mean(im))/np.std(im)
	for im in testData[0]:
		im = (im-np.mean(im))/np.std(im)

	lenet(trainData, testData)

def lenetOneHot(trainData, testData):


	newLabels = []
	#Make dataset one-hot
	for label in trainData[1]:
		if(label == 0):
			newLabels.append(np.array([1, 0]))
		else:
			newLabels.append(np.array([0, 1]))

	trainData = (trainData[0], newLabels)
	newLabels = []


	for label in testData[1]:
		if(label == 0):
			newLabels.append(np.array([1, 0]))
		else:
			newLabels.append(np.array([0, 1]))
			
	testData = (testData[0], newLabels)

	#print trainData[1]

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
	y_ = tf.placeholder(tf.float32, shape=[None,2])
	x_image = tf.reshape(x, [-1,imSize,imSize,colors])

	W_conv1 = weight_variable([7, 7, 3, 8])
	b_conv1 = bias_variable([8])

	W_conv2 = weight_variable([7, 7, 8, 16])
	b_conv2 = bias_variable([16])

	W_fc1 = weight_variable([imSize/4 * imSize/4 * 16, 1024])
	b_fc1 = bias_variable([1024])

	#Network
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, imSize/4*imSize/4*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 2])
	b_fc2 = bias_variable([2])

	y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	loss = tf.reduce_mean(tf.squared_difference(y,y_))

	train_step = tf.train.MomentumOptimizer(1e-2,0.8).minimize(cross_entropy)
	sess.run(tf.initialize_all_variables())

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Train model
	for i in range(1000):
		batch_xs, batch_ys = getBatch(trainData,200)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
		if(i%100 == 0):
			#Too much data to process training set at once
			trainAcc = []
			for i in range(1,len(trainData[1])-2,2):
				trainAcc.append((sess.run(accuracy, feed_dict={x: trainData[0][i:i+1], y_: trainData[1][i:i+1], keep_prob: 1.0})))

			#print testAcc
			print "Train Accuracy: ", np.mean(trainAcc)
			#Too much data to process test set at once
			testAcc = []
			for i in range(1,len(testData[1])-2,2):
				testAcc.append((sess.run(accuracy, feed_dict={x: testData[0][i:i+1], y_: testData[1][i:i+1], keep_prob: 1.0})))

			#print testAcc
			print "Test Accuracy: ", np.mean(testAcc)


	#Too much data to process training set at once
	trainAcc = []
	for i in range(1,len(trainData[1])-2,2):
		trainAcc.append((sess.run(accuracy, feed_dict={x: trainData[0][i:i+1], y_: trainData[1][i:i+1], keep_prob: 1.0})))

	#print testAcc
	print "Train Accuracy: ", np.mean(trainAcc)
	#Too much data to process test set at once
	testAcc = []
	for i in range(1,len(testData[1])-2,2):
		testAcc.append((sess.run(accuracy, feed_dict={x: testData[0][i:i+1], y_: testData[1][i:i+1], keep_prob: 1.0})))

	#print testAcc
	print "Test Accuracy: ", np.mean(testAcc)

	testPred = []
	for i in range(1,len(testData[1])):
		testPred.append((sess.run(tf.argmax(y,1), feed_dict={x: np.expand_dims(testData[0][i],axis=0), y_: np.expand_dims(testData[1][i],axis=0), keep_prob: 1.0})))
