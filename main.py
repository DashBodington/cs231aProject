#import tensorflow as tf
from util import *
import numpy as np
import tensorflow as tf

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = False

print "Loading data..."
if createNewDataset:
	allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder)
else:
	(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )

print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

trainData, testData = createDatasets(allImages, allLabels, 0.9)

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

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
	batch_xs, batch_ys = getBatch(trainData,1000)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.squared_difference(y,y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testData[0], y_: testData[1]}))