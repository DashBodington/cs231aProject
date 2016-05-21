#import tensorflow as tf
from util import *
import numpy as np
import cv2
import math
import random

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = False

print "Loading data...",
if createNewDataset:
	allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder)
else:
	(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )

print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

#Bin data by price rating
byPrice = {}
for bus in allImages.keys():
	if not allLabels[bus] in byPrice.keys():
		byPrice[allLabels[bus]] = []
	byPrice[allLabels[bus]].append(bus)

#Shuffle businesses within their bins
for cost in byPrice.values():
	random.shuffle(cost)

#Split businesses into train and test
trainRatio = 0.9
trainData = []
testData = []
for cost in byPrice.values():
	for i in xrange(len(cost)):
		if(i < trainRatio*len(cost)):
			#Training set
			trainData.append(bus)
		else:
			#Test set
			testData.append(bus)

