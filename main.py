#import tensorflow as tf
from util import *
import numpy as np
import cv2
import math

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = False

print "Loading data...",
if createNewDataset:
	allImages, allLabels = loadData(imageLabels, businessLabels)
else:
	(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )

print "Finished"

print len(allImages)
print len(allLabels)


counts = [0, 0, 0, 0, 0]
for bus in allLabels.keys():
	counts[allLabels[bus]-1] += 1

print counts