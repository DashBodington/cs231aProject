#import tensorflow as tf
from util import *

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos'
dataFolder = '/home/dash/Documents/yelpData/data'
imageLabels = imageFolder + '/photo_id_to_business_id.json'
businessLabels = dataFolder + '/yelp_academic_dataset_business.json'

print 'Loading Image Data...',
imgData = getImageData(imageLabels, 1)
numImages = len(imgData)
print 'Finished'

print 'Loading Business Data...',
busData = getBusinessData(businessLabels)
print 'Finished'

allData = {}
for img in imgData:
	if not img['business_id'] in allData:
		allData[img['business_id']] = []
	allData[img['business_id']].append(img['photo_id'])

allLabels = {}
for bus in allData.keys():
	for datum in busData:
		#print datum
		if(bus == datum['business_id']):
			if 'Price Range' in datum['attributes'].keys():
				allLabels[bus] = datum['attributes']['Price Range']
				break



print len(allData)
print len(allLabels)
print len(imgData)
print imgData[0]
