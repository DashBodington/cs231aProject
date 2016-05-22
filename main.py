#import tensorflow as tf
from util import *
from learning import *

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = False

random.seed(0)

print "Loading data..."
if createNewDataset:
	allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder)
else:
	(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )

print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

trainData, testData = createDatasets(allImages, allLabels, 0.8)

#Learning methods
#Perceptron
#perceptron(trainData,testData)

lenet(trainData, testData)

