#import tensorflow as tf
from util import *
from learning import *
from alexnet import *

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = False

#random.seed(0)

print "Loading data..."
if createNewDataset:
	allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=227)
	trainData, testData = createDatasets(allImages, allLabels, 0.7)
	pickle.dump((trainData, testData), open("dataset.p","wb"))
else:
	#(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )
	data = pickle.load( open( "dataset.p", "rb" ) )
	trainData = data[0]
	testData = data[1]
	print len(trainData[1]), "train images"
	print len(testData[1]), "test images"

#print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

#trainData, testData = createDatasets(allImages, allLabels, 0.7)

#Save to file
#pickle.dump((trainData, testData), open("dataset.p","wb"))



#Learning methods
#Perceptron
#perceptronOneHot(trainData,testData)

#lenetOneHot(trainData, testData)

print trainData[0].shape
print len(trainData[1])
vals = [0.0, 0.0, 0.0, 0.0]
for truth in trainData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))

print testData[0].shape#
print len(testData[1])
vals = [0.0, 0.0, 0.0, 0.0]
for truth in testData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))


train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


img = np.reshape(trainData[0][425],(227,227,3))

x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
i = x_dummy.copy()
i[0,:,:,:] = (img).astype(float32)

alexnet(i)


#print trainData[0][425].shape
#cv2.im#show('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()