#import tensorflow as tf
from util import *
from learning import *
from alexnet import *
from sklearn import svm

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

createNewDataset = 3

#random.seed(0)

print "Loading data..."
if createNewDataset ==1:
	allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=227)
	trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=0.5)
	pickle.dump(testData, open("testdataset.p","wb"))
	pickle.dump(trainData, open("traindataset.p","wb"))
elif createNewDataset ==2:
	#(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )
	#data = pickle.load( open( "dataset.p", "rb" ) )
	trainData = pickle.load(open("traindataset.p","rb"))
	testData = pickle.load(open("testdataset.p","rb"))
	print len(trainData[1]), "train images"
	print len(testData[1]), "test images"


	train_x = zeros((1, 227,227,3)).astype(float32)
	train_y = zeros((1, 1000))
	xdim = train_x.shape[1:]
	ydim = train_y.shape[1]


	img = np.reshape(trainData[0][425],(227,227,3))

	x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
	im = x_dummy.copy()
	im[0,:,:,:] = (img).astype(float32)

	trainFeats = []
	testFeats = []

	trainFeats = alexnet(trainData[0],verbose=False)

	testFeats = alexnet(testData[0], verbose=False)

	print len(trainFeats)
	print len(testFeats)

	trainData = (trainFeats,trainData[1])
	testData = (testFeats,testData[1])

	pickle.dump(testData, open("testalexfeats.p","wb"))
	pickle.dump(trainData, open("trainalexfeats.p","wb"))
elif createNewDataset ==3:

	testData = pickle.load(open("testalexfeats.p","rb"))
	trainData = pickle.load(open("trainalexfeats.p","rb"))

	tol = 0.01
	for im in testData[0]:
		im.real[abs(im) < tol] = 0.0
	for im in trainData[0]:
		im.real[abs(im) < tol] = 0.0


#print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

#trainData, testData = createDatasets(allImages, allLabels, 0.7)

#Save to file
#pickle.dump((trainData, testData), open("dataset.p","wb"))



#Learning methods
#Perceptron
#perceptronOneHot(trainData,testData)

#lenetOneHot(trainData, testData)

#print trainData[0].shape
print len(trainData[1])
vals = [0.0, 0.0, 0.0, 0.0]
for truth in trainData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))

#print testData[0].shape#
print len(testData[1])
vals = [0.0, 0.0, 0.0, 0.0]
for truth in testData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))



perceptronOneHot(trainData,testData)
#
#clf = svm.SVC()
#clf.fit(trainData[0], trainData[1])  
#
#pred = clf.predict(testData[0])
#
#print np.mean(np.equal(testData[1],pred))

#print trainData[0][425].shape
#cv2.im#show('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()