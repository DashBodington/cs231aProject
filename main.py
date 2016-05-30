#import tensorflow as tf
from util import *
from learning import *
from alexnet import *
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

#Ensure a 50/50 class split in training and testing
evenData = True
#Perform classification with multiple algorithms
classify = False

featureType = "Image"
#featureType = "Alexnet"
#featureType = "colorHistogram"


#random.seed(0)

print "Loading data..."
if featureType == "Image":

	createNew = False
	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=64)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
		pickle.dump(testData, open("testdatasetimages64.p","wb"))
		pickle.dump(trainData, open("traindatasetimages64.p","wb"))
	#Load premade dataset
	testData = pickle.load(open("testdatasetimages64.p","rb"))
	trainData = pickle.load(open("traindatasetimages64.p","rb"))

elif featureType == "Alexnet":

	sparsify = False

	testData = pickle.load(open("testalexfeatsfc8.p","rb"))
	trainData = pickle.load(open("trainalexfeatsfc8.p","rb"))

	if(sparsify):
		tol = 0.01
		for im in testData[0]:
			im.real[abs(im) < tol] = 0.0
		for im in trainData[0]:
			im.real[abs(im) < tol] = 0.0
elif featureType == "colorHistogram":

	testData = pickle.load(open("testdatasetimages64.p","rb"))
	trainData = pickle.load(open("traindatasetimages64.p","rb"))
	
	testData = colorHist(testData)
	trainData = colorHist(trainData)
	


#print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

#trainData, testData = createDatasets(allImages, allLabels, 0.7)

#Save to file
#pickle.dump((trainData, testData), open("dataset.p","wb"))


#print testData[0][0].shape
#Learning methods
#Perceptron
#perceptronOneHot(trainData,testData)

#print trainData[0].shape

if(evenData):
	trainData = evenDataset(trainData)
	testData = evenDataset(testData)


print len(trainData[1]), "Train cases:",
vals = [0.0, 0.0]
for truth in trainData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))

#print testData[0].shape#
print len(testData[1]), "Test cases: ",
vals = [0.0, 0.0]
for truth in testData[1]:
	vals[int(truth)] += 1.0

print (np.array(vals)/np.sum(vals))




#pred = lenetOneHot(trainData, testData)

#Classifiers
if(classify):

	#perceptronOneHot(trainData,testData)

	print "SVM",
	#Scikit learn, svm
	clf = svm.SVC()
	clf.fit(trainData[0], trainData[1])  
	pred = clf.predict(testData[0])
	print np.mean(np.equal(testData[1],pred))

	print "Naive Bayes",
	#Scikit learn, naive bayes
	clf = GaussianNB()
	clf.fit(trainData[0], trainData[1])  
	pred = clf.predict(testData[0])
	print np.mean(np.equal(testData[1],pred))

	print "K Nearest Neighbor",
	#K nearest neighbor
	clf = neighbors.KNeighborsClassifier(20)
	clf.fit(trainData[0], trainData[1])  
	pred = clf.predict(testData[0])
	print np.mean(np.equal(testData[1],pred))

	print "LDA",
	#K nearest neighbor
	clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
	clf.fit(trainData[0], trainData[1])  
	pred = clf.predict(testData[0])
	print np.mean(np.equal(testData[1],pred))

#print trainData[0][425].shape#
#cv2.im#show('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()