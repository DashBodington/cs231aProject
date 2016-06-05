#import tensorflow as tf
from util import *
from learning import *
from alexnet import *
from featureExtractors import *
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, datasets, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#Define all the data we'll use
imageFolder = '/home/dash/Documents/yelpData/photos/'
dataFolder = '/home/dash/Documents/yelpData/data/'
imageLabels = imageFolder + 'photo_id_to_business_id.json'
businessLabels = dataFolder + 'yelp_academic_dataset_business.json'

#Ensure a 50/50 class split in training and testing
evenTest = True
evenTrain = True
#Perform classification with multiple algorithms
OTBclassify = True
#Do extra test with In-n-out and fancy spanish restaurant
extraTest = False
#Print top and bottom image numbers
printTop = False
#Perform dimensionality reduction of features using PCA to numComp components
pcaFeats = False
numComp = 100
createNew = False

#Select FeatureType
#featureType = "Image"
#featureType = "Alexnet"
#featureType = "colorHistogram"
#featureType = "SURFBOW"
#featureType = "SIFTBOW"
featureType = "multiFeat"

#
random.seed(0)

print "Loading data..."
if featureType == "Image":

	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=64)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
		pickle.dump(testData, open("testdatasetimages64.p","wb"))#
		pickle.dump(trainData, open("traindatasetimages64.p","wb"))
	#Load premade dataset
	testData = pickle.load(open("testdatasetimages64.p","rb"))
	trainData = pickle.load(open("traindatasetimages64.p","rb"))

elif featureType == "Alexnet":
	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=227)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
	#
		createAlexnetFeats(trainData, testData, save=True)

	sparsify = False
#
	testData = pickle.load(open("testalexfeatsfc8.p","rb"))
	trainData = pickle.load(open("trainalexfeatsfc8.p","rb"))
#
	if(sparsify):
		tol = 0.01
		for im in testData[0]:
			im.real[abs(im) < tol] = 0.0
		for im in trainData[0]:
			im.real[abs(im) < tol] = 0.0

elif featureType == "colorHistogram":
	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=64)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
		pickle.dump(testData, open("testdatasetimages64.p","wb"))#
		pickle.dump(trainData, open("traindatasetimages64.p","wb"))

		testData = pickle.load(open("testdatasetimages64.p","rb"))
		trainData = pickle.load(open("traindatasetimages64.p","rb"))
		#
		testData = colorHist(testData)
		trainData = colorHist(trainData)
		#
		pickle.dump(testData, open("testdatasetcolorhist.p","wb"))
		pickle.dump(trainData, open("traindatasetcolorhist.p","wb"))

	testData = pickle.load(open("testdatasetcolorhist.p","rb"))
	trainData = pickle.load(open("traindatasetcolorhist.p","rb"))#

elif featureType == "SURFBOW":
	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=64)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
		pickle.dump(testData, open("testdatasetimages64.p","wb"))#
		pickle.dump(trainData, open("traindatasetimages64.p","wb"))

		testData = pickle.load(open("testdatasetimages64.p","rb"))
		trainData = pickle.load(open("traindatasetimages64.p","rb"))
		bow = surfBow(trainData,means=20)
		print "Training feats"
		trainData = bowSurfFeats(trainData,bow)
		print "Test Feats"
		testData = bowSurfFeats(testData,bow)

		pickle.dump(testData, open("testdatasetsurfbow.p","wb"))
		pickle.dump(trainData, open("traindatasetsurfbow.p","wb"))

	testData = pickle.load(open("testdatasetsurfbow.p","rb"))
	trainData = pickle.load(open("traindatasetsurfbow.p","rb"))#

elif featureType == "SIFTBOW":
	if(createNew):
		testData = pickle.load(open("testdatasetimages64.p","rb"))
		trainData = pickle.load(open("traindatasetimages64.p","rb"))
		bow = siftBow(trainData,means=20)
		print "Training feats"
		trainData = bowSiftFeats(trainData,bow)
		print "Test Feats"
		testData = bowSiftFeats(testData,bow)	
		pickle.dump(testData, open("testdatasetsiftbow.p","wb"))
		pickle.dump(trainData, open("traindatasetsiftbow.p","wb"))

	testData = pickle.load(open("testdatasetsiftbow.p","rb"))
	trainData = pickle.load(open("traindatasetsiftbow.p","rb"))#

elif featureType == "multiFeat":
	if(createNew):
		allImages, allLabels = loadData(imageLabels, businessLabels, imageFolder, imsize=227)
		trainData, testData = createDatasets(allImages, allLabels, trainRatio=0.7, dataFraction=1.0)
		tr1, ts1 = createAlexnetFeats(trainData, testData)
		ts2 = colorHist(testData)
		tr2 = colorHist(trainData)
		#PCA the alexnet feats
		pca = PCA(n_components=100,whiten=True)
		pca.fit(tr1[0])
		tr1 = (pca.transform(tr1[0]), tr1[1])
		ts1 = (pca.transform(ts1[0]), ts1[1])
		#PCA the colorhist
		pca = PCA(n_components=30,whiten=True)
		pca.fit(tr2[0])
		tr2 = (pca.transform(tr2[0]), tr2[1])
		ts2 = (pca.transform(ts2[0]), ts2[1])
		
		trainFeats = []
		testFeats = []
		for i in xrange(len(tr1[0])):
			trainFeats.append(np.concatenate((tr1[0][i],tr2[0][i]),axis=0))
		for i in xrange(len(ts1[0])):
			testFeats.append(np.concatenate((ts1[0][i],ts2[0][i]),axis=0))
	
		trainData = (trainFeats, tr1[1])
		testData = (testFeats, ts1[1])
	
		pickle.dump(testData, open("testdatasetmultifeat.p","wb"))
		pickle.dump(trainData, open("traindatasetmultifeat.p","wb"))

	testData = pickle.load(open("testdatasetmultifeat.p","rb"))
	trainData = pickle.load( open("traindatasetmultifeat.p","rb"))

print "Finished."
#print "Finished loading", len(allImages), "businesses with", sum(len(ims) for ims in allImages.values()), "images."

#trainData, testData = createDatasets(allImages, allLabels, 0.7)

#Save to file
#pickle.dump((trainData, testData), open("dataset.p","wb"))


#print testData[0][0].shape
#Learning methods
#Perceptron
#perceptronOneHot(trainData,testData)

#print trainData[0].shape

if evenTrain:
	trainData, otherinds = evenDataset(trainData)
if evenTest:
	testData, choseninds = evenDataset(testData)


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

extraIms = []
if extraTest:
	extraIms.append(createAlexnetFeat(prepImage(cv2.imread("hd.jpg",1))))
	extraIms.append(createAlexnetFeat(prepImage(cv2.imread("fanciness.jpg",1))))
	#cv2.imshow('image',extraIms[0])
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#cv2.imshow('image',extraIms[1])
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#pred = lenetOneHot(trainData, testData)
if pcaFeats:
	pca = PCA(n_components=numComp,whiten=True)
	#print len(trainData[0]), trainData[0][0].shape
	pca.fit(trainData[0])
	trainData = (pca.transform(trainData[0]), trainData[1])
	testData = (pca.transform(testData[0]), testData[1])
	#print len(trainData[0]), trainData[0][0].shape
	for i in xrange(len(extraIms)):
		extraIms[i] = pca.transform(extraIms[i])


#Fit the one-layer neural net. No longer a perceptron after modifications, but the name remains
probs = perceptronOneHot2(trainData,testData, extraIms)

if printTop:
	rp1 = probs[:,0].argsort()[-5:][::-1]
	rp2 = probs[:,0].argsort()[:5][::-1]
	for i in rp1:
		print choseninds[i]

	for i in rp2:
		print choseninds[i]

#Classifiers
if(OTBclassify):

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
	#LDA
	clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
	clf.fit(trainData[0], trainData[1])  
	pred = clf.predict(testData[0])
	print np.mean(np.equal(testData[1],pred))

#print trainData[0][425].shape#
#cv2.im#show('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

