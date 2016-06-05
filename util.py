import os.path
import json
import unicodedata
import collections
import pickle
import numpy as np
import cv2
import math
import random
from alexnet import *
from sklearn import neighbors, datasets
from sklearn.cluster import MiniBatchKMeans, KMeans

#Make things print immediately
import os
import sys
unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = unbuffered


def getImageData(filename, numLines):
    #Get data for images which are tagged as food
    lineNum = 0
    data = []
    #Load all lines (usually just one line)
    with open(filename) as f:
        for line in f:
            if lineNum < numLines:
                rawData =  json.loads(line)
                lineNum += 1
            else:
                break
    #Convert strings from unicode
    rawData = convertUnicode(rawData)

    #Pick out only the images tagged with food
    for imgData in rawData:
        if 'food' in imgData['label']:
            data.append(imgData)

    return data

def getBusinessData(filename):
    #Load businesses so that we can assign photos to them
    rawData = []
    #Load all lines (usually just one line)
    with open(filename) as f:
        for line in f:
            rawData.append(json.loads(line))

    #Convert strings from unicode
    data = convertUnicode(rawData)

    return data

def convertUnicode(data):
    #Recursively convert strings in data structure from unicode
    if isinstance(data, basestring):
        return unicodedata.normalize('NFKD', data).encode('ascii','ignore')
    elif isinstance(data, collections.Mapping):
        return dict(map(convertUnicode, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convertUnicode, data))
    else:
        return data

def prepImage(image, imsize=227):
    cs= int(math.floor(min(image.shape[0:2])/2))
    #print cs
    image = image[int(math.floor(image.shape[0]/2))-cs:int(math.floor(image.shape[0]/2))+cs, 
    int(math.floor(image.shape[1]/2))-cs:int(math.floor(image.shape[1]/2))+cs]
    img = cv2.resize(image,(imsize,imsize),interpolation = cv2.INTER_AREA)
    return img

def loadData(imageLabels, businessLabels, imageFolder, imsize = 227):
    print 'Loading Image Data...',
    imgData = getImageData(imageLabels, 1)
    numImages = len(imgData)
    print 'Finished'

    print 'Loading Business Data...',
    busData = getBusinessData(businessLabels)
    print 'Finished'

    #allData contains business-image list pairs
    print 'Grouping images by business',
    allData = {}
    for img in imgData:
        if not img['business_id'] in allData:
            allData[img['business_id']] = []
        allData[img['business_id']].append(img['photo_id'])
    print 'Finished'

    #allLabels contains business-cost pairs
    print 'Finding business labels...',
    allLabels = {}
    for bus in allData.keys():
        for datum in busData:
            #print datum
            if(bus == datum['business_id']):
                if 'Price Range' in datum['attributes'].keys():
                    allLabels[bus] = datum['attributes']['Price Range']
                    break
    print 'Finished'

    #remove businesses/images which don't have a price rating
    print 'Removing unlabeled data...',
    for bus in allData.keys():
        if not bus in allLabels.keys():
            del allData[bus]
    print 'Finished'

    "Resizing, cropping, and storing images...",
    allImages = {}
    for bus in allData.keys():
        for im in allData[bus]:
            name = imageFolder + im + '.jpg'
            img = cv2.imread(name,1)
            #Crop image to square
            cs= int(math.floor(min(img.shape[0:2])/2))
            img = img[int(math.floor(img.shape[0]/2))-cs:int(math.floor(img.shape[0]/2))+cs, 
            int(math.floor(img.shape[1]/2))-cs:int(math.floor(img.shape[1]/2))+cs]
            img = cv2.resize(img,(imsize,imsize),interpolation = cv2.INTER_AREA)
            if not bus in allImages.keys():
                allImages[bus] = []
            allImages[bus].append(img)
            #cv2.imshow('img',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    print 'Finished'

    #Save to file
    pickle.dump((allImages, allLabels), open("imageset.p","wb"))

    return allImages, allLabels

def createDatasets(allImages, allLabels, trainRatio, dataFraction):

    for bus in allLabels:
        if allLabels[bus] >= 3:
            allLabels[bus] = 1
        else:
            allLabels[bus]= 0

    #Bin data by price rating
    byPrice = {}
    for bus in allImages.keys():
        if not allLabels[bus] in byPrice.keys():
            byPrice[allLabels[bus]] = []
        byPrice[allLabels[bus]].append(bus)

    #Shuffle businesses within their bins
    for cost in byPrice.values():
        random.shuffle(cost)

    #find min number to make sampling more uniform
    minbus = 10e10
    for cost in byPrice.values():
        if(minbus > len(cost)):
            minbus = len(cost)


    #Split businesses into train and test
    trainBus = []
    testBus = []
    for cost in byPrice.values():
        for i in xrange(int(len(cost)*dataFraction)):
            if(i < trainRatio*len(cost)*dataFraction):
                #Training set
                trainBus.append(cost[i])
            else:
                #Test set
                testBus.append(cost[i])

    #Pack all the data into simpler vectors
    trainImages = []
    trainLabels = []
    for bus in trainBus:
        for im in allImages[bus]:
            trainImages.append(im)
            trainLabels.append(allLabels[bus])

    testImages = []
    testLabels = []
    for bus in testBus:
        #print len(allImages[bus])
        for im in allImages[bus]:
            testImages.append(im)
            testLabels.append(allLabels[bus])

    #Shuffle the training dataset
    ti2 = []
    tl2 = []
    inds = range(1,len(trainLabels))
    random.shuffle(inds)
    for i in inds:
        ti2.append(trainImages[i])
        tl2.append(trainLabels[i])
    trainImages = ti2
    trainLabels = tl2

    print len(trainLabels), "train images"
    print len(testLabels), "test images"

    #Vectorize images
    trainImages = np.reshape(np.array(trainImages),(len(trainImages),-1))
    testImages = np.reshape(np.array(testImages),(len(testImages),-1))


    #im = trainImages[1]
    #im = np.reshape(im, (128, 128, 3))
    #cv2.imshow('image',im)
    #cv2.waitKey()#


    return (trainImages, trainLabels), (testImages, testLabels)

def evenDataset(data):
    choseninds = []
    pos = 0
    neg = 0
    for i in xrange(len(data[1])):
        if(data[1][i] == 1):
            pos += 1
        else:
            neg += 1
    num = min(pos, neg)
    newIms = []
    newLabels = []
    pos = 0
    neg = 0
    for i in xrange(len(data[1])):
        if(data[1][i] == 1):
            if(pos < num):
                newIms.append(data[0][i])
                newLabels.append(data[1][i])
                pos += 1
                choseninds.append(i)
        else:
            if(neg < num):
                newIms.append(data[0][i])
                newLabels.append(data[1][i])
                neg += 1
                choseninds.append(i)
    return (newIms,newLabels), choseninds


def createAlexnetFeats(trainData, testData):
    #(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )
    #data = pickle.load( open( "dataset.p", "rb" ) )
    #trainData = pickle.load(open("traindataset.p","rb"))
    #testData = pickle.load(open("testdataset.p","rb"))
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

    

    trainData = (trainFeats,trainData[1])
    testData = (testFeats,testData[1])


    print len(trainFeats)
    print len(testFeats)
    print "Saving alex feats...",
    pickle.dump(testData, open("testalexfeatsfc8.p","wb"))
    pickle.dump(trainData, open("trainalexfeatsfc8.p","wb"))
    print "Finished"


def createAlexnetFeat(img):
    #(allImages, allLabels) = pickle.load( open( "imageset.p", "rb" ) )
    #data = pickle.load( open( "dataset.p", "rb" ) )
    #trainData = pickle.load(open("traindataset.p","rb"))
    #testData = pickle.load(open("testdataset.p","rb"))


    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]

    trainFeats = []
    testFeats = []

    Feats = alexnet(np.expand_dims(img,axis=0),verbose=False)


    return Feats


def colorHist(inData):
    imgs = inData[0]
    labels = inData[1]
    newFeats = []
    for im in imgs:
        ihist = np.zeros((30))
        for i in xrange(im.shape[0]):
            #print im.shape
            ibin = int(np.floor(float(im[i])/256.0*10))
            col = int(np.floor(2.99*float(i)/float(im.shape[0])))
            #print ibin
            #print col
            ihist[ibin + col*10] += 1.0
        ihist = ihist/np.sum(ihist)
        newFeats.append(ihist)
    return(newFeats,labels)

def surfBow(trainData, means=10, maxfeats=50):
    print "Finding BOW...",
    surf = cv2.xfeatures2d.SURF_create()
    #First find bag of words
    descriptors = np.zeros((len(trainData[0])*maxfeats,64))
    #For every image
    currentpos = 0
    for i in xrange(len(trainData[0])):
        #Reshape it properly
        img = np.reshape(trainData[0][i],(64,64,3))
        (kp, des) = surf.detectAndCompute(img, None)
        if not des is None:
            sz = min(des.shape[0],maxfeats-1)
            #print sz, currentpos, des.shape[0], descriptors[currentpos:currentpos + sz,:].shape, des[0:sz,:].shape
            #print descriptors.shape
            descriptors[currentpos:(currentpos + sz),:] = des[0:sz,:]
            currentpos += min(des.shape[0],maxfeats-1)
            #print des.shape
    descriptors = descriptors[0:currentpos,:]

    k_means = KMeans(n_clusters=10, n_init=1)
    print "Fit"
    k_means.fit(descriptors, y=None)
    bow = k_means.cluster_centers_
    print "Finished"
    return bow

def bowSurfFeats(data, bow):
    clusternum = range(0,bow.shape[0])
    clf = neighbors.KNeighborsClassifier(1)
    surf = cv2.xfeatures2d.SURF_create()
    clf.fit(bow, clusternum)  
    labels = data[1]
    newFeats = []
    for im in data[0]:
        img = np.reshape(im,(64,64,3))
        bowfeat = np.zeros(len(clusternum))
        kp, des = surf.detectAndCompute(img, None)
        if not des is None:
            for i in xrange(des.shape[0]):
                bowfeat[clf.predict(np.expand_dims(des[i], axis=0))] += 1
            bowfeat = bowfeat/np.sum(bowfeat)
        newFeats.append(bowfeat)

        #print newFeats.shape

    return (newFeats, labels)

def siftBow(trainData, means=10, maxfeats=50):
    print "Finding BOW...",
    sift = cv2.xfeatures2d.SIFT_create()
    #First find bag of words
    descriptors = np.zeros((len(trainData[0])*maxfeats,128))
    #For every image
    currentpos = 0
    for i in xrange(len(trainData[0])):
        #Reshape it properly
        img = np.reshape(trainData[0][i],(64,64,3))
        (kp, des) = sift.detectAndCompute(img, None)
        if not des is None:
            sz = min(des.shape[0],maxfeats-1)
            #print sz, currentpos, des.shape[0], descriptors[currentpos:currentpos + sz,:].shape, des[0:sz,:].shape
            #print des.shape, sz
            descriptors[currentpos:(currentpos + sz),:] = des[0:sz,:]
            currentpos += min(des.shape[0],maxfeats-1)
            #print des.shape
    descriptors = descriptors[0:currentpos,:]

    k_means = KMeans(n_clusters=10, n_init=1)
    print "Fit"
    k_means.fit(descriptors, y=None)
    bow = k_means.cluster_centers_
    print "Finished"
    return bow

def bowSiftFeats(data, bow):
    clusternum = range(0,bow.shape[0])
    clf = neighbors.KNeighborsClassifier(1)
    sift = cv2.xfeatures2d.SIFT_create()
    clf.fit(bow, clusternum)  
    labels = data[1]
    newFeats = []
    for im in data[0]:
        img = np.reshape(im,(64,64,3))
        bowfeat = np.zeros(len(clusternum))
        kp, des = sift.detectAndCompute(img, None)
        if not des is None:
            for i in xrange(des.shape[0]):
                bowfeat[clf.predict(np.expand_dims(des[i], axis=0))] += 1
            bowfeat = bowfeat/np.sum(bowfeat)
        newFeats.append(bowfeat)

        #print newFeats.shape

    return (newFeats, labels)