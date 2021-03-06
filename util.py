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
