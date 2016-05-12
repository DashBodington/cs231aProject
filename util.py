import os.path
import json
import unicodedata
import collections

#Make things print immediately
import os
import sys
unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = unbuffered

def loadData(filename, numLinesData, numLinesTest):

    if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

    print "Loading Data..."
    trainData = []
    testData = []
    lineNum = 0

    with open(filename) as f:
        for line in f:
            if lineNum < numLinesData:
                trainData.append(json.loads(line))
                lineNum += 1
            elif lineNum < numLinesData + numLinesTest:
                testData.append(json.loads(line))
                lineNum += 1
            else:
                break

    print "Done Loading Data!"

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