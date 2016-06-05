def createAlexnetFeats(trainData, testData,save=False):
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
    if(save):
        print "Saving alex feats...",
        pickle.dump(testData, open("testalexfeatsfc8.p","wb"))
        pickle.dump(trainData, open("trainalexfeatsfc8.p","wb"))
        print "Finished"
    return trainData, testData


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