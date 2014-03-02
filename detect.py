import cv2, glob, numpy
from scipy.cluster.vq import *
files=glob.glob('images/*.jpg')
labelFile=open('label.txt')
labels=dict()
patches=list()
count=0
for line in labelFile:
    if line.strip() not in labels.keys():
        labels[line.strip()]=len(labels.keys())+1
    count+=1
print count
sz=list()
for file in files:
    img=cv2.imread(file, 0)
    #img = cv2.imread('template.jpg', 0)

    sift = cv2.SIFT()
    kp = sift.detect(img,None)
    #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('sift_keypoints.jpg',img)
    for k in kp:
        #name=file.split('/')[len(file.split('/'))-1]
        #name=name.strip('.jpg')
        #c=name.split('_')[0]
        #k.class_id=labels[c]
        x, y=k.pt
        x, y=int(x), int(y)
        patch=numpy.array(img[x:x+5, y:y+5]).astype(int)
        n, m=patch.shape
        if n*m==25:
            patch=patch.reshape(1, n*m)
            patches.append(patch[0])
        sz.append(len(kp))

patches=numpy.array(patches)
print patches.shape
#numpy.savetxt('patches.txt', patches)
centroid, l=kmeans2(numpy.array(patches), 260)
print len(centroid)
numpy.savetxt('centroid.txt', centroid)
numpy.savetxt('kmeans_label.txt', l)
files=glob.glob('images/*.jpg')
count=0
#data=list()
#training data are put in train.txt
train=open('train.txt', 'w+')
for file in files:
    print file
    img=cv2.imread(file, 0)
    sift=cv2.SIFT()
    kp=sift.detect(img, None)
    idx=count
    sz=0
    for k in kp:
        x, y=k.pt
        x, y=int(x), int(y)
        patch=numpy.array(img[x:x+5, y:y+5]).astype(int)
        n, m=patch.shape
        if n*m==25:
            sz+=1
    count+=sz
    d=list()
    while idx<count:
        d.append(l[idx])
        idx+=1
    #data.append(d)
    for e in d:
        train.write(str(e)+',')
    train.write('\n')