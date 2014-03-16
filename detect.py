import cv2, glob, numpy, shutil
from scipy.cluster.vq import *
from random import random
files=glob.glob('images/*.jpg')


patches=list()
count=0
labels=[line.strip() for line in open('label.txt')]
ann=[line.strip().split(',') for line in open('annotations.txt')]

count=0
sz=list()
filelist=list()
for file in files:
    rd=random()
    if rd>0.75:
        labels.pop(count)
        ann.pop(count)
        filename=file.split('/')[len(file.split('/'))-1]
        shutil.move(file, 'test/'+filename)
        continue
    filelist.append(file)
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
    count+=1
#save labels and ann
f=open('label1.txt', 'w+')
for e in labels:
    f.write(str(e)+'\n')
f.close()
f=open('annotations1.txt', 'w+')
for a in ann:
    for e in a:
        f.write(str(e)+'\n')
    f.write('\n')
f.close()


patches=numpy.array(patches)
print patches.shape
#numpy.savetxt('patches.txt', patches)
centroid, l=kmeans2(numpy.array(patches), 260)
print len(centroid)
numpy.savetxt('centroid.txt', centroid)
numpy.savetxt('kmeans_label.txt', l)

count=0
#data=list()
#training data are put in train.txt
train=open('train.txt', 'w+')
for file in filelist:
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
train.close()