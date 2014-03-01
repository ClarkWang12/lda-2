import cv2, glob
import numpy as np

files=glob.glob('/Users/wenzhou/Documents/Workspace/lda/images/*.jpg')
for file in files:
    img=cv2.imread(file, 0)
    #img = cv2.imread('template.jpg', 0)

    sift = cv2.SIFT()
    kp = sift.detect(img,None)
    #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('sift_keypoints.jpg',img)
    for k in kp:
        name=file.split('/')[len(file.split('/'))-1]
        name=name.strip('.jpg')
        k.class_id=int(name.split('_')[1])
    
