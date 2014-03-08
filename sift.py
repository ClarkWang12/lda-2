import cv2
img=cv2.imread('coast.jpg', 0)
sift=cv2.SIFT()
kp=sift.detect(img, None)
img=cv2.drawKeypoints(img,kp)
cv2.imwrite('sift_keypoints.jpg',img)