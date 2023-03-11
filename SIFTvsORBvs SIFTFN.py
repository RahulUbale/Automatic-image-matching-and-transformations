
import sys
import cv2
import numpy as np
from functools import reduce

orb = cv2.ORB_create(nfeatures=1000)
orb1 = cv2.SIFT_create(nfeatures=700)
img1 = cv2.imread('eiffel_18.jpg')
img2 = cv2.imread('eiffel_19.jpg')
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
   if m.distance < 0.73*n.distance:
       good.append([m])
imgo2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imwrite("imgo2.jpg", imgo2)
print(len(good))
kp1, des1 = orb1.detectAndCompute(img1, None)
kp2, des2 = orb1.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(des1, des2, k=3)
good1 = []
for m, n ,t in matches1:
      if m.distance < 0.75*n.distance and m.distance < 0.566*t.distance :
       good1.append([m])
print(len(good1))       
imgs2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good1, None, flags=2)
cv2.imwrite("imgs2.jpg", imgs2)



FLAN_INDEX_KDTREE = 0
index_params = dict (algorithm = FLAN_INDEX_KDTREE)
search_params = dict (checks=700)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches2 = flann.knnMatch (des1, des2,k=2)
good_matches = []

for m1, m2 in matches2:
  if m1.distance < 0.566 * m2.distance:
    good_matches.append([m1])

imgf1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imwrite("imgof1.jpg", imgf1)
print(len(good_matches))











































