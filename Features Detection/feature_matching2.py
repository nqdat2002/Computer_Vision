import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('img1.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('img2.png',cv.IMREAD_GRAYSCALE) # trainImage
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()