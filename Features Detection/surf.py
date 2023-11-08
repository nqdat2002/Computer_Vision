import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('img1.png', cv.IMREAD_GRAYSCALE)
surf = cv.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(img, None)
print(len(kp))

print(surf.getHessianThreshold())
surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(img, None)
print(len(kp))

img2 = cv.drawKeypoints(img,kp, None, (255,0,0), 4)
plt.imshow(img2)
plt.show()