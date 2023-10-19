import cv2 as cv
import numpy as np

# Erosion
img = cv.imread('img1.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

# Dilation
dilation = cv.dilate(img,kernel,iterations = 1)

# Opening
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Closing
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# Morphological Gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# Top Hat
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# Black Hat
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# Structuring Element

# Rectangular Kernel

# >>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# array([[1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel

# >>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# array([[0, 0, 1, 0, 0],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1],
#  [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel

# >>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
# array([[0, 0, 1, 0, 0],
#  [0, 0, 1, 0, 0],
#  [1, 1, 1, 1, 1],
#  [0, 0, 1, 0, 0],
#  [0, 0, 1, 0, 0]], dtype=uint8)
