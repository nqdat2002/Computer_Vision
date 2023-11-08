import cv2 as cv
img = cv.imread('img1.png', cv.IMREAD_GRAYSCALE)
star = cv.xfeatures2d.StarDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp = star.detect(img,None)
kp, des = brief.compute(img, kp)
print('Descriptor Size: ',brief.descriptorSize())
print('Descriptor Shape', des.shape)