import cv2 as cv
import numpy as np

# Changing Color-space
# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print( flags )

# Object Tracking
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

# How to find HSV values to track?
# green = np.uint8([[[0,255,0 ]]])
# hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
# print( hsv_green )
# [[[ 60 255 255]]]