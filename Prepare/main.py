import cv2 as cv
import  sys

img = cv.imread(cv.samples.findFile("img1.png"))
if img is None:
    sys.exit("Find Not Found and Could not read the image!!")

cv.imshow("Display Window", img)
k = cv.waitKey(0)

if k == ord('s'):
    print("You has pressed the key is S!")
    cv.imwrite("newCapture.JPG", img)

if k == ord('c'):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("Change to Color", img)