import cv2
import numpy as np
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
image = cv2.imread('img.png')
oriImage = image.copy()
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
while True:
    img = image.copy()
    if not cropping:
        cv2.imshow("image", image)
    elif cropping:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", img)
        cv2.imwrite('cropped.jpg', img)
    keyCode = cv2.waitKey(0)
    if keyCode == 27:
        break

cv2.destroyAllWindows()