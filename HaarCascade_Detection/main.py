import cv2
import numpy as np
# with img
img = cv2.imread("test_face_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Load cascade classifiers:
cas_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") # make sure the folder containing *.xml is known to your IDE
cas_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces:
faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "haarcascade_frontalface_alt2.xml")
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
retval, faces_haar_default = cv2.face.getFacesHAAR(img, "haarcascade_frontalface_default.xml")
faces_haar_default = np.squeeze(faces_haar_default)

fr = img.copy()
for (x, y, w, h) in faces_haar_alt2:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('detected',img)
cv2.waitKey(0)