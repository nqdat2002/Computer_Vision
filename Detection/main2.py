import cv2
import numpy as np

cas_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") # make sure the folder containing *.xml is known to your IDE
cas_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 13
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_alt2 = cas_alt2.detectMultiScale(gray)
    newfr = frame.copy()
    for (x, y, w, h) in faces_alt2:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        newfr = frame[y:y + h, x:x + w]

    cv2.imshow('newframe', frame)
    keycode = cv2.waitKey(1)
    if keycode == 27:
        break
    if keycode == ord('s'):
        cv2.imwrite('imgtest' + str(i) + '.jpg', newfr)
        i += 1
        if i == 20: break
cap.release()
cv2.destroyAllWindows()