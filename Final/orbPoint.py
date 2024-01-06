import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
bbox = (287, 23, 86, 320)
tracker = cv2.ORB_create()
tracker.setMaxFeatures(500)
tracker.setFastThreshold(0)
kp = tracker.detect(frame, None)
kp, des = tracker.compute(frame, kp)
while True:
    ret, frame = cap.read()
    kp = tracker.detect(frame, None)
    kp, des = tracker.compute(frame, kp)
    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
    cv2.imshow("ORB Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()