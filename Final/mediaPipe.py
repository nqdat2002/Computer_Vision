import cv2
cap = cv2.VideoCapture('./images/face01.jpg')
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker = cv2.TrackerMOSSE_create()
tracker.init(frame, bbox)
while True:
    ret, frame = cap.read()
    ret, bbox = tracker.update(frame)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
