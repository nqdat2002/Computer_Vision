import cv2 as cv

tracker = cv.TrackerKCF_create()
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Can not get the capture")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can not get frame")
        exit()

    cv.imshow("Tracking", frame)
    if cv.waitKey(30) == ord("s"):
        break

bound_box = cv.selectROI(frame, False)
ret = tracker.init(frame, bound_box)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can not get frame")
        break

    ret, bound_box = tracker.update(frame)

    if ret:
        print(bound_box)
        p1 = (int(bound_box[0]), int(bound_box[1]))
        p2 = (int(bound_box[0] + bound_box[2]), int(bound_box[1] + bound_box[3]))
        cv.rectangle(frame, p1, p2, (0, 0, 255), 2, 2)

    cv.imshow("Tracking", frame)
    if cv.waitKey(30) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
