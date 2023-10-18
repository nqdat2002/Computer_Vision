import cv2
cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()
cnt = 5
while success:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()
    keycode = cv2.waitKey(1)
    if keycode != -1:
        keycode &= 0xFF
        if keycode == 27:
            break
        if keycode == ord('s'):
            cv2.imwrite('CVFALL2023B20DCPT053/CVFALL2023B20DCPT05300'+str(cnt)+'.jpg',frame)
            cnt += 1
cv2.destroyAllWindows()
cameraCapture.release()