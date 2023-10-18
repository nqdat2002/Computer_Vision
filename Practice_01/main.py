import cv2 as cv

# cameraCapture = cv2.VideoCapture(0)
# success, frame = cameraCapture.read()
#
# while success:
#     cv2.imshow('My window', frame)
#     success, frame = cameraCapture.read()
#     keycode = cv2.waitKey(1)
#     if keycode != -1:
#         keycode &= 0xFF
#         if keycode == 27:
#             break
#         if keycode == 32:
#             cv2.imwrite('data/mypic01.jpg')
# cv2.destroyWindow('My window')
# cameraCapture.release()


def handlerLeftMouseClick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_TRIPLEX
        LB = 'LEFT Button'
        cv.putText(img, LB, (x, y), font, 1, (255, 255, 0), 2)
        cv.imshow('image', img)

img = cv.imread("img.png")
cv.imshow("TEST", img)
cv.setMouseCallback("TEST", handlerLeftMouseClick)
cv.waitKey(0)
cv.destroyAllWindows()
