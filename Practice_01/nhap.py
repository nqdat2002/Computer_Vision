import cv2
url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"

img = cv2.imread(url_img)
cv2.imshow('image', img)
def mouse_click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_TRIPLEX
        LB = 'Left Button'
        cv2.putText(img, LB, (x, y),
                    font, 1,
                    (255, 255, 0),
                    2)
        cv2.imshow('image', img)


cv2.setMouseCallback('image', mouse_click)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002wlegend.jpg', img)
cv2.destroyAllWindows()